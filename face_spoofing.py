from pathlib import Path
import json
import threading

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import streamlit as st
from torchvision import models, transforms

try:
    import av
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
    WEBRTC_AVAILABLE = True
    WEBRTC_IMPORT_ERROR = None
except Exception as exc:
    WEBRTC_AVAILABLE = False
    WEBRTC_IMPORT_ERROR = exc

# Supaya prediksi realtime di CPU tidak terlalu rakus thread.
torch.set_num_threads(1)
PREDICT_LOCK = threading.Lock()

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "ensemble_config.json"
MODEL1_PATH = BASE_DIR / "face_spoofing_model1.pth"
MODEL2_PATH = BASE_DIR / "face_spoofing_model2.pth"

# Jika nama file model berubah sedikit, aplikasi tetap mencoba mencari file terdekat.
def resolve_file(preferred_path: Path, pattern: str) -> Path:
    if preferred_path.exists():
        return preferred_path
    candidates = sorted(BASE_DIR.glob(pattern))
    if candidates:
        return candidates[0]
    return preferred_path

MODEL1_PATH = resolve_file(MODEL1_PATH, "face_spoofing_model1*.pth")
MODEL2_PATH = resolve_file(MODEL2_PATH, "face_spoofing_model2*.pth")

DISPLAY_LABELS = {
    "realperson": "Real Person / Asli",
    "fake_mask": "Fake Mask",
    "fake_mannequin": "Fake Mannequin",
    "fake_printed": "Fake Printed",
    "fake_screen": "Fake Screen",
    "fake_unknown": "Fake Unknown",
}


def pretty_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def safe_torch_load(path: Path, map_location: str):
    """Kompatibel untuk beberapa versi PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(obj):
    """Menerima state_dict langsung atau checkpoint dict."""
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break

    cleaned = {}
    for key, value in obj.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned[new_key] = value
    return cleaned


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    try:
        model = models.efficientnet_b0(weights=None)
    except TypeError:
        model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet18(num_classes: int) -> nn.Module:
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@st.cache_resource(show_spinner="Memuat model, karena rupanya dua jaringan saraf belum cukup dramatis...")
def load_ensemble():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {CONFIG_PATH}")
    if not MODEL1_PATH.exists():
        raise FileNotFoundError(f"File model 1 tidak ditemukan: {MODEL1_PATH}")
    if not MODEL2_PATH.exists():
        raise FileNotFoundError(f"File model 2 tidak ditemukan: {MODEL2_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    label_mapping = config["label_mapping"]
    index_to_label = {index: label for label, index in label_mapping.items()}
    labels = [index_to_label[index] for index in sorted(index_to_label)]
    num_classes = len(labels)
    img_size = int(config.get("img_size", 224))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model1 = build_efficientnet_b0(num_classes)
    model2 = build_resnet18(num_classes)

    state_dict1 = extract_state_dict(safe_torch_load(MODEL1_PATH, map_location=device))
    state_dict2 = extract_state_dict(safe_torch_load(MODEL2_PATH, map_location=device))

    model1.load_state_dict(state_dict1, strict=True)
    model2.load_state_dict(state_dict2, strict=True)

    model1.to(device).eval()
    model2.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Jika saat training Anda memakai mean/std berbeda, ubah bagian ini.
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    weights = (
        float(config.get("model1_weight", 0.5)),
        float(config.get("model2_weight", 0.5)),
    )

    return {
        "config": config,
        "device": device,
        "model1": model1,
        "model2": model2,
        "labels": labels,
        "weights": weights,
        "preprocess": preprocess,
    }


def prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


@torch.inference_mode()
def predict_pil(image: Image.Image):
    bundle = load_ensemble()
    image = prepare_image(image)
    tensor = bundle["preprocess"](image).unsqueeze(0).to(bundle["device"])

    with PREDICT_LOCK:
        logits1 = bundle["model1"](tensor)
        logits2 = bundle["model2"](tensor)
        prob1 = F.softmax(logits1, dim=1)
        prob2 = F.softmax(logits2, dim=1)
        w1, w2 = bundle["weights"]
        probs = (w1 * prob1 + w2 * prob2).squeeze(0).detach().cpu().numpy()

    best_index = int(np.argmax(probs))
    best_label = bundle["labels"][best_index]
    confidence = float(probs[best_index])
    prob_dict = {bundle["labels"][i]: float(probs[i]) for i in range(len(probs))}
    return best_label, confidence, prob_dict


def show_prediction_result(image: Image.Image):
    st.image(image, caption="Gambar yang diprediksi", use_container_width=True)

    label, confidence, prob_dict = predict_pil(image)
    is_real = label == "realperson"

    if is_real:
        st.success(f"Prediksi: **{pretty_label(label)}** | Confidence: **{confidence:.2%}**")
    else:
        st.error(f"Prediksi: **{pretty_label(label)}** | Confidence: **{confidence:.2%}**")

    chart_data = {
        pretty_label(label_name): probability
        for label_name, probability in sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    }
    st.bar_chart(chart_data)

    with st.expander("Lihat probabilitas detail"):
        st.table({"Kelas": list(chart_data.keys()), "Probabilitas": [f"{v:.4f}" for v in chart_data.values()]})


if WEBRTC_AVAILABLE:
    class FaceSpoofingVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.predict_every_n_frames = 5
            self.last_label = "-"
            self.last_confidence = 0.0

        def recv(self, frame):
            bgr = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            if self.frame_count % self.predict_every_n_frames == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                try:
                    label, confidence, _ = predict_pil(pil_image)
                    self.last_label = label
                    self.last_confidence = confidence
                except Exception as exc:
                    self.last_label = f"Error: {exc}"
                    self.last_confidence = 0.0

            if self.last_label == "realperson":
                color = (0, 180, 0)
                status = "REAL"
            elif self.last_label.startswith("Error"):
                color = (0, 165, 255)
                status = "ERROR"
            else:
                color = (0, 0, 220)
                status = "SPOOF"

            text = f"{status} | {pretty_label(self.last_label)} | {self.last_confidence:.1%}"
            cv2.rectangle(bgr, (10, 10), (min(900, 10 + len(text) * 14), 60), color, -1)
            cv2.putText(
                bgr,
                text,
                (20, 47),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(bgr, format="bgr24")


def main():
    st.set_page_config(
        page_title="Face Spoofing Detection",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Face Spoofing Detection")
    st.caption("Ensemble EfficientNet-B0 + ResNet18 untuk klasifikasi real person dan spoofing.")

    try:
        bundle = load_ensemble()
    except Exception as exc:
        st.error(f"Model gagal dimuat: {exc}")
        st.stop()

    config = bundle["config"]
    with st.sidebar:
        st.header("Info Model")
        st.write(f"Device: `{bundle['device']}`")
        st.write(f"Model 1: `{config.get('model1_name', 'Model 1')}`")
        st.write(f"Model 2: `{config.get('model2_name', 'Model 2')}`")
        st.write(f"Input size: `{config.get('img_size', 224)} x {config.get('img_size', 224)}`")
        st.write(f"Bobot model 1: `{float(config.get('model1_weight', 0.5)):.4f}`")
        st.write(f"Bobot model 2: `{float(config.get('model2_weight', 0.5)):.4f}`")
        st.divider()
        st.caption("Catatan: kamera browser biasanya butuh izin akses kamera. Di deployment publik, pakai HTTPS, karena browser suka aturan. Sangat mengejutkan.")

    mode = st.radio(
        "Pilih metode prediksi",
        ["Upload file gambar", "Kamera realtime"],
        horizontal=True,
    )

    if mode == "Upload file gambar":
        uploaded_file = st.file_uploader(
            "Upload gambar wajah",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            show_prediction_result(image)
        else:
            st.info("Upload gambar terlebih dahulu untuk memulai prediksi.")

    else:
        st.subheader("Kamera realtime")
        st.write("Klik **Start**, izinkan akses kamera di browser, lalu arahkan wajah ke kamera.")

        if not WEBRTC_AVAILABLE:
            st.error(
                "Package `streamlit-webrtc` atau `av` belum terpasang. "
                "Install dulu dengan: `pip install streamlit-webrtc av`"
            )
            if WEBRTC_IMPORT_ERROR:
                st.code(str(WEBRTC_IMPORT_ERROR))
            st.stop()

        webrtc_streamer(
            key="face-spoofing-realtime",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=FaceSpoofingVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )


if __name__ == "__main__":
    main()

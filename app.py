# 🌿 Prompt-Based Crop Disease Identifier using FastAI + Streamlit
# -------------------------------------------------------------------------
# Run with:
#   streamlit run app.py

# ------------------- Fix WindowsPath on Linux/Streamlit -------------------
import pathlib
if not hasattr(pathlib, 'WindowsPath'):
    pathlib.WindowsPath = pathlib.PosixPath
# -------------------------------------------------------------------------

from fastai.vision.all import *
import streamlit as st
from pathlib import Path
import PIL.Image

# ------------------- Configuration -------------------
MODEL_PATH = Path("tomato_disease_model.pkl")  # OS-independent path

# ------------------- Load Model (Cached for Streamlit) -------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("❌ Model file not found. Please train it first using FastAI.")
        st.stop()
    try:
        # Load model on CPU
        learn = load_learner(str(MODEL_PATH), cpu=True)
        st.success("✅ Model loaded successfully!")
        return learn
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# ------------------- Cure Suggestion Logic -------------------
def get_cure_suggestion(disease_name: str):
    disease = disease_name.lower()
    cures = {
        "healthy": "✅ The leaf is healthy. No treatment required.",
        "bacterial spot": (
            "⚠ *Bacterial Spot Detected!*\n"
            "- 💊 Apply copper-based fungicides (copper hydroxide).\n"
            "- 🌿 Avoid overhead watering, improve air circulation."
        ),
        "leaf mold": (
            "⚠ *Leaf Mold Detected!*\n"
            "- 💊 Use sulfur-based fungicides.\n"
            "- 🌿 Reduce humidity and increase air ventilation."
        ),
        "early blight": (
            "⚠ *Early Blight Detected!*\n"
            "- 💊 Use mancozeb or chlorothalonil-based fungicides.\n"
            "- 🌿 Remove infected leaves and rotate crops."
        ),
        "late blight": (
            "⚠ *Late Blight Detected!*\n"
            "- 💊 Apply copper sulfate or metalaxyl fungicides.\n"
            "- 🔥 Destroy infected plants to stop spread."
        ),
        "septoria leaf spot": (
            "⚠ *Septoria Leaf Spot Detected!*\n"
            "- 💊 Use fungicides with chlorothalonil or mancozeb.\n"
            "- 🌿 Prune lower leaves and maintain spacing."
        ),
    }

    for key, suggestion in cures.items():
        if key in disease:
            return suggestion
    return "⚠ Unknown disease. Please verify the dataset labels or retrain model."

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="🌾 Crop Disease Identifier", layout="wide")
st.title("🌿 Prompt-Based Crop Disease Identifier")
st.markdown("### Identify plant leaf diseases using a trained FastAI model")

# Load model
learn = load_model()

# Input prompt
prompt = st.text_input("💬 Enter your prompt (e.g., 'Identify the disease in this tomato leaf')")

uploaded_file = st.file_uploader("📸 Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file and prompt:
    img = PILImage.create(uploaded_file)
    st.image(img.to_thumb(400, 400), caption="Uploaded Image", use_container_width=False)

    # Model prediction
    with st.spinner("🔍 Analyzing image..."):
        pred_class, pred_idx, probs = learn.predict(img)

    # Clean prediction label
    clean_name = str(pred_class).replace("_", " ").strip()
    if "_" in clean_name:
        clean_name = clean_name.split("_")[-1]

    # Display prediction
    st.subheader("🩺 Prediction Results")
    st.success(f"*Predicted Disease:* {clean_name.title()}")
    st.info(f"*Confidence:* {probs[pred_idx]:.2%}")

    # Cure suggestion
    st.subheader("💊 Suggested Cure")
    st.markdown(get_cure_suggestion(clean_name))

else:
    st.warning("💡 Please enter a prompt and upload an image to begin prediction.")

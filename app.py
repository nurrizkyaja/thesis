import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "models/convnext_tiny_finetuned/best_model.pth"

def replace_last_linear(module: nn.Module, out_features: int):
    last_linear = None
    last_parent = None
    last_name = None

    for parent in module.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                last_linear = child
                last_parent = parent
                last_name = name

    if last_linear is not None:
        in_features = last_linear.in_features
        setattr(last_parent, last_name, nn.Linear(in_features, out_features))

def strip_module_prefix(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

@st.cache_resource
def load_model():
    model = models.convnext_tiny(weights=None)
    
    replace_last_linear(model.classifier, out_features=NUM_CLASSES)
    
    device = torch.device("cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model

def transform_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return img_transform(image).unsqueeze(0)

st.set_page_config(page_title="Klasifikasi Sampah", page_icon="♻️", layout="centered")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file {MODEL_PATH} ada. Error: {e}")
    model_loaded = False

st.title("♻️ Klasifikasi Sampah")
st.markdown("---")

st.subheader("1. Unggah Foto Sampah")
uploaded_file = st.file_uploader("Tarik dan lepas gambar di sini, atau klik untuk memilih gambar", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang akan diklasifikasi", use_container_width=True)
    
    st.markdown("---")
    st.subheader("2. Hasil Klasifikasi")
    
    with st.spinner("Sedang memproses gambar..."):
        input_tensor = transform_image(image)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_idx = torch.argmax(outputs, dim=1).item()
            predicted_class = CLASS_NAMES[predicted_idx]
        
    result_card_html = f"""
    <div style="
        background-color: #f0f2f6; 
        border-radius: 10px; 
        padding: 20px; 
        border-left: 8px solid #28a745;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 10px;">
        <p style="margin: 0; font-size: 16px; color: #555;">Kategori Sampah Terdeteksi:</p>
        <h2 style="margin: 5px 0 0 0; color: #155724; text-transform: uppercase;">{predicted_class}</h2>
    </div>
    """
    st.markdown(result_card_html, unsafe_allow_html=True)
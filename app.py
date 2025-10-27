# ==========================
# 🌲 Forest Health Detector - Streamlit App (using Pickle model)
# ==========================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------
# 1️⃣ U-Net Model Definition (same as training)
# --------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class SmallUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bott = DoubleConv(64, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)
        self.up2  = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec2 = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, n_classes, 1)
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool(x1); x2 = self.enc2(x2)
        x3 = self.pool(x2); x3 = self.bott(x3)
        x = self.up1(x3); x = torch.cat([x, x2], dim=1); x = self.dec1(x)
        x = self.up2(x); x = torch.cat([x, x1], dim=1); x = self.dec2(x)
        return self.outc(x)

# --------------------------
# 2️⃣ Load Pickle Model
# --------------------------
with open("unet_eurosat_all.pkl", "rb") as f:
    model = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --------------------------
# 3️⃣ Helper Functions
# --------------------------
IMG_SIZE = 128
FOREST_CLASS = 1

def predict_image(image, model):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    img_tensor = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = output.argmax(dim=1).cpu().numpy()[0]

    return pred_mask

def analyze_forest(mask):
    forest_pixels = np.sum(mask == FOREST_CLASS)
    total_pixels = mask.size
    forest_percent = (forest_pixels / total_pixels) * 100

    if forest_percent == 0:
        return "🟥 Completely Deforested", forest_percent
    elif forest_percent < 30:
        return "🟧 Under Deforestation", forest_percent
    else:
        return "🟩 Healthy Forest", forest_percent

def visualize(image, mask):
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[mask == FOREST_CLASS] = [34, 139, 34]  # Forest = Green
    mask_rgb[mask != FOREST_CLASS] = [200, 200, 200]  # Non-Forest = Gray

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image); ax[0].set_title("Uploaded Image"); ax[0].axis("off")
    ax[1].imshow(mask_rgb); ax[1].set_title("Predicted Forest Map"); ax[1].axis("off")
    st.pyplot(fig)

# --------------------------
# 4️⃣ Streamlit Interface
# --------------------------
st.set_page_config(page_title="🌲 Forest Health Detector", layout="wide")
st.title("🌳 AI-Powered Forest Health Detector")
st.markdown("Upload a satellite image to detect **deforestation and forest health** using a trained U-Net model.")

uploaded_file = st.file_uploader("📤 Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze Forest Health"):
        st.write("⏳ Processing image...")
        mask = predict_image(image, model)
        status, forest_percent = analyze_forest(mask)
        st.success(f"🌿 Forest Status: {status}")
        st.write(f"🌍 Forest Coverage: **{forest_percent:.2f}%**")
        visualize(image, mask)

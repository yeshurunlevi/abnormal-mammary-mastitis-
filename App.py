#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm import tqdm
import torch.nn as nn

# === MODEL DEFINITION ===
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, 2, 1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# === SETUP ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 128
STRIDE = 256
BATCH_SIZE = 16
transform = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor()])

# === LOAD MODEL + BASELINE ===
@st.cache_resource
def load_model_and_baseline(model_path, baseline_path):
    model = Autoencoder().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    with open(baseline_path, "r") as f:
        baseline = float(f.read().strip())
    return model, baseline

# === PATCH SCORING ===
def compute_patch_scores(folder, model, baseline):
    results = []
    files = glob.glob(os.path.join(folder, "*.png"))
    batch, coords = [], []

    for f in tqdm(files, desc=f"Scoring {os.path.basename(folder)}"):
        fname = os.path.basename(f)
        slide = fname.split("_")[0]
        x, y = int(fname.split("_")[1]), int(fname.split("_")[2].split(".")[0])
        try:
            img = Image.open(f).convert("RGB")
        except Exception as e:
            continue
        tensor = transform(img)
        batch.append(tensor)
        coords.append((slide, x, y))

        if len(batch) == BATCH_SIZE:
            batch_tensor = torch.stack(batch).to(DEVICE)
            with torch.no_grad():
                recon = model(batch_tensor)
                rmse = torch.sqrt(torch.mean((batch_tensor - recon) ** 2, dim=(1, 2, 3))).cpu().numpy()
            for (slide_id, x, y), s in zip(coords, rmse):
                results.append((slide_id, x, y, s))
            batch, coords = [], []

    if batch:
        batch_tensor = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            recon = model(batch_tensor)
            rmse = torch.sqrt(torch.mean((batch_tensor - recon) ** 2, dim=(1, 2, 3))).cpu().numpy()
        for (slide_id, x, y), s in zip(coords, rmse):
            results.append((slide_id, x, y, s))

    return results

# === MAIN STREAMLIT UI ===
st.title("Mammary Gland Anomaly Detector (Autoencoder)")

input_folder = st.text_input("📂 Input tiles folder path")
output_folder = st.text_input("💾 Output folder path")
model_path = st.text_input("🧠 Path to trained model (.pth)", value="PATH/TO/autoencoder.pth")
baseline_path = st.text_input("📏 Path to test baseline txt", value="PATH/TO/baseline_test_rmse.txt")

if st.button("🔍 Run Analysis"):
    if not os.path.exists(input_folder) or not os.path.exists(output_folder):
        st.error("Please check folder paths.")
    else:
        os.makedirs(output_folder, exist_ok=True)
        heatmap_dir = os.path.join(output_folder, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)

        model, baseline = load_model_and_baseline(model_path, baseline_path)
        results = compute_patch_scores(input_folder, model, baseline)

        df = pd.DataFrame(results, columns=["Slide", "X", "Y", "RMSE"])
        df["Norm_RMSE"] = df["RMSE"] / baseline
        df["Abnormal"] = df["Norm_RMSE"] > 1.0

        summary = df.groupby("Slide")["Abnormal"].mean().reset_index()
        summary.rename(columns={"Abnormal": "Abnormal_Pct"}, inplace=True)
        csv_path = os.path.join(output_folder, "anomalies.csv")
        summary.to_csv(csv_path, index=False)
        st.success(f"✅ CSV saved to {csv_path}")

        for slide_id, group_df in df.groupby("Slide"):
            xs, ys, scores = group_df["X"].values, group_df["Y"].values, group_df["Norm_RMSE"].values
            grid_w, grid_h = (max(xs) // STRIDE) + 1, (max(ys) // STRIDE) + 1
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)

            for x, y, s in zip(xs, ys, norm_scores):
                i, j = y // STRIDE, x // STRIDE
                color = (np.array(plt.cm.jet(s)[:3]) * 255).astype(np.uint8)
                grid[i, j] = color

            heatmap = cv2.resize(grid, (grid_w * 10, grid_h * 10), interpolation=cv2.INTER_NEAREST)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(heatmap)
            ax.set_title(f"{slide_id} Heatmap")
            ax.axis('off')
            sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=np.min(scores), vmax=np.max(scores)))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            path = os.path.join(heatmap_dir, f"{slide_id}_heatmap.png")
            fig.savefig(path, bbox_inches='tight')
            plt.close()
            st.image(path, caption=f"{slide_id} Heatmap", use_column_width=True)

        st.success("🎉 Analysis complete!")


# In[2]:


streamlit run app.py


# In[ ]:





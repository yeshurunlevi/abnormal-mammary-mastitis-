# -*- coding: utf-8 -*-
"""Untitled.

Original file is located at
    https://colab.research.google.com/drive/1AVU_6cntooesUteyUppHPQvtYjBSOG4P
"""

import streamlit as st
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm import tqdm
import openslide
import tempfile
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte

# === SETTINGS ===
PATCH_SIZE = 512
STRIDE = 256
LEVEL = 0
INPUT_SIZE = 128
BLACK_THRESHOLD = 0.8
WHITE_THRESHOLD = 0.8
ENTROPY_THRESHOLD = 3.0
VARIANCE_THRESHOLD = 50.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODEL ===
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

@st.cache_resource
def load_model(model_path):
    model = Autoencoder().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

@st.cache_data
def load_baseline(baseline_path):
    with open(baseline_path, "r") as f:
        return float(f.read().strip())

# === UTILITIES ===
def is_good_patch(patch_np):
    gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
    total = gray.size
    if np.sum(gray < 20) / total > BLACK_THRESHOLD: return False
    if np.sum(gray > 235) / total > WHITE_THRESHOLD: return False
    if np.var(gray) < VARIANCE_THRESHOLD: return False
    if np.mean(entropy(img_as_ubyte(gray), disk(5))) < ENTROPY_THRESHOLD: return False
    return True

def extract_patches(slide_path, output_dir, slide_id):
    slide = openslide.OpenSlide(slide_path)
    w, h = slide.level_dimensions[LEVEL]
    count = 0
    for y in range(0, h - PATCH_SIZE, STRIDE):
        for x in range(0, w - PATCH_SIZE, STRIDE):
            patch = slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            patch_np = np.array(patch)
            if is_good_patch(patch_np):
                fname = f"{slide_id}_{x}_{y}.png"
                cv2.imwrite(os.path.join(output_dir, fname), cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR))
                count += 1
    return count

def compute_patch_scores(folder, model, transform):
    results = []
    files = glob.glob(os.path.join(folder, "*.png"))
    batch, coords = [], []
    for f in tqdm(files):
        fname = os.path.basename(f)
        slide, x, y = fname.split("_")[0], int(fname.split("_")[1]), int(fname.split("_")[2].split(".")[0])
        img = Image.open(f).convert("RGB")
        tensor = transform(img)
        batch.append(tensor)
        coords.append((slide, x, y))
        if len(batch) == 16:
            rmse = batch_rmse(model, batch)
            results += [(s, x, y, r) for (s, x, y), r in zip(coords, rmse)]
            batch, coords = [], []
    if batch:
        rmse = batch_rmse(model, batch)
        results += [(s, x, y, r) for (s, x, y), r in zip(coords, rmse)]
    return results

def batch_rmse(model, batch):
    with torch.no_grad():
        batch_tensor = torch.stack(batch).to(DEVICE)
        recon = model(batch_tensor)
        return torch.sqrt(torch.mean((batch_tensor - recon) ** 2, dim=(1, 2, 3))).cpu().numpy()

def save_heatmaps(df, output_folder, stride=STRIDE):
    os.makedirs(output_folder, exist_ok=True)
    for slide_id, group in df.groupby("Slide"):
        xs, ys, scores = group["X"].values, group["Y"].values, group["Norm_RMSE"].values
        norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
        grid_w = (max(xs) // stride) + 1
        grid_h = (max(ys) // stride) + 1
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for x, y, s in zip(xs, ys, norm_scores):
            i, j = y // stride, x // stride
            color = (np.array(plt.cm.jet(s)[:3]) * 255).astype(np.uint8)
            grid[i, j] = color
        heatmap = cv2.resize(grid, (grid_w*10, grid_h*10), interpolation=cv2.INTER_NEAREST)
        fig, ax = plt.subplots()
        ax.imshow(heatmap)
        ax.axis('off')
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=min(scores), vmax=max(scores)))
        sm.set_array([])
        fig.colorbar(sm, ax=ax)
        fig.savefig(os.path.join(output_folder, f"{slide_id}_heatmap.png"), bbox_inches='tight')
        plt.close()

# === STREAMLIT INTERFACE ===
st.title("Mammary Scan Anomaly Detector")

model_path = st.file_uploader("Upload trained Autoencoder model (.pth)", type="pth")
baseline_path = st.file_uploader("Upload baseline RMSE file", type="txt")

scan_dirs = st.text_input("Enter MRXS scan folder paths separated by semicolons (;)")
output_folder = st.text_input("Select output folder", value=tempfile.gettempdir())

if st.button("Run Analysis") and model_path and baseline_path and scan_dirs:
    model = load_model(model_path)
    baseline = load_baseline(baseline_path)
    transform = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor()])
    output_csv = os.path.join(output_folder, "anomaly_summary.csv")
    output_tiles = os.path.join(output_folder, "tiles")
    os.makedirs(output_tiles, exist_ok=True)

    all_results = []
    for folder in scan_dirs.split(";"):
        slide_files = glob.glob(os.path.join(folder.strip(), "*.mrxs"))
        for slide_path in slide_files:
            slide_id = os.path.splitext(os.path.basename(slide_path))[0]
            count = extract_patches(slide_path, output_tiles, slide_id)
            st.write(f"{slide_id}: {count} tiles extracted.")

    st.write("Scoring patches...")
    patch_scores = compute_patch_scores(output_tiles, model, transform)
    df = pd.DataFrame(patch_scores, columns=["Slide", "X", "Y", "RMSE"])
    df["Norm_RMSE"] = df["RMSE"] / baseline
    df["Abnormal"] = df["Norm_RMSE"] > 1.0

    summary = df.groupby("Slide")["Abnormal"].mean().reset_index()
    summary.columns = ["Scan", "Anomaly_Pct"]
    summary.to_csv(output_csv, index=False)
    save_heatmaps(df, os.path.join(output_folder, "heatmaps"))

    st.success("✅ Done!")
    st.write("CSV saved to:", output_csv)
    st.dataframe(summary)

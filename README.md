# abnormal-mammary-mastitis-
abnormal mammary (mastitis)


---

# Mammary Gland Anomaly Detection using Autoencoder

This project provides an end-to-end deep learning pipeline to detect anomalies in mammary gland histological scans. It includes tools to preprocess full-slide scans, train an Autoencoder for unsupervised anomaly detection, generate heatmaps of pixel-wise errors, and visualize abnormal regions.

---

## 📂 Repository Structure

```
├── App.py                          # Streamlit-based browser app for anomaly detection
├── autoencoder.pth                # Pretrained Autoencoder model
├── baseline_test_rmse.txt         # Baseline RMSE from normal test set
├── Cutting, downloading and filtering tiles.ipynb     # Patch extractor for .mrxs scans
├── Training and saving an autoencoder algorithm.ipynb # Autoencoder training script
├── Processing and output.ipynb    # Generate heatmaps and anomaly maps from tiles
├── test_heatmaps/                 # Heatmap results on the test set
├── twenty_four_heatmaps/         # Heatmap results on infected samples
```

---

## What This Project Does

1. **Tile Extraction**: Converts full `.mrxs` histology scans into small filtered 128×128 image tiles.
2. **Model Training**: Trains a convolutional Autoencoder on normal patches to learn healthy patterns.
3. **Anomaly Detection**: Computes reconstruction errors (RMSE) for new samples and compares them to a baseline from normal tissue.
4. **Visualization & Output**:

   * Generates heatmaps showing anomaly scores across each scan.
   * Outputs a CSV file summarizing the average anomaly score per scan, relative to the test baseline.

---

## What the Application Outputs

For each selected scan folder:

* A heatmap image per scan visualizing localized anomaly levels.
* A CSV file listing:

  * Scan filename
  * Average RMSE anomaly
  * Relative anomaly ratio (compared to test set mean RMSE)

Example:

```csv
scan_id,average_rmse,relative_anomaly
twenty_four_001.mrxs,0.0241,1.87
twenty_four_002.mrxs,0.0263,2.04
...
```

---

## How to Run the Streamlit Application

### 1. Install


### 2. Start the app

```bash
streamlit run App.py
```

### 3. In the App UI:

* Select one or more folders containing `.mrxs` scan files.
* Select an output folder where:

  * Heatmap images will be saved
  * A summary CSV file will be generated

---

## Example Outputs

* `test_heatmaps/`: Heatmaps from normal test scans.
* `twenty_four_heatmaps/`: Heatmaps from scans taken 24 hours post-infection.
* `summary.csv`: CSV file with average anomaly per scan and ratio to baseline.

---

## Notes

* The Autoencoder was trained only on normal samples.
* Reconstruction error (RMSE) is used to flag abnormal regions.
* The CSV result helps quantify inflammation levels over time or between groups.
* Patch filtering ensures only meaningful tiles are analyzed.

---

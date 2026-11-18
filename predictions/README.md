# Precomputed 3D Swin-UNETR Predictions

This folder stores precomputed discrete 3D segmentation masks predicted by the 3D Swin-UNETR model for the preloaded patient scans.

## Instant Cache Feature
The full-stack clinical studio (`app.py`) checks this folder to serve predictions in **< 1.2 seconds** without having to re-run neural network inference every time.

## Discrete Label Mapping & Jet Colormap Synchronization

Each predicted volume is saved as a single-channel 3D integer mask (`uint8`) with the following strict anatomical hierarchy:

| Integer Value | Colormap Color | Anatomical Region | Biological Definition |
| :---: | :---: | :--- | :--- |
| **0** | *Transparent* | Normal Brain Tissue / Background | Healthy neural tissue and background air |
| **1** | **🔵 Blue** | **Necrotic Core (NCR)** | Dead, avascular, ischemic tissue in the geometric center of the tumor |
| **2** | **🟢 Green** | **Peritumoral Edema (ED)** | Vasogenic fluid swelling and infiltrative non-enhancing glioma cells on the periphery |
| **3** | **🔴 Red** | **Enhancing Tumor (ET)** | Viable, hyper-vascularized proliferating rim with compromised Blood-Brain Barrier |

## Clinical Targets

* **Tumor Core (TC):** Label 1 + Label 3 ($\text{Necrotic Core} + \text{Enhancing Rim}$) — the neurosurgical resection target.
* **Whole Tumor (WT):** Label 1 + Label 2 + Label 3 ($\text{Core} + \text{Edema}$) — the radiation planning target (GTV/CTV).

# Preloaded Multimodal Patient Scans

This folder contains preprocessed 4-channel NIfTI (`.nii.gz`) brain MRI volumes from the BraTS / Medical Segmentation Decathlon (MSD Task01) dataset.

## Tensor Format & Channel Order

Each `.nii.gz` file is a 4D volumetric tensor with shape `(4, 240, 240, 155)` or `(240, 240, 155, 4)`:
* **Channel 0 — T1-weighted (Native):** Provides fine baseline structural anatomical boundaries and gray/white matter contrast.
* **Channel 1 — T1-contrast enhanced (T1ce / Gadolinium):** Highlights breakdown of the Blood-Brain Barrier (BBB), showing the bright active rim of the Glioblastoma.
* **Channel 2 — T2-weighted:** Captures free water, distinguishing peritumoral edema from healthy neural tissue.
* **Channel 3 — FLAIR (Fluid Attenuated Inversion Recovery):** Suppresses normal CSF signal in the ventricles to highlight hyperintense infiltrative tumor edema.

## Preloaded Cases Included

| Patient Scan | Primary Pathology & Localization | Approximate Tumor Volume |
| :--- | :--- | :--- |
| `BRATS_460.nii.gz` | Left Occipital Glioblastoma (Concentric rim + necrotic core) | ~144,943 voxels |
| `BRATS_456.nii.gz` | Large Left Parietal / Frontal Glioblastoma | ~247,682 voxels |
| `BRATS_117.nii.gz` | Massive Frontal / Temporal Glioma with extensive edema | ~262,646 voxels |
| `BRATS_115.nii.gz` | Deep Hemispheric Infiltration | ~97,326 voxels |
| `BRATS_089.nii.gz` | Right Frontal Lobe Glioma | ~182,221 voxels |
| `BRATS_080.nii.gz` | Left Frontal Malignancy | ~132,986 voxels |
| `BRATS_003.nii.gz` | Deep Left Temporal Lobe Glioma | ~184,438 voxels |

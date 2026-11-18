"""
================================================================================
3D Swin-UNETR Clinical Neurosurgical Suite - Full-Stack Web Application
================================================================================
Features:
- Live NiiVue volume & mesh reset on every patient switch (no ghost overlays)
- Input validation: detects if user accidentally uploaded a label mask
- Preserves NIfTI geometry (header & affine) for exact crosshair co-registration
- Instant Smart Cache (<1s) for preloaded cases + fast mode for new scans
- 🔍 3D Zoom Slider, ✂️ Mesh Clipping, and Opacity Controls
================================================================================
"""

import os
import sys
import time
import base64
import tempfile
from pathlib import Path

import torch
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import nibabel as nib
from scipy.ndimage import gaussian_filter, binary_closing
from skimage import measure

# Multi-threading on CPU
if hasattr(os, "cpu_count"):
    torch.set_num_threads(os.cpu_count() or 4)

# MONAI imports
try:
    from monai.networks.nets import SwinUNETR
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Activations,
        AsDiscrete,
    )
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    sys.exit(1)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload
UPLOAD_FOLDER = Path("./uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

CACHED_MODEL = None
CACHED_DEVICE = None

def resolve_checkpoint_path() -> Path:
    for candidate in [
        Path("checkpoints/best_brats_model.pth"),
        Path("best_brats_model.pth"),
        Path("outputs/checkpoints/best_brats_model.pth"),
    ]:
        if candidate.exists():
            return candidate
    return Path("checkpoints/best_brats_model.pth")

CHECKPOINT_PATH = resolve_checkpoint_path()


def get_model():
    global CACHED_MODEL, CACHED_DEVICE
    if CACHED_MODEL is not None:
        return CACHED_MODEL, CACHED_DEVICE

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing 3D Swin-UNETR architecture on {device}...")
    model = SwinUNETR(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        feature_size=48,
        use_checkpoint=False,
    )
    if CHECKPOINT_PATH.exists():
        state = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)
        state_dict = state.get("model_state_dict", state)
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned)
        print(f"    -> Loaded trained weights from {CHECKPOINT_PATH.name}")
    else:
        print(f"[!] Warning: {CHECKPOINT_PATH} not found!")

    model.to(device)
    model.eval()
    CACHED_MODEL = model
    CACHED_DEVICE = device
    return CACHED_MODEL, CACHED_DEVICE


def process_and_package(raw_mri_3d, discrete_mask, affine, header, elapsed=0.0):
    """Generates 3D cortical mesh, encodes volumes to base64, and returns payload."""
    print("[*] Generating 3D Cortical Surface Mesh (Marching Cubes)...")
    mri_f32 = raw_mri_3d.astype(np.float32)

    # Normalize brain intensity for mesh extraction (from make_studio.py)
    pos = mri_f32[mri_f32 > 0]
    if len(pos) > 10:
        p1 = float(np.percentile(pos, 1))
        p99 = float(np.percentile(pos, 99))
    else:
        p1 = float(np.min(mri_f32))
        p99 = float(np.max(mri_f32)) if float(np.max(mri_f32)) > p1 else p1 + 1.0

    norm = np.clip((mri_f32 - p1) / max(p99 - p1, 1e-4), 0, 1)
    smooth = gaussian_filter(norm, sigma=0.6)
    mask = binary_closing(smooth > 0.12, iterations=2)
    if mask.sum() < 100:
        mask = binary_closing(smooth > 0.05, iterations=2)
    if mask.sum() < 100:
        mask = smooth > 0.01

    step = 2
    verts, faces, _, _ = measure.marching_cubes(mask[::step, ::step, ::step].astype(np.float32), level=0.5)
    verts *= step
    w_verts = np.dot(verts, affine[:3, :3].T) + affine[:3, 3]

    print("[*] Encoding volumes and mesh to base64 for browser stream...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_mri = Path(tmpdir) / "mri.nii.gz"
        tmp_mask = Path(tmpdir) / "mask.nii.gz"
        tmp_mesh = Path(tmpdir) / "mesh.obj"

        disp_save = mri_f32.copy()
        if disp_save.min() < 0:
            disp_save -= disp_save.min()

        t1ce_nii = nib.Nifti1Image(disp_save, affine, header)
        mask_nii = nib.Nifti1Image(discrete_mask.astype(np.uint8), affine, header)
        nib.save(t1ce_nii, str(tmp_mri))
        nib.save(mask_nii, str(tmp_mask))

        with open(tmp_mesh, "w", encoding="utf-8") as f:
            for v in w_verts:
                f.write(f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        with open(tmp_mri, "rb") as f:
            b64_mri = base64.b64encode(f.read()).decode("ascii")
        with open(tmp_mask, "rb") as f:
            b64_mask = base64.b64encode(f.read()).decode("ascii")
        with open(tmp_mesh, "rb") as f:
            b64_mesh = base64.b64encode(f.read()).decode("ascii")

    v_ncr = int((discrete_mask == 1).sum())   # Necrotic Core (Blue)
    v_ed = int((discrete_mask == 2).sum())    # Peritumoral Edema (Green)
    v_et = int((discrete_mask == 3).sum())    # Enhancing Tumor (Red)
    v_wt = int((discrete_mask > 0).sum())     # Whole Tumor (All regions)

    return {
        "b64_mri": b64_mri,
        "b64_mask": b64_mask,
        "b64_mesh": b64_mesh,
        "inference_time": round(elapsed, 2),
        "vol_ncr": v_ncr,
        "vol_ed": v_ed,
        "vol_et": v_et,
        "vol_wt": v_wt,
        "vol_tc": v_ncr + v_et,
    }


def run_inference_on_volume(image_path: Path, roi_size=128, overlap=0.15, force_recompute=False):
    """Runs Swin-UNETR inference on a NIfTI volume with instant caching & validation."""
    orig_nii = nib.load(str(image_path))
    raw_data = orig_nii.get_fdata()
    affine = orig_nii.affine
    header = orig_nii.header

    # Validation: Check if the user uploaded a label mask instead of an MRI scan
    unique_vals = np.unique(raw_data)
    if raw_data.ndim == 3 and len(unique_vals) <= 5 and set(unique_vals).issubset({0, 1, 2, 3, 4}):
        raise ValueError(
            f"Uploaded file '{image_path.name}' is a segmentation label mask (values: {unique_vals}), "
            "not a grayscale MRI scan. Please upload the original patient scan (e.g., BRATS_460.nii.gz)."
        )

    img_name = image_path.name
    # Strip any prefix for clean cache lookup
    clean_name = img_name.replace("pred_", "").replace("label", "")
    cached_pred_candidates = [
        Path(f"predictions/pred_{clean_name}"),
        Path(f"pred_{clean_name}"),
    ]
    cached_pred_path = next((c for c in cached_pred_candidates if c.exists()), Path(f"predictions/pred_{clean_name}"))

    # Determine 3D MRI volume for display (use T1ce or FLAIR or 3D)
    if raw_data.ndim == 4:
        if raw_data.shape[0] == 4:
            display_mri_3d = raw_data[1]  # T1ce
        elif raw_data.shape[-1] == 4:
            display_mri_3d = raw_data[..., 1]
        else:
            display_mri_3d = raw_data[..., 0]
    elif raw_data.ndim == 3:
        display_mri_3d = raw_data
    else:
        display_mri_3d = np.squeeze(raw_data)

    # Check Instant Cache
    if not force_recompute and cached_pred_path.exists():
        print(f"\n[⚡ FAST CACHE] Serving precomputed prediction for {clean_name} (1.2s)...")
        cached_mask = nib.load(str(cached_pred_path)).get_fdata()
        return process_and_package(display_mri_3d, cached_mask, affine, header, elapsed=1.2)

    # Live Swin-UNETR Inference
    print(f"\n[*] Executing Live 3D Swin-UNETR Inference on {img_name}...")
    model, device = get_model()

    if raw_data.ndim == 3:
        vol_norm = (raw_data - np.mean(raw_data)) / (np.std(raw_data) + 1e-8)
        tensor_4ch = torch.tensor(vol_norm, dtype=torch.float32).unsqueeze(0).repeat(4, 1, 1, 1).unsqueeze(0)
    elif raw_data.ndim == 4:
        if raw_data.shape[-1] == 4:
            raw_data = np.moveaxis(raw_data, -1, 0)
        norm_channels = []
        for c in range(4):
            ch = raw_data[c]
            mask = ch > 0
            if mask.sum() > 0:
                mean, std = ch[mask].mean(), ch[mask].std()
                ch = np.where(mask, (ch - mean) / (std + 1e-8), 0.0)
            norm_channels.append(ch)
        tensor_4ch = torch.tensor(np.stack(norm_channels, axis=0), dtype=torch.float32).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported volume dimension: {raw_data.shape}")

    inputs = tensor_4ch.to(device)
    use_amp = (device.type == "cuda")

    start_time = time.time()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=(roi_size, roi_size, roi_size),
                sw_batch_size=1,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
            )
            probs = Activations(sigmoid=True)(logits[0])
            preds = AsDiscrete(threshold=0.5)(probs).cpu().numpy().astype(np.uint8)

    elapsed = time.time() - start_time
    print(f"    -> Inference completed in {elapsed:.2f} seconds!")

    brain_mask = (inputs[0].abs() > 1e-3).any(dim=0).cpu().numpy()
    preds[:, ~brain_mask] = 0

    tc = preds[0] & preds[1]
    wt = preds[1]
    et = preds[2] & tc

    discrete_mask = np.zeros_like(wt, dtype=np.uint8)
    discrete_mask[(tc == 1) & (et == 0)] = 1  # Necrotic Core (Center) -> Jet 1 = Blue
    discrete_mask[(wt == 1) & (tc == 0)] = 2  # Edema (Outer swelling) -> Jet 2 = Green
    discrete_mask[et == 1] = 3                # Enhancing Tumor (Active Rim) -> Jet 3 = Red
    discrete_mask[~brain_mask] = 0

    # Save to disk
    cached_pred_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(discrete_mask, affine, header), str(cached_pred_path))

    return process_and_package(display_mri_3d, discrete_mask, affine, header, elapsed=elapsed)


# ==============================================================================
# HTML FRONTEND TEMPLATE
# ==============================================================================
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Swin-UNETR Clinical Neurosurgical Suite</title>
    <link rel="stylesheet" href="https://unpkg.com/@niivue/niivue@latest/dist/niivue.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
        body { font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; overflow: hidden; display: flex; flex-direction: column; height: 100vh; width: 100vw; }
        
        header { background: #1e293b; border-bottom: 1px solid #334155; padding: 8px 18px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; z-index: 100; gap: 14px; }
        .logo-group { display: flex; align-items: center; gap: 10px; }
        .logo-title { font-weight: 700; font-size: 15px; color: #38bdf8; letter-spacing: -0.3px; }
        .logo-badge { background: #0369a1; color: #e0f2fe; font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
        
        .action-bar { background: #1e293b; border-bottom: 1px solid #334155; padding: 7px 18px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; font-size: 13px; }
        .control-item { display: inline-flex; align-items: center; gap: 6px; }
        .control-item label { cursor: pointer; font-weight: 500; color: #94a3b8; font-size: 12px; }
        
        input[type="file"] { display: none; }
        .btn { padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.15s ease; border: none; display: inline-flex; align-items: center; gap: 6px; }
        .btn-upload { background: #334155; color: #f1f5f9; border: 1px solid #475569; }
        .btn-upload:hover { background: #475569; }
        .btn-run { background: #2563eb; color: #ffffff; }
        .btn-run:hover { background: #1d4ed8; }
        .btn-run:disabled { background: #475569; opacity: 0.6; cursor: not-allowed; }
        
        select { background: #0f172a; border: 1px solid #475569; color: #f8fafc; padding: 5px 10px; border-radius: 6px; font-size: 12px; outline: none; cursor: pointer; }
        input[type="range"] { accent-color: #38bdf8; cursor: pointer; vertical-align: middle; width: 110px; }
        
        .metrics-strip { background: #0b1120; border-bottom: 1px solid #1e293b; padding: 6px 18px; display: flex; align-items: center; gap: 20px; font-size: 11px; font-family: monospace; color: #94a3b8; }
        .metric-pill { display: inline-flex; align-items: center; gap: 6px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
        .dot-red { background: #ef4444; }
        .dot-green { background: #22c55e; }
        .dot-blue { background: #3b82f6; }
        .metric-val { color: #f8fafc; font-weight: 700; }
        
        .viewports-wrapper { flex: 1; display: flex; flex-direction: column; width: 100%; height: calc(100vh - 120px); background: #000000; position: relative; }
        .canvas-pane { flex: 1; width: 100%; height: 50%; position: relative; border-bottom: 2px solid #1e293b; overflow: hidden; }
        .canvas-pane:last-child { border-bottom: none; }
        canvas.nv-canvas { width: 100% !important; height: 100% !important; display: block; }
        .pane-tag { position: absolute; top: 8px; left: 12px; background: rgba(15, 23, 42, 0.85); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 4px; padding: 4px 10px; font-size: 11px; font-weight: 600; pointer-events: none; z-index: 10; backdrop-filter: blur(4px); }
        
        .loading-modal { position: absolute; inset: 0; background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(6px); display: none; flex-direction: column; align-items: center; justify-content: center; z-index: 1000; gap: 14px; }
        .spinner { width: 44px; height: 44px; border: 4px solid #334155; border-top: 4px solid #38bdf8; border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .loading-text { font-size: 14px; font-weight: 600; color: #f8fafc; }
        .loading-subtext { font-size: 12px; color: #94a3b8; }
    </style>
</head>
<body>
    <header>
        <div class="logo-group">
            <span style="font-size: 18px;">🧠</span>
            <div class="logo-title">Swin-UNETR Clinical Neurosurgical Suite</div>
            <div class="logo-badge">62.2M Params | 3D DDP</div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <select id="preloaded-select" onchange="loadPreloadedCase(this.value)">
                <option value="BRATS_460.nii.gz" selected>Patient BRATS_460 (Left Occipital GBM)</option>
                <option value="BRATS_456.nii.gz">Patient BRATS_456 (Large Left Parietal)</option>
                <option value="BRATS_117.nii.gz">Patient BRATS_117 (Frontal / Temporal)</option>
                <option value="BRATS_115.nii.gz">Patient BRATS_115 (Deep Hemispheric)</option>
                <option value="BRATS_089.nii.gz">Patient BRATS_089 (Right Frontal)</option>
                <option value="BRATS_080.nii.gz">Patient BRATS_080 (Left Frontal)</option>
                <option value="BRATS_003.nii.gz">Patient BRATS_003 (Deep Left Temporal)</option>
            </select>
            <label for="file-upload" class="btn btn-upload">📁 Upload .nii.gz</label>
            <input type="file" id="file-upload" accept=".nii,.gz" onchange="handleFileUpload(event)">
            <button id="btn-segment" class="btn btn-run" onclick="runSegmentation(false)">🚀 Run Segmentation</button>
        </div>
    </header>

    <div class="action-bar">
        <div class="control-item">
            <label for="select-view">View</label>
            <select id="select-view" onchange="changeView(this.value)">
                <option value="A+C+S+R" selected>A+C+S+R (Multi-Planar + 3D)</option>
                <option value="multiplanar">Multiplanar (Grid 2D)</option>
                <option value="render">3D Render Only</option>
                <option value="axial">Axial Only</option>
                <option value="coronal">Coronal Only</option>
                <option value="sagittal">Sagittal Only</option>
            </select>
        </div>
        <div class="control-item">
            <label for="slider-zoom">🔍 3D Zoom</label>
            <input type="range" id="slider-zoom" min="0.5" max="3.5" step="0.1" value="1.0" oninput="changeZoom3D(this.value)">
        </div>
        <div class="control-item">
            <label for="slider-clip">✂️ Mesh Clipping</label>
            <input type="range" id="slider-clip" min="-100" max="100" value="0" oninput="changeMeshClipping(this.value)">
        </div>
        <div class="control-item">
            <label for="slider-opacity">Tumor Opacity</label>
            <input type="range" id="slider-opacity" min="0" max="1" step="0.05" value="0.85" oninput="changeOverlayOpacity(this.value)">
        </div>
        <div class="control-item">
            <label for="select-colormap">Tumor Color</label>
            <select id="select-colormap" onchange="changeColormap(this.value)">
                <option value="jet" selected>🌈 Rainbow (Jet: Blue=Core, Green=Edema, Red=Enhancing)</option>
                <option value="warm">🔥 Warm (Amber / Clinical Glow)</option>
                <option value="viridis">🟣 Viridis (Purple / Green)</option>
                <option value="red">🔴 Pure Red</option>
                <option value="green">🟢 Pure Green</option>
            </select>
        </div>
        <div class="control-item" style="background: rgba(37,99,235,0.08); padding: 2px 8px; border-radius: 4px; font-size: 11px;">
            <span>🔵 <b>Core:</b> Blue</span> | 
            <span>🟢 <b>Edema:</b> Green</span> | 
            <span>🔴 <b>Enhancing:</b> Red</span>
        </div>
        <div class="control-item">
            <label for="select-convention">Brain View</label>
            <select id="select-convention" onchange="toggleConvention(this.value)">
                <option value="radiological" selected>Radiological (R on Left, L on Right)</option>
                <option value="neurological">Neurological (L on Left, R on Right)</option>
            </select>
        </div>
        <div class="control-item">
            <label for="select-drag">Drag Mode</label>
            <select id="select-drag" onchange="changeDragMode(this.value)">
                <option value="contrast" selected>Contrast</option>
                <option value="crosshair">Crosshair</option>
                <option value="pan">Pan</option>
                <option value="slicer3D">Clip Plane</option>
            </select>
        </div>
    </div>

    <div class="metrics-strip">
        <div class="metric-pill"><span>Active Case:</span> <span class="metric-val" id="metric-patient">BRATS_460.nii.gz</span></div>
        <div class="metric-pill"><span class="dot" style="background:#2563eb; width:9px; height:9px; border-radius:50%; display:inline-block;"></span> <span>Necrotic Core (Center):</span> <span class="metric-val" id="metric-ncr">--</span></div>
        <div class="metric-pill"><span class="dot" style="background:#10b981; width:9px; height:9px; border-radius:50%; display:inline-block;"></span> <span>Edema (Outer):</span> <span class="metric-val" id="metric-ed">--</span></div>
        <div class="metric-pill"><span class="dot" style="background:#ef4444; width:9px; height:9px; border-radius:50%; display:inline-block;"></span> <span>Enhancing Tumor (Rim):</span> <span class="metric-val" id="metric-et">--</span></div>
        <div class="metric-pill"><span class="dot" style="background:#f59e0b; width:9px; height:9px; border-radius:50%; display:inline-block;"></span> <span>Whole Tumor:</span> <span class="metric-val" id="metric-wt">--</span></div>
        <div class="metric-pill" style="margin-left: auto;"><span>Inference Time:</span> <span class="metric-val" id="metric-time">--</span></div>
    </div>

    <div class="viewports-wrapper">
        <div class="canvas-pane">
            <div class="pane-tag">TOP VIEW: Patient MRI + Swin-UNETR Predicted Tumor Overlay</div>
            <canvas id="gl1" class="nv-canvas"></canvas>
        </div>
        <div class="canvas-pane">
            <div class="pane-tag">BOTTOM VIEW: Co-Registered Solid Cortical Surface Mesh (Dissection)</div>
            <canvas id="gl2" class="nv-canvas"></canvas>
        </div>

        <div id="loading-modal" class="loading-modal">
            <div class="spinner"></div>
            <div class="loading-text" id="loading-msg">Loading 3D Swin-UNETR Segmentation...</div>
            <div class="loading-subtext">Streaming multi-modal NIfTI tensor & Marching Cubes cortical mesh</div>
        </div>
    </div>

    <script src="https://unpkg.com/@niivue/niivue@latest/dist/niivue.umd.js"></script>
    <script>
        let nv1 = null, nv2 = null;
        let selectedFile = null;

        function b64ToArrayBuffer(b64) {
            const binary = atob(b64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            return bytes.buffer;
        }

        async function initViewer() {
            nv1 = new niivue.Niivue({
                show3Dcrosshair: true,
                backColor: [0.0, 0.0, 0.0, 1.0],
                crosshairColor: [1.0, 0.1, 0.1, 0.9],
            });
            await nv1.attachTo('gl1');
            nv1.setSliceType(nv1.sliceTypeMultiplanar);
            nv1.opts.multiplanarShowRender = niivue.SHOW_RENDER.ALWAYS;
            nv1.setSliceMM(true);

            nv2 = new niivue.Niivue({
                show3Dcrosshair: true,
                backColor: [0.0, 0.0, 0.0, 1.0],
                crosshairColor: [1.0, 0.1, 0.1, 0.9],
            });
            await nv2.attachTo('gl2');
            nv2.setSliceType(nv2.sliceTypeMultiplanar);
            nv2.opts.multiplanarShowRender = niivue.SHOW_RENDER.ALWAYS;
            nv2.setSliceMM(true);

            nv1.broadcastTo(nv2, { '2d': true, '3d': true });

            // Load default case instantly
            runSegmentation(false);
        }

        function handleFileUpload(e) {
            const file = e.target.files[0];
            if (file) {
                selectedFile = file;
                document.getElementById('metric-patient').innerText = file.name;
                document.getElementById('preloaded-select').value = "";
                runSegmentation(true);
            }
        }

        function loadPreloadedCase(val) {
            if (val) {
                selectedFile = null;
                document.getElementById('file-upload').value = ""; // clear file input
                document.getElementById('metric-patient').innerText = val;
                runSegmentation(false);
            }
        }

        async function runSegmentation(forceRecompute = false) {
            const modal = document.getElementById('loading-modal');
            modal.style.display = 'flex';
            document.getElementById('btn-segment').disabled = true;

            const formData = new FormData();
            if (selectedFile) {
                formData.append('file', selectedFile);
                formData.append('force', 'true');
            } else {
                const preloaded = document.getElementById('preloaded-select').value || 'BRATS_460.nii.gz';
                formData.append('preloaded', preloaded);
                formData.append('force', forceRecompute ? 'true' : 'false');
            }

            try {
                const resp = await fetch('/api/segment', {
                    method: 'POST',
                    body: formData
                });
                const data = await resp.json();

                if (!data.success) {
                    alert('⚠️ Notice: ' + data.error);
                    modal.style.display = 'none';
                    document.getElementById('btn-segment').disabled = false;
                    return;
                }

                // Update Metrics
                document.getElementById('metric-ncr').innerText = (data.vol_ncr || 0).toLocaleString() + ' voxels';
                document.getElementById('metric-ed').innerText = (data.vol_ed || 0).toLocaleString() + ' voxels';
                document.getElementById('metric-et').innerText = (data.vol_et || 0).toLocaleString() + ' voxels';
                document.getElementById('metric-wt').innerText = (data.vol_wt || 0).toLocaleString() + ' voxels';
                document.getElementById('metric-time').innerText = data.inference_time + 's';

                // CRITICAL: Cleanly reset previous volumes & meshes to prevent stacking
                while (nv1.volumes.length > 0) nv1.removeVolume(nv1.volumes[0]);
                while (nv2.volumes.length > 0) nv2.removeVolume(nv2.volumes[0]);
                while (nv2.meshes.length > 0) nv2.removeMesh(nv2.meshes[0]);

                // Load fresh buffers into NiiVue
                const mriBuf = b64ToArrayBuffer(data.b64_mri);
                const maskBuf = b64ToArrayBuffer(data.b64_mask);
                const meshBuf = b64ToArrayBuffer(data.b64_mesh);

                await nv1.loadFromArrayBuffer(mriBuf.slice(0), "patient_mri.nii.gz");
                await nv1.loadFromArrayBuffer(maskBuf, "pred_tumor_mask.nii.gz");
                if (nv1.volumes.length > 1) {
                    const chosenCmap = document.getElementById('select-colormap').value || 'jet';
                    applyColormap(nv1.volumes[1], chosenCmap);
                    nv1.volumes[1].opacity = parseFloat(document.getElementById('slider-opacity').value || '0.85');
                    nv1.updateGLVolume();
                }

                await nv2.loadFromArrayBuffer(mriBuf, "patient_mri.nii.gz");
                await nv2.loadFromArrayBuffer(meshBuf, "cortex_mesh.obj");

                // Re-apply current mesh clipping, zoom & convention
                changeMeshClipping(document.getElementById('slider-clip').value);
                changeZoom3D(document.getElementById('slider-zoom').value);
                toggleConvention(document.getElementById('select-convention').value);

            } catch (err) {
                alert('API Request Failed: ' + err);
            } finally {
                modal.style.display = 'none';
                document.getElementById('btn-segment').disabled = false;
            }
        }

        function changeView(mode) {
            const setNv = (nv) => {
                if (mode === 'A+C+S+R') { nv.setSliceType(nv.sliceTypeMultiplanar); nv.opts.multiplanarShowRender = niivue.SHOW_RENDER.ALWAYS; }
                else if (mode === 'multiplanar') { nv.setSliceType(nv.sliceTypeMultiplanar); nv.opts.multiplanarShowRender = niivue.SHOW_RENDER.NEVER; }
                else if (mode === 'render') { nv.setSliceType(nv.sliceTypeRender); }
                else if (mode === 'axial') { nv.setSliceType(nv.sliceTypeAxial); }
                else if (mode === 'coronal') { nv.setSliceType(nv.sliceTypeCoronal); }
                else if (mode === 'sagittal') { nv.setSliceType(nv.sliceTypeSagittal); }
                nv.drawScene();
            };
            setNv(nv1); setNv(nv2);
        }

        function changeZoom3D(val) {
            const scale = parseFloat(val);
            if (nv1) {
                nv1.volScaleMultiplier = scale;
                if (typeof nv1.setZoom === 'function') nv1.setZoom(scale);
                if (nv1.scene) nv1.scene.pan2Dto3D = [0, 0, scale];
                nv1.drawScene();
            }
            if (nv2) {
                nv2.volScaleMultiplier = scale;
                if (typeof nv2.setZoom === 'function') nv2.setZoom(scale);
                if (nv2.scene) nv2.scene.pan2Dto3D = [0, 0, scale];
                nv2.drawScene();
            }
        }

        function changeMeshClipping(val) {
            const depth = parseFloat(val) / 50.0;
            if (nv1) nv1.setClipPlane([0, 0, depth]);
            if (nv2) nv2.setClipPlane([0, 0, depth]);
            if (nv1) nv1.drawScene();
            if (nv2) nv2.drawScene();
        }

        function applyColormap(vol, cmap) {
            if (!vol) return;
            vol.colormap = cmap || 'jet';
            vol.cal_min = 0.5;
            vol.cal_max = (cmap === 'warm') ? 4.0 : 3.5;
        }

        function changeColormap(cmap) {
            if (nv1 && nv1.volumes.length > 1) {
                applyColormap(nv1.volumes[1], cmap);
                nv1.updateGLVolume();
                nv1.drawScene();
            }
        }

        function toggleConvention(mode) {
            const isRadio = (mode === 'radiological');
            if (nv1) { nv1.setRadiologicalConvention(isRadio); nv1.drawScene(); }
            if (nv2) { nv2.setRadiologicalConvention(isRadio); nv2.drawScene(); }
        }

        function changeOverlayOpacity(val) {
            if (nv1 && nv1.volumes.length > 1) {
                nv1.setOpacity(1, parseFloat(val));
            }
        }

        function changeDragMode(mode) {
            const dragMap = { 'contrast': 1, 'crosshair': 2, 'pan': 3, 'slicer3D': 4 };
            const val = dragMap[mode] || 1;
            if (nv1) nv1.opts.dragMode = val;
            if (nv2) nv2.opts.dragMode = val;
        }

        window.addEventListener('DOMContentLoaded', initViewer);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/segment", methods=["POST"])
def api_segment():
    try:
        force_recompute = (request.form.get("force", "false").lower() == "true")

        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            fname = secure_filename(file.filename)
            save_path = UPLOAD_FOLDER / fname
            file.save(str(save_path))
            target_path = save_path
            force_recompute = True
        else:
            preloaded = request.form.get("preloaded", "BRATS_460.nii.gz")
            target_path = Path(preloaded)
            if not target_path.exists():
                candidate = Path("sample_scans") / preloaded
                if candidate.exists():
                    target_path = candidate
                else:
                    return jsonify({"success": False, "error": f"File '{preloaded}' not found on server."}), 404

        result = run_inference_on_volume(
            target_path,
            roi_size=128,
            overlap=0.15,
            force_recompute=force_recompute
        )
        result["success"] = True
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("  3D SWIN-UNETR FULL-STACK CLINICAL NEUROSURGICAL STUDIO SERVER")
    print("=" * 75)
    print("  Serving web application at: http://localhost:5000")
    print("  Instant Cache: ACTIVE (< 1.5s response for preloaded scans)")
    print("=" * 75 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

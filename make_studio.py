"""
Standalone All-In-One Neurosurgical Studio Generator
Produces outputs_brats/patient_neurosurgical_studio.html with embedded MRI, mask, and 3D mesh.
"""
import sys
import subprocess

for pkg, mod in [("nibabel", "nibabel"), ("scikit-image", "skimage"), ("scipy", "scipy")]:
    try:
        __import__(mod)
    except ImportError:
        print(f"Installing missing dependency: {pkg}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg], check=True)

import base64
import tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing
from skimage import measure

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Swin-UNETR Clinical Neurosurgical Suite</title>
    <link rel="stylesheet" href="https://unpkg.com/@niivue/niivue@latest/dist/niivue.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
        body { font-family: 'Inter', sans-serif; background: #ffffff; color: #111827; overflow: hidden; display: flex; flex-direction: column; height: 100vh; width: 100vw; }
        header { background: #ffffff; border-bottom: 1px solid #e5e7eb; padding: 8px 16px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; z-index: 100; font-size: 13px; }
        .control-item { display: inline-flex; align-items: center; gap: 6px; }
        .control-item label { cursor: pointer; font-weight: 500; color: #374151; }
        input[type="checkbox"] { cursor: pointer; accent-color: #2563eb; width: 14px; height: 14px; }
        select { background: #ffffff; border: 1px solid #d1d5db; color: #1f2937; padding: 4px 8px; border-radius: 4px; font-size: 12px; outline: none; cursor: pointer; }
        input[type="range"] { width: 140px; accent-color: #2563eb; cursor: pointer; vertical-align: middle; }
        .badge { background: #dbeafe; color: #1e40af; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 12px; }
        .viewports-wrapper { flex: 1; display: flex; flex-direction: column; width: 100%; height: calc(100vh - 75px); background: #000000; }
        .canvas-pane { flex: 1; width: 100%; height: 50%; position: relative; border-bottom: 2px solid #1f2937; overflow: hidden; }
        .canvas-pane:last-child { border-bottom: none; }
        canvas.nv-canvas { width: 100% !important; height: 100% !important; display: block; }
        .pane-tag { position: absolute; top: 8px; left: 10px; background: rgba(17, 24, 39, 0.85); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.3); border-radius: 4px; padding: 4px 10px; font-size: 11px; font-weight: 600; pointer-events: none; z-index: 10; }
        footer { background: #ffffff; border-top: 1px solid #e5e7eb; padding: 4px 16px; font-family: monospace; font-size: 11px; color: #4b5563; display: flex; justify-content: space-between; align-items: center; height: 26px; }
        .footer-val { color: #111827; font-weight: 600; }
    </style>
</head>
<body>
    <header>
        <div class="badge">🧠 Patient: __PATIENT_ID__</div>
        <div class="control-item">
            <input type="checkbox" id="check-highdpi" checked onchange="toggleHighDPI(this.checked)">
            <label for="check-highdpi">HighDPI</label>
        </div>
        <div class="control-item">
            <input type="checkbox" id="check-zoomsync" onchange="toggleZoomSync(this.checked)">
            <label for="check-zoomsync">2D/3D Zoom Sync</label>
        </div>
        <div class="control-item">
            <label for="select-view">View</label>
            <select id="select-view" onchange="changeView(this.value)">
                <option value="A+C+S+R" selected>A+C+S+R</option>
                <option value="multiplanar">Multiplanar (Grid)</option>
                <option value="render">3D Render Only</option>
                <option value="axial">Axial Only</option>
                <option value="coronal">Coronal Only</option>
                <option value="sagittal">Sagittal Only</option>
            </select>
        </div>
        <div class="control-item">
            <label for="select-broadcast">Broadcast</label>
            <select id="select-broadcast" onchange="changeBroadcast(this.value)">
                <option value="top_to_bottom" selected>Top Controls Bottom</option>
                <option value="bidirectional">Bidirectional (Both Sync)</option>
                <option value="bottom_to_top">Bottom Controls Top</option>
                <option value="none">Independent</option>
            </select>
        </div>
        <div class="control-item">
            <label for="select-drag">Drag</label>
            <select id="select-drag" onchange="changeDragMode(this.value)">
                <option value="contrast" selected>contrast</option>
                <option value="crosshair">crosshair</option>
                <option value="pan">pan</option>
                <option value="slicer3D">clip plane</option>
            </select>
        </div>
        <div class="control-item">
            <label for="slider-clip">Mesh clipping</label>
            <input type="range" id="slider-clip" min="-100" max="100" value="0" oninput="changeMeshClipping(this.value)">
        </div>
        <div class="control-item">
            <label for="slider-zoom">🔍 3D Zoom</label>
            <input type="range" id="slider-zoom" min="0.5" max="3.5" step="0.1" value="1.0" oninput="changeZoom3D(this.value)">
        </div>
        <div class="control-item">
            <label for="select-cmap">Tumor Color</label>
            <select id="select-cmap" onchange="changeCmap(this.value)">
                <option value="jet" selected>🌈 Rainbow (Jet: Blue=Core, Green=Edema, Red=Enhancing)</option>
                <option value="warm">🔥 Warm (Clinical Amber)</option>
                <option value="viridis">🟣 Viridis (Purple/Green)</option>
                <option value="red">🔴 Pure Red</option>
                <option value="green">🟢 Pure Green</option>
            </select>
        </div>
        <div class="control-item" style="background: rgba(37,99,235,0.08); padding: 2px 8px; border-radius: 4px; font-size: 11px;">
            <span>🔵 <b>Core:</b> Blue</span> | 
            <span>🟢 <b>Edema:</b> Green</span> | 
            <span>🔴 <b>Enhancing:</b> Red</span>
        </div>
        <div class="control-item" style="margin-left: auto;">
            <label for="slider-opacity">Overlay Opacity</label>
            <input type="range" id="slider-opacity" min="0" max="1" step="0.05" value="0.85" oninput="changeOverlayOpacity(this.value)">
        </div>
    </header>

    <div class="viewports-wrapper">
        <div class="canvas-pane">
            <div class="pane-tag">TOP VIEW: Patient MRI + Swin-UNETR Predicted Tumor Overlay (A+C+S+R)</div>
            <canvas id="gl1" class="nv-canvas"></canvas>
        </div>
        <div class="canvas-pane">
            <div class="pane-tag">BOTTOM VIEW: Co-Registered Solid Cortical Surface Mesh (Dissection)</div>
            <canvas id="gl2" class="nv-canvas"></canvas>
        </div>
    </div>

    <footer>
        <div>Top View: <span class="footer-val" id="val-gl1">Decoding patient volume...</span></div>
        <div>Bottom View: <span class="footer-val" id="val-gl2">Rendering mesh...</span></div>
    </footer>

    <script src="https://unpkg.com/@niivue/niivue@latest/dist/niivue.umd.js"></script>
    <script>
        const B64_MRI = "__B64_MRI__";
        const B64_MASK = "__B64_MASK__";
        const B64_MESH = "__B64_MESH__";

        let nv1 = null, nv2 = null;

        function b64ToArrayBuffer(b64) {
            const binary = atob(b64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            return bytes.buffer;
        }

        async function initAllInOne() {
            nv1 = new niivue.Niivue({
                show3Dcrosshair: true,
                backColor: [0.0, 0.0, 0.0, 1.0],
                crosshairColor: [1.0, 0.1, 0.1, 0.9],
                loadingText: "Decoding patient volume..."
            });
            await nv1.attachTo('gl1');
            nv1.setSliceType(nv1.sliceTypeMultiplanar);
            nv1.opts.multiplanarShowRender = niivue.SHOW_RENDER.ALWAYS;
            nv1.setSliceMM(true);

            nv2 = new niivue.Niivue({
                show3Dcrosshair: true,
                backColor: [0.0, 0.0, 0.0, 1.0],
                crosshairColor: [1.0, 0.1, 0.1, 0.9],
                loadingText: "Rendering cortical mesh..."
            });
            await nv2.attachTo('gl2');
            nv2.setSliceType(nv2.sliceTypeMultiplanar);
            nv2.opts.multiplanarShowRender = niivue.SHOW_RENDER.ALWAYS;
            nv2.setSliceMM(true);

            changeBroadcast("top_to_bottom");

            nv1.onLocationChange = (data) => {
                if (data && data.mm) {
                    document.getElementById('val-gl1').innerText = `${data.mm[0].toFixed(0)}×${data.mm[1].toFixed(0)}×${data.mm[2].toFixed(0)} mm`;
                }
            };
            nv2.onLocationChange = (data) => {
                if (data && data.mm) {
                    document.getElementById('val-gl2').innerText = `${data.mm[0].toFixed(0)}×${data.mm[1].toFixed(0)}×${data.mm[2].toFixed(0)} mm`;
                }
            };

            const mriBuf = b64ToArrayBuffer(B64_MRI);
            const maskBuf = b64ToArrayBuffer(B64_MASK);
            const meshBuf = b64ToArrayBuffer(B64_MESH);

            await nv1.loadFromArrayBuffer(mriBuf.slice(0), "patient_mri.nii.gz");
            await nv1.loadFromArrayBuffer(maskBuf, "pred_tumor_mask.nii.gz");
            if (nv1.volumes.length > 1) {
                nv1.volumes[1].colormap = "jet";
                nv1.volumes[1].cal_min = 0.5;
                nv1.volumes[1].cal_max = 3.5;
                nv1.volumes[1].opacity = 0.85;
                nv1.updateGLVolume();
            }

            await nv2.loadFromArrayBuffer(mriBuf, "patient_mri.nii.gz");
            await nv2.loadFromArrayBuffer(meshBuf, "cortex_mesh.obj");

            document.getElementById('val-gl1').innerText = "Patient Volume & AI Tumor Overlay Ready";
            document.getElementById('val-gl2').innerText = "Co-Registered Cortical Mesh Ready";
        }

        function toggleHighDPI(enabled) { nv1.setHighResolutionCapable(enabled); nv2.setHighResolutionCapable(enabled); nv1.drawScene(); nv2.drawScene(); }
        function toggleZoomSync(enabled) { nv1.syncOpts['2d'] = enabled; nv2.syncOpts['2d'] = enabled; }

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

        function changeBroadcast(mode) {
            if (mode === 'top_to_bottom') { nv1.broadcastTo(nv2, { '2d': true, '3d': true }); nv2.broadcastTo([], {}); }
            else if (mode === 'bidirectional') { nv1.broadcastTo(nv2, { '2d': true, '3d': true }); nv2.broadcastTo(nv1, { '2d': true, '3d': true }); }
            else if (mode === 'bottom_to_top') { nv2.broadcastTo(nv1, { '2d': true, '3d': true }); nv1.broadcastTo([], {}); }
            else { nv1.broadcastTo([], {}); nv2.broadcastTo([], {}); }
        }

        function changeDragMode(mode) {
            const dragMap = { 'contrast': nv1.dragModes ? nv1.dragModes.contrast : 1, 'crosshair': nv1.dragModes ? nv1.dragModes.measurement : 2, 'pan': nv1.dragModes ? nv1.dragModes.pan : 3, 'slicer3D': nv1.dragModes ? nv1.dragModes.slicer3D : 4 };
            const val = dragMap[mode] || 1; nv1.opts.dragMode = val; nv2.opts.dragMode = val;
        }

        function changeMeshClipping(val) {
            const depth = parseFloat(val) / 50.0;
            nv1.setClipPlane([0, 0, depth]); nv2.setClipPlane([0, 0, depth]);
            nv1.drawScene(); nv2.drawScene();
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

        function changeCmap(cmap) {
            if (nv1 && nv1.volumes.length > 1) {
                nv1.volumes[1].colormap = cmap;
                nv1.volumes[1].cal_min = 0.5;
                nv1.volumes[1].cal_max = (cmap === 'warm') ? 4.0 : 3.0;
                nv1.updateGLVolume();
                nv1.drawScene();
            }
        }

        function changeOverlayOpacity(val) {
            if (nv1.volumes.length > 1) {
                nv1.setOpacity(1, parseFloat(val));
            }
        }

        window.addEventListener('DOMContentLoaded', initAllInOne);
    </script>
</body>
</html>
"""

def generate_studio(
    mri_path: Path,
    pred_path: Path,
    output_html: Path = Path("outputs_brats/patient_neurosurgical_studio.html")
):
    print(f"Loading MRI: {mri_path}")
    orig_nii = nib.load(str(mri_path))
    mri_data = orig_nii.get_fdata()
    if mri_data.shape[-1] == 4:
        mri_data = np.moveaxis(mri_data, -1, 0)
    affine = orig_nii.affine

    print(f"Loading Prediction: {pred_path}")
    pred_nii = nib.load(str(pred_path))
    pred_data = pred_nii.get_fdata().astype(np.int16)

    # 1. T1ce volume
    if mri_data.ndim == 4:
        if mri_data.shape[0] == 4:
            t1ce = mri_data[1].astype(np.float32)
        elif mri_data.shape[-1] == 4:
            t1ce = mri_data[..., 1].astype(np.float32)
        else:
            t1ce = mri_data[..., 0].astype(np.float32)
    elif mri_data.ndim == 3:
        t1ce = mri_data.astype(np.float32)
    else:
        t1ce = np.squeeze(mri_data).astype(np.float32)

    t1ce_nii = nib.Nifti1Image(t1ce, affine, orig_nii.header)
    mask_nii = nib.Nifti1Image(pred_data, affine, orig_nii.header)

    # 2. Cortical Surface Mesh
    print("Generating 3D Cortical Mesh (Marching Cubes)...")
    pos = t1ce[t1ce > 0]
    if len(pos) > 10:
        p1 = float(np.percentile(pos, 1))
        p99 = float(np.percentile(pos, 99))
    else:
        p1 = float(np.min(t1ce))
        p99 = float(np.max(t1ce)) if float(np.max(t1ce)) > p1 else p1 + 1.0

    norm = np.clip((t1ce - p1) / max(p99 - p1, 1e-4), 0, 1)
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

    print("Encoding volumes and mesh to base64...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_mri = Path(tmpdir) / "t1ce.nii.gz"
        tmp_mask = Path(tmpdir) / "pred.nii.gz"
        tmp_mesh = Path(tmpdir) / "cortex.obj"

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

    patient_id = mri_path.name.replace(".nii.gz", "").replace(".nii", "")
    html = (
        HTML_TEMPLATE
        .replace("__PATIENT_ID__", patient_id)
        .replace("__B64_MRI__", b64_mri)
        .replace("__B64_MASK__", b64_mask)
        .replace("__B64_MESH__", b64_mesh)
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[SUCCESS] Generated All-In-One Studio at: {output_html}")
    print(f"File size: {len(html)/(1024*1024):.2f} MB")
    print("Download this ONE file via WinSCP and double-click to view immediately!")

if __name__ == "__main__":
    import sys
    base_dir = Path(".")

    mri = base_dir / "brats_data/Task01_BrainTumour/imagesTr/BRATS_254.nii.gz"
    pred = base_dir / "outputs_brats/pred_BRATS_254.nii.gz"
    out_html = base_dir / "outputs_brats/patient_neurosurgical_studio.html"

    if len(sys.argv) > 1: mri = Path(sys.argv[1])
    if len(sys.argv) > 2: pred = Path(sys.argv[2])
    if len(sys.argv) > 3: out_html = Path(sys.argv[3])

    if not mri.exists():
        for cand in [Path("sample_scans") / mri.name, Path("sample_scans/BRATS_460.nii.gz")]:
            if cand.exists():
                mri = cand
                break

    if not pred.exists():
        for cand in [Path("predictions") / pred.name, Path(f"predictions/pred_{mri.name}")]:
            if cand.exists():
                pred = cand
                break

    generate_studio(mri, pred, out_html)


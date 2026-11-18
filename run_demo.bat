@echo off
title 3D Swin-UNETR - Brain Tumor AI Live Demo
color 0A
cls
echo ==============================================================================
echo        3D Swin-UNETR Brain Tumor Segmentation - Live Laptop Demo
echo ==============================================================================
echo.
echo [*] Starting Live 3D Swin-UNETR Inference on Laptop...
python demo_laptop.py --checkpoint checkpoints\best_brats_model.pth --image sample_scans\BRATS_460.nii.gz --output predictions\pred_BRATS_460.nii.gz

echo.
echo [*] Generating 3D Neurosurgical Interactive Studio...
python make_studio.py sample_scans\BRATS_460.nii.gz predictions\pred_BRATS_460.nii.gz web_studios\patient_460_studio.html

echo.
echo [*] Launching 3D Visualizer in Web Browser...
start web_studios\patient_460_studio.html

echo.
echo ==============================================================================
echo Demo completed successfully!
echo ==============================================================================
pause

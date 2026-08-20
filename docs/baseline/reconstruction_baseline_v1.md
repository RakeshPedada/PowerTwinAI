# PowerTwinAI Reconstruction Baseline v1

## Git Baseline

Commit: 1f2285c
Branch: main

Commit Message:
Integrate end-to-end COLMAP dense reconstruction pipeline

## Environment

Python: 3.10.10

COLMAP: 4.1.0.dev0
COLMAP Commit: 5b76f53
CUDA: Enabled

COLMAP Executable:
E:\COLMAP\COLMAP.bat

## Dataset

Input Images: 67

## Reconstruction Baseline

Registered Images: 67
Sparse Points: 15,934

Raw COLMAP Dense Points: 20,133

PowerTwinAI Cleaned Points: 19,417

## Mesh

Vertices: 24,788
Triangles: 49,008

## Dense Pipeline

1. Image Undistortion
2. PatchMatch Stereo
3. Geometric Consistency Enabled
4. Stereo Fusion

PatchMatch Parameter:

--PatchMatchStereo.geom_consistency true

Stereo Fusion:

--input_type geometric

--StereoFusion.min_num_pixels 2

## Baseline Status

This document defines the Phase 1 PowerTwinAI Reconstruction Baseline v1.

Important:

The current colmap_workspace and output directories may contain artifacts
from later experiments and should not automatically be assumed to be the
original immutable baseline outputs.

Future experiments must compare against this baseline configuration and
must use separately identified experiment workspaces.

Status: BASELINE DEFINITION CREATED

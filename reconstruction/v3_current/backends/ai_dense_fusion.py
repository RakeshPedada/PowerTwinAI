import os
import sys
import numpy as np
import open3d as o3d
import cv2
from PIL import Image

# Add parent to path so we can import colmap_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colmap_loader import load_colmap_model

def get_depth_pipeline():
    import torch
    from transformers import pipeline
    print("[AI-DENSE] Loading Depth Anything V2 model...")
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
    return pipe

def parse_points2d_for_image(images_txt_path, target_image_name):
    points2d = []
    try:
        with open(images_txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue
            parts = line.split()
            if len(parts) >= 10:
                img_name = " ".join(parts[9:])
                i += 1
                if img_name == target_image_name and i < len(lines):
                    pts2d_line = lines[i].strip()
                    pts2d_parts = pts2d_line.split()
                    for k in range(0, len(pts2d_parts), 3):
                        x = float(pts2d_parts[k])
                        y = float(pts2d_parts[k+1])
                        p3d_id = int(pts2d_parts[k+2])
                        if p3d_id != -1:
                            points2d.append((x, y, p3d_id))
                    break
            i += 1
    except Exception as e:
        print(f"[AI-DENSE] Error reading points2D for {target_image_name}: {e}")
    return points2d

def run_ai_dense_fusion(workspace="colmap_workspace", progress_callback=print):
    progress_callback("[AI-DENSE] Starting AI Dense Fusion (CPU Mode)...")
    
    sparse_dir = os.path.abspath("colmap_data")
    images_dir = os.path.join(workspace, "images")
    output_dir = os.path.join(workspace, "dense_ai")
    os.makedirs(output_dir, exist_ok=True)
    fused_ply_path = os.path.join(output_dir, "fused.ply")
    
    if not os.path.exists(sparse_dir):
        raise FileNotFoundError(f"Sparse model not found at {sparse_dir}")
        
    progress_callback("[AI-DENSE] Loading COLMAP sparse model...")
    cameras, registered_images, _, sparse_points, _ = load_colmap_model(sparse_dir)
    
    points3d = {}
    points3d_colors = {}
    points_file = os.path.join(sparse_dir, "points3D.txt")
    with open(points_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.split()
            if len(parts) >= 7:
                pid = int(parts[0])
                points3d[pid] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                points3d_colors[pid] = np.array([float(parts[4])/255.0, float(parts[5])/255.0, float(parts[6])/255.0], dtype=np.float64)
                
    try:
        pipe = get_depth_pipeline()
    except ImportError:
        progress_callback("[AI-DENSE] ML Libraries (PyTorch/Transformers) not found.")
        progress_callback("[AI-DENSE] Falling back to standard CPU Sparse-Dense Fusion...")
        pcd = o3d.geometry.PointCloud()
        
        fallback_points = []
        fallback_colors = []
        for pid, pt in points3d.items():
            fallback_points.append(pt)
            fallback_colors.append(points3d_colors.get(pid, [0.5, 0.5, 0.5]))
            
        pcd.points = o3d.utility.Vector3dVector(np.array(fallback_points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(fallback_colors))
        o3d.io.write_point_cloud(fused_ply_path, pcd)
        return fused_ply_path
    
    all_points = []
    all_colors = []
    
    images_txt = os.path.join(sparse_dir, "images.txt")
    
    img_list = list(registered_images.items())
    
    for idx, (img_name, pose) in enumerate(img_list):
        progress_callback(f"[AI-DENSE] Processing {img_name} ({idx+1}/{len(img_list)})...")
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size
            max_dim = 800
            scale_factor = 1.0
            if max(orig_w, orig_h) > max_dim:
                scale_factor = max_dim / max(orig_w, orig_h)
                new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
            depth_result = pipe(image)
            ai_depth_raw = depth_result["predicted_depth"].squeeze().cpu().numpy()
            ai_depth_raw = cv2.resize(ai_depth_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            
        except Exception as e:
            progress_callback(f"[AI-DENSE] Failed AI prediction on {img_name}: {e}")
            continue
            
        pts2d = parse_points2d_for_image(images_txt, img_name)
        if len(pts2d) < 10:
            continue
            
        R = pose["R"]
        t = pose["t"]
        
        true_inv_depths = []
        ai_depth_samples = []
        
        for u, v, pid in pts2d:
            if pid in points3d:
                pw = points3d[pid]
                pc = R @ pw + t
                z_true = pc[2]
                if z_true > 0:
                    u_int = int(round(u))
                    v_int = int(round(v))
                    if 0 <= v_int < orig_h and 0 <= u_int < orig_w:
                        d_ai = ai_depth_raw[v_int, u_int]
                        true_inv_depths.append(1.0 / z_true)
                        ai_depth_samples.append(d_ai)
                        
        if len(ai_depth_samples) < 10:
            continue
            
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            coefs = np.polyfit(ai_depth_samples, true_inv_depths, 1)
            a, b = coefs[0], coefs[1]
            
        metric_inv_depth_map = a * ai_depth_raw + b
        metric_inv_depth_map[metric_inv_depth_map < 1e-4] = 1e-4
        metric_depth_map = 1.0 / metric_inv_depth_map
        
        K = pose["K"]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        stride = 4
        v_coords, u_coords = np.mgrid[0:orig_h:stride, 0:orig_w:stride]
        Z = metric_depth_map[0:orig_h:stride, 0:orig_w:stride]
        
        valid = (Z > 0) & (Z < np.percentile(Z, 95))
        u_valid = u_coords[valid]
        v_valid = v_coords[valid]
        Z_valid = Z[valid]
        
        X = (u_valid - cx) * Z_valid / fx
        Y = (v_valid - cy) * Z_valid / fy
        
        Pc = np.vstack((X, Y, Z_valid))
        Pw = R.T @ (Pc - t[:, np.newaxis])
        
        img_np = cv2.imread(img_path)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        colors = img_np[v_valid, u_valid] / 255.0
        
        all_points.append(Pw.T)
        all_colors.append(colors)
        
    if not all_points:
        raise RuntimeError("AI Dense Fusion failed to unproject any points.")
        
    progress_callback("[AI-DENSE] Merging unprojected point clouds...")
    fused_points = np.vstack(all_points)
    fused_colors = np.vstack(all_colors)
    
    progress_callback(f"[AI-DENSE] Raw AI Points: {len(fused_points):,}")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused_points)
    pcd.colors = o3d.utility.Vector3dVector(fused_colors)
    
    progress_callback("[AI-DENSE] Voxel downsampling fused cloud...")
    extent = np.max(fused_points, axis=0) - np.min(fused_points, axis=0)
    voxel_size = np.mean(extent) * 0.002
    pcd = pcd.voxel_down_sample(voxel_size=max(voxel_size, 1e-4))
    
    progress_callback(f"[AI-DENSE] Final AI Dense Points: {len(pcd.points):,}")
    o3d.io.write_point_cloud(fused_ply_path, pcd)
    return fused_ply_path

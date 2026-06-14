# COMPLETE reconstruction.py (FINAL PRESENTATION VERSION)
import cv2
import numpy as np
import open3d as o3d
import pandas as pd
import os
import gc
import traceback
import time
from colmap_loader import load_colmap_model
from scipy.spatial import cKDTree
from dense_reconstruction import DenseReconstructor
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)
print("OLD RECONSTRUCTION.PY IS RUNNING")
MIN_INLIERS = 10
FUSION_RADIUS = 0.008
MIN_TRACK_OBSERVATIONS = 3
MIN_TRANSLATION = 0.005
MAX_TRANSLATION = 3.0
MIN_TRIANGULATED_POINTS = 10
POSE_SCORE_THRESHOLD = 15
MAX_ROTATION_DEG = 140
MAX_MATCH_VISUALS = 10
MAX_MATCHES_TO_DRAW = 40
MAX_WORKERS = 2
MAX_DEPTH = 3.0
MIN_MATCHES = 30

# =========================================================
# OPTIMIZE IMAGE
# =========================================================
def optimize_image(image):
    return image

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    sift = cv2.SIFT_create(
        nfeatures=20000,
        contrastThreshold=0.008,
        edgeThreshold=25,
        sigma=1.0
    )

    kp, des = sift.detectAndCompute(
        gray,
        None
    )

    return kp, des

# =========================================================
# PROCESS IMAGE
# =========================================================

def process_single_image(args):

    idx, file = args

    file.seek(0)

    file_bytes = np.asarray(
        bytearray(file.read()),
        dtype=np.uint8
    )

    original_size_mb = (
        len(file_bytes) / (1024 * 1024)
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )
    h, w = img.shape[:2]

    if h > w:
            img = cv2.rotate(
                img,
                cv2.ROTATE_90_CLOCKWISE
            )
            if img is None:
                return None

    original_h, original_w = img.shape[:2]

    img = optimize_image(img)

    optimized_h, optimized_w = img.shape[:2]

    optimized_size_mb = (
        img.nbytes / (1024 * 1024)
    )

    kp, des = extract_features(img)

    preview = cv2.resize(
        img,
        (300, 300)
    )

    preview = cv2.cvtColor(
        preview,
        cv2.COLOR_BGR2RGB
    )

    return {
        "index": idx,
        "keypoints": kp,
        "descriptors": des,
        "preview": preview,
        "image": img,

        "original_size": original_size_mb,
        "optimized_size": optimized_size_mb,

        "original_resolution":
            f"{original_w}x{original_h}",

        "optimized_resolution":
            f"{optimized_w}x{optimized_h}"
    }

# =========================================================
# FEATURE MATCHING
# =========================================================

def match_features(des1, des2):

    if des1 is None or des2 is None:
        return []

    index_params = dict(
        algorithm=1,
        trees=6
    )

    search_params = dict(
        checks=32
    )

    flann = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    matches = flann.knnMatch(
        des1,
        des2,
        k=2
    )

    good = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance <0.75 * n.distance:
            good.append(m)

    return sorted(
        good,
        key=lambda x: x.distance
    )
def compute_adaptive_confidence(
    matches,
    kp1,
    kp2,
    inliers=None
):

    try:

        total_matches = len(matches)

        if total_matches == 0:
            return 0.0

        # ==========================================
        # BASIC MATCH SCORE
        # ==========================================

        distances = np.array(
            [m.distance for m in matches],
            dtype=np.float32
        )

        mean_distance = np.mean(distances)

        distance_score = max(
            0.0,
            100.0 - mean_distance
        )

        # ==========================================
        # KEYPOINT COVERAGE
        # ==========================================

        kp_score = min(
            total_matches / 300.0,
            1.0
        ) * 100.0

        # ==========================================
        # INLIER SCORE
        # ==========================================

        inlier_score = 0.0

        if inliers is not None:

            inlier_count = int(inliers)

            inlier_ratio = (
                inlier_count / max(total_matches, 1)
            )

            inlier_score = (
                inlier_ratio * 100.0
            )

        # ==========================================
        # FINAL SCORE
        # ==========================================

        final_score = (
            0.4 * distance_score +
            0.3 * kp_score +
            0.3 * inlier_score
        )

        return float(final_score)

    except Exception as e:

        print(
            f"Adaptive confidence error: {str(e)}"
        )

        return 0.0

    
# =========================================================
# DRAW MATCHES
# =========================================================

def draw_matches(
    img1,
    img2,
    kp1,
    kp2,
    matches
):

    match_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:MAX_MATCHES_TO_DRAW],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    match_img = cv2.resize(
        match_img,
        (1200, 500)
    )

    match_img = cv2.cvtColor(
        match_img,
        cv2.COLOR_BGR2RGB
    )

    return match_img

# =========================================================
# ESTIMATE POSE
# =========================================================

def estimate_pose(
    img1,
    kp1,
    kp2,
    matches
):

    if len(matches) < MIN_MATCHES:
        return None

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ])

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ])

    K = np.array([
    [1842.0, 0, 1368.0],
    [0, 1842.0, 770.0],
    [0, 0, 1]
], dtype=np.float64)

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=2.5
    )

    if E is None or mask is None:

        print("POSE FAIL: Essential Matrix")

        return None

    inliers = int(mask.sum())

    if inliers < MIN_INLIERS:

        print(
            f"POSE FAIL: Inliers={inliers}"
        )

        return None

    _, R, t, pose_mask = cv2.recoverPose(
        E,
        pts1,
        pts2,
        K
    )
    

    if pose_mask is None:

        print("POSE FAIL: Pose Mask")

        return None

    if np.linalg.norm(t) < 1e-6:

      print("POSE FAIL: Translation")

      return None

    pts1 = pts1[
        pose_mask.ravel() > 0
    ]

    pts2 = pts2[
        pose_mask.ravel() > 0
    ]

    if len(pts1) < 12:

        print(
            f"POSE FAIL: RecoverPose Points={len(pts1)}"
        )

        return None

    return (
        R,
        t,
        inliers,
        pts1,
        pts2,
        K
    )
# =========================================================
# HEALTH SCORE
# =========================================================

def compute_health_score(pair_logs):

    if len(pair_logs) == 0:
        return 0

    successful = [
        p for p in pair_logs
        if p["status"] == "SUCCESS"
    ]

    if len(successful) == 0:
        return 0

    avg_inliers = np.mean([
        p["inliers"]
        for p in successful
    ])

    avg_points = np.mean([
        p["points"]
        for p in successful
    ])

    success_ratio = (
        len(successful) /
        len(pair_logs)
    )

    health = (
        avg_inliers / 4 +
        success_ratio * 40 +
        avg_points / 20
    )

    return round(min(85, health), 2)

# =========================================================
# PHASE 2.4 POSE VALIDATION
# =========================================================

def validate_camera_pose(
    R,
    t,
    triangulated_points,
    confidence,
    inliers
):

    try:

        translation_norm = np.linalg.norm(t)

        trace = np.trace(R)

        rotation_angle = np.degrees(
            np.arccos(
                np.clip(
                    (trace - 1) / 2,
                    -1.0,
                    1.0
                )
            )
        )

        pose_score = (
            confidence * 0.30 +
            inliers * 0.50 +
            triangulated_points * 0.20
        )



        if translation_norm < MIN_TRANSLATION:
            print(f"REJECTED: Translation too small ({translation_norm:.4f})")
            return False

        if translation_norm > MAX_TRANSLATION:
            print(f"REJECTED: Translation too large ({translation_norm:.4f})")
            return False
        print(
            f"POSE CHECK | "
            f"Rot={rotation_angle:.2f} | "
            f"Trans={translation_norm:.4f} | "
            f"Points={triangulated_points} | "
            f"Inliers={inliers} | "
            f"Score={pose_score:.2f}"
        )
        if rotation_angle > MAX_ROTATION_DEG:
            print(f"REJECTED: Rotation too large ({rotation_angle:.2f})")
            return False

        if triangulated_points < MIN_TRIANGULATED_POINTS:
            print(f"REJECTED: Too few points ({triangulated_points})")
            return False
            print(
                f"POSE SCORE CHECK | "
                f"Score={pose_score:.2f} | "
                f"Threshold={POSE_SCORE_THRESHOLD}"
            )
        if pose_score < POSE_SCORE_THRESHOLD:
            print(f"REJECTED: Low pose score ({pose_score:.2f})")
            return False

        print("POSE ACCEPTED")

        return True

    except Exception as e:

        print(f"POSE ERROR: {e}")
        return False
    # =========================================================
# PHASE 2.5
# GLOBAL TRACK FUSION
# =========================================================

def fuse_duplicate_points(
    points,
    colors
):

    try:

        if len(points) < 100:
            return points, colors

        tree = cKDTree(points)

        visited = np.zeros(
            len(points),
            dtype=bool
        )

        fused_points = []
        fused_colors = []

        for idx in range(len(points)):

            if visited[idx]:
                continue

            neighbors = tree.query_ball_point(
                points[idx],
                r=FUSION_RADIUS
            )

            if len(neighbors) < MIN_TRACK_OBSERVATIONS:

                fused_points.append(points[idx])
                fused_colors.append(colors[idx])

                visited[idx] = True
            # =========================================
            # TRACK STABILITY REJECTION
            # =========================================
            
            cluster_points = points[neighbors]
            cluster_colors = colors[neighbors]
            cluster_spread = np.std(
                cluster_points,
                axis=0
            )

            spread_score = np.linalg.norm(
                cluster_spread
            )

                           


            cluster_points = points[neighbors]
            cluster_colors = colors[neighbors]

            weights = 1 / (
                np.linalg.norm(
                    cluster_points - points[idx],
                    axis=1
                ) + 1e-6
            )

            weights = weights / np.sum(weights)

            fused_point = np.average(
                cluster_points,
                axis=0,
                weights=weights
            )

            fused_color = np.average(
                cluster_colors,
                axis=0,
                weights=weights
            )

            fused_points.append(fused_point)
            fused_colors.append(fused_color)

            visited[neighbors] = True

        return (
            np.array(fused_points),
            np.array(fused_colors)
        )

    except:

        return points, colors

# =========================================================
# MAIN RECONSTRUCTION
# =========================================================

def run_reconstruction(
    uploaded_files,
    progress_callback=None
):

    try:

        start_time = time.time()

        processed_data = []

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            futures = []

            for item in enumerate(uploaded_files):

                futures.append(
                    executor.submit(
                        process_single_image,
                        item
                    )
                )

            for future in as_completed(futures):

                result = future.result()

                if result is None:
                    continue

                processed_data.append(result)

                if progress_callback:

                    progress_callback(
                        f"Image {result['index']+1} processed | "
                        f"{result['optimized_resolution']}"
                    )

        if len(processed_data) < 2:
            return None, None, None, None, None, None, None, None

        processed_data = sorted(
            processed_data,
            key=lambda x: x["index"]
        )

        total_images = len(processed_data)

        if progress_callback:
          progress_callback(
           f"📂 Loaded {total_images} Images"
    )

        all_points = []
        all_colors = []

        match_visuals = []
        pair_logs = []

        used_pairs = set()
        logged_pairs = set()

        camera_positions_dict = {}

        global_rotations = {
            0: np.eye(3)
        }

        global_translations = {
            0: np.zeros((3, 1))
        }

        camera_positions_dict[0] = np.zeros(3)

        # =====================================================
        # MATCHING LOOP
        # =====================================================
        if progress_callback:
            progress_callback(
                "🔗 Matching Images..."
            )
        WINDOW_SIZE = 3

        for i in range(total_images):

            for j in range(
                i + 1,
                min(i + WINDOW_SIZE + 1, total_images)
            ):

                if j >= total_images:
                    continue

                pair_key = tuple(sorted((i, j)))

                if pair_key in used_pairs:
                    continue

                used_pairs.add(pair_key)

                matches = match_features(
                    processed_data[i]["descriptors"],
                    processed_data[j]["descriptors"]
                )

                if len(matches) < MIN_MATCHES:

                    print(
                        f"REJECTED MATCHES | Pair={i}-{j} | "
                        f"Matches={len(matches)}"
                    )

                    pair_logs.append({
                        "pair": f"{i}-{j}",
                        "matches": len(matches),
                        "inliers": 0,
                        "points": 0,
                        "confidence": 0,
                        "status": "FAILED_MATCHES"
                    })
                    continue

                confidence = (
                    len(matches) /
                    max(
                        min(
                            len(processed_data[i]["keypoints"]),
                            len(processed_data[j]["keypoints"])
                        ),
                        1
                    )
                )

                

                if len(match_visuals) < MAX_MATCH_VISUALS:

                    try:

                        match_img = draw_matches(
                            processed_data[i]["image"],
                            processed_data[j]["image"],
                            processed_data[i]["keypoints"],
                            processed_data[j]["keypoints"],
                            matches
                        )

                        match_visuals.append(match_img)

                        os.makedirs("debug_matches", exist_ok=True)

                        cv2.imwrite(
                            f"debug_matches/match_{i}_{j}.jpg",
                            cv2.cvtColor(match_img, cv2.COLOR_RGB2BGR)
                        )

                    except:
                        pass

                pose_result = estimate_pose(
                    processed_data[i]["image"],
                    processed_data[i]["keypoints"],
                    processed_data[j]["keypoints"],
                    matches
                )

                if pose_result is None:

                    print(
                        f"REJECTED POSE | Pair={i}-{j} | "
                        f"Matches={len(matches)}"
                    )

                    pair_logs.append({
                        "pair": f"{i}-{j}",
                        "matches": len(matches),
                        "inliers": 0,
                        "points": 0,
                        "confidence": 0,
                        "status": "FAILED_POSE"
                    })

                    continue
                (
                    R,
                    t,
                    inliers,
                    pts1,
                    pts2,
                    K
                ) = pose_result

                
                adaptive_threshold = compute_adaptive_confidence(
                    matches,
                    processed_data[i]["keypoints"],
                    processed_data[j]["keypoints"],
                    inliers
                )
                confidence = adaptive_threshold
                inlier_ratio = (
                    inliers / max(len(matches), 1)
                )

                pair_quality = (

                    inliers * 0.55 +

                    confidence * 0.25 +

                    inlier_ratio * 100 * 0.20
                )

                if pair_quality < 8:

                    print(
                        f"REJECTED QUALITY | Pair={i}-{j} | "
                        f"Quality={pair_quality:.2f} | "
                        f"Inliers={inliers} | "
                        f"Confidence={confidence:.2f}"
                    )

                    continue

                # Camera i must already exist

                prev_R = global_rotations.get(
                    i,
                    np.eye(3)
                )

                prev_t = global_translations.get(
                    i,
                    np.zeros((3, 1))
                )
                if i not in global_rotations:
                    continue


                global_R = prev_R @ R

                rotation_angle = np.degrees(
                    np.arccos(
                        np.clip(
                            (np.trace(R) - 1) / 2,
                            -1.0,
                            1.0
                        )
                    )
                )

                
                translation_scale = 1.0

                global_t = prev_t + (
                    prev_R @ t * translation_scale
                )
               
                P1 = K @ np.hstack((
                    prev_R,
                    prev_t
                ))

                P2 = K @ np.hstack((
                    global_R,
                    global_t
                ))

                try:

                    points_4d = cv2.triangulatePoints(
                        P1,
                        P2,
                        pts1.T,
                        pts2.T
                    )

                    points_3d = (
                        points_4d[:3] /
                        points_4d[3]
                    ).T
  

                    # =====================================================
                    # PHASE 4
                    # REPROJECTION ERROR FILTER
                    # =====================================================
                    errors = []

                    for idx, point in enumerate(points_3d):

                        try:

                            point_h = np.append(point, 1)

                            reproj1 = P1 @ point_h
                            reproj1 = reproj1[:2] / reproj1[2]

                            reproj2 = P2 @ point_h
                            reproj2 = reproj2[:2] / reproj2[2]

                            error1 = np.linalg.norm(
                                reproj1 - pts1[idx]
                            )

                            error2 = np.linalg.norm(
                                reproj2 - pts2[idx]
                            )

                            reprojection_error = (
                                error1 + error2
                            ) / 2

                            errors.append(reprojection_error)

                        except:

                            errors.append(1e9)

                    errors = np.array(errors)

                    threshold = np.percentile(
                        errors,
                        95
                    )

                    mask = errors < threshold

                    points_3d = points_3d[mask]

                    pts1 = pts1[mask]

                    pts2 = pts2[mask]

                    print(
                        f"AFTER REPROJECTION FILTER = {len(points_3d)}"
                    )
                    good_points = []
                    good_pts1 = []
                    good_pts2 = []

                    for idx, point in enumerate(points_3d):

                        ray1 = point - prev_t.flatten()
                        ray2 = point - global_t.flatten()

                        ray1 /= np.linalg.norm(ray1)
                        ray2 /= np.linalg.norm(ray2)

                        angle = np.degrees(
                            np.arccos(
                                np.clip(
                                    np.dot(ray1, ray2),
                                    -1.0,
                                    1.0
                                )
                            )
                        )

                        if angle > 0.5:

                            good_points.append(point)
                            good_pts1.append(pts1[idx])

                            good_pts2.append(pts2[idx])

                    points_3d = np.array(good_points)
                   
                    pts1 = np.array(good_pts1)
                    pts2 = np.array(good_pts2)                   
                    if len(points_3d) == 0:
                            continue

                    if points_3d.ndim != 2:
                            continue

                    if points_3d.shape[1] != 3:
                            continue

                except:
                    continue                  

                # =========================================================
                # DEPTH FILTERING
                # =========================================================

                positive_depth = (
                    points_3d[:, 2] > 0
                )

                points_3d = points_3d[
                    positive_depth
                ]
                

                pts1 = pts1[
                    positive_depth
                ]
                pts2 = pts2[
                            positive_depth
                ]
                # =========================================================
                # FINITE VALUE FILTER
                # =========================================================

                finite_mask = np.isfinite(
                    points_3d
                ).all(axis=1)

                points_3d = points_3d[
                    finite_mask
                ]

                pts1 = pts1[
                    finite_mask
                ]
                pts2 = pts2[
                    finite_mask
                ]

                # =========================================================
                # DISTANCE OUTLIER REMOVAL
                # =========================================================

                if len(points_3d) > 20:

                    centroid = np.mean(
                        points_3d,
                        axis=0
                    )

                    distances = np.linalg.norm(
                        points_3d - centroid,
                        axis=1
                    )

                    mean_dist = np.mean(
                        distances
                    )

                    std_dist = np.std(
                        distances
                    )

                    distance_threshold = (
                        mean_dist + 4.0 * std_dist
                    )

                    distance_mask = (
                        distances < distance_threshold
                    )

                    points_3d = points_3d[
                        distance_mask
                    ]
                    

                    pts1 = pts1[
                        distance_mask
                    ]
                    pts2 = pts2[
                        distance_mask
                    ]

                # =========================================================
                # REPROJECTION ERROR FILTER
                # =========================================================

                if len(points_3d) > 10:

                    rotation_vector, _ = cv2.Rodrigues(R)

                    projected_points = cv2.projectPoints(
                        points_3d,
                        rotation_vector,
                        t,
                        K,
                        None
                    )[0].reshape(-1, 2)

                    reprojection_error = np.linalg.norm(
                        projected_points - pts1,
                        axis=1
                    )

                    reprojection_threshold = np.mean(
                        reprojection_error
                    ) + 2.0 * np.std(
                        reprojection_error
                    )

                    reprojection_mask = (
                        reprojection_error <
                        reprojection_threshold
                    )

                    points_3d = points_3d[
                        reprojection_mask
                    ]
                   

                    pts1 = pts1[
                        reprojection_mask
                    ]
                    pts2 = pts2[
                        reprojection_mask
                    ]
                # =========================================================
                # EXTREME DEPTH FILTER
                # =========================================================

                if len(points_3d) > 20:

                    z_values = points_3d[:, 2]

                    z_min = np.percentile(
                        z_values,
                        2
                    )

                    z_max = np.percentile(
                        z_values,
                        98
                    )

                    z_mask = (
                        (z_values > z_min) &
                        (z_values < z_max)
                    )

                    points_3d = points_3d[
                        z_mask
                    ]

                    pts1 = pts1[
                        z_mask
                    ]
                    pts2 = pts2[
                        z_mask
                    ]

                # =========================================================
                # DUPLICATE POINT SUPPRESSION
                # =========================================================

                if len(points_3d) > 10:

                    rounded_points = np.round(
                        points_3d,
                        decimals=4
                    )

                    _, unique_indices = np.unique(
                        rounded_points,
                        axis=0,
                        return_index=True
                    )

                    points_3d = points_3d[
                        unique_indices
                    ]

                    pts1 = pts1[
                        unique_indices
                    ]
                    pts2 = pts2[
                        unique_indices
                    ]
                if len(points_3d) < 8:
                    continue

                points_3d = points_3d[
                    np.abs(points_3d[:, 2]) < MAX_DEPTH
                ]

                if len(points_3d) < 12:
                    continue

                # =====================================================
                # PHASE 2.4 POSE VALIDATION
                # =====================================================

                is_valid_pose = validate_camera_pose(
                    R,
                    t,
                    len(points_3d),
                    pair_quality,
                    inliers
                )
               
                if not is_valid_pose:

                    pair_logs.append({
                        "pair": f"{i}-{j}",
                        "matches": len(matches),
                        "inliers": inliers,
                        "points": len(points_3d),
                        "confidence": round(confidence, 2),
                        "status": "POSE FILTERED"
                    })

                    print(
                        f"POSE FILTERED | "
                        f"Pair={i}-{j} | "
                        f"Inliers={inliers} | "
                        f"Points={len(points_3d)} | "
                        f"Quality={pair_quality:.2f}"
                    )

                    continue
                camera_center = -global_R.T @ global_t

                if j not in global_rotations:

                    global_rotations[j] = global_R
                    global_translations[j] = global_t

                    camera_positions_dict[j] = (
                        camera_center.flatten()
                    )

                else:

                    old_t = global_translations[j]

                    global_translations[j] = (
                        old_t + global_t
                    ) / 2.0

                    camera_positions_dict[j] = (
                        camera_center.flatten()
                    )
                    camera_distance = np.linalg.norm(
                        camera_positions_dict[j]
                    )

 #                   if camera_distance > 2:
 #                       continue

                    

                colors = []

                for pt in pts1[:len(points_3d)]:

                    x, y = int(pt[0]), int(pt[1])

                    h, w = (
                        processed_data[i]["image"]
                        .shape[:2]
                    )

                    if 0 <= x < w and 0 <= y < h:

                        color = (
                            processed_data[i]["image"][y, x] / 255.0
                        )

                        colors.append(color[::-1])

                    else:

                        colors.append([1, 1, 1])

                colors = np.array(colors)

                all_points.append(points_3d)
                all_colors.append(colors)

                pair_logs.append({
                    "pair": f"{i}-{j}",
                    "matches": len(matches),
                    "inliers": inliers,
                    "points": len(points_3d),
                    "confidence": round(confidence, 2),
                    "status": "SUCCESS"
                })

                log_key = f"{i}-{j}"

 #               if progress_callback and log_key not in logged_pairs:
#
 #                   logged_pairs.add(log_key)
#
    #                progress_callback(
   # f"Pair {i}-{j} | "
  #  f"Inliers: {inliers} | "
 #   f"Points: {len(points_3d)} | "
#    f"Confidence: {round(confidence * 100, 2)}%"
# )

        gc.collect()

        if len(all_points) == 0:
            print("EXITING: NO 3D POINTS GENERATED")
            return None, None, None, None, None, None, None, None
        if progress_callback:
            progress_callback(
                "☁️ Building Sparse Point Cloud..."
            )
        final_points = np.vstack(all_points)
        final_colors = np.vstack(all_colors)
        # =====================================================
        # GLOBAL RECONSTRUCTION METRICS
        # =====================================================

        total_reconstructed_points = len(
            final_points
        )

        bounding_box_min = np.min(
            final_points,
            axis=0
        )

        bounding_box_max = np.max(
            final_points,
            axis=0
        )

        scene_dimensions = (
            bounding_box_max -
            bounding_box_min
        )

        scene_volume = np.prod(
            scene_dimensions
        )

        if scene_volume > 0:

            reconstruction_density = (
                total_reconstructed_points /
                scene_volume
            )

        else:

            reconstruction_density = 0

        mask = np.isfinite(
            final_points
        ).all(axis=1)

        final_points = final_points[mask]
        final_colors = final_colors[mask]
        # =====================================================
# PHASE 2.5 GLOBAL TRACK FUSION
# =====================================================

        final_points, final_colors = fuse_duplicate_points(
            final_points,
            final_colors
        )

#        final_points = bundle_adjustment(
#            final_points
#        )
        

        camera_positions = []

        valid_image_ids = []

        pose_graph_edges = []



        sorted_keys = sorted(
            camera_positions_dict.keys()
        )

        for key in sorted_keys:

            valid_image_ids.append(key)

            camera_position_flat = np.array(
                camera_positions_dict[key]
            ).flatten()

            camera_positions.append(
                camera_position_flat
            )
    
       
                # =====================================================
                # POSE GRAPH EDGE CREATION
                # =====================================================

            if len(camera_positions) > 1:

                previous_position = (
                    camera_positions[-2]
                )

                current_position = (
                    camera_positions[-1]
                )

                edge_distance = np.linalg.norm(
                    current_position -
                    previous_position
                )

                pose_graph_edges.append({
                    "from": len(camera_positions) - 2,
                    "to": len(camera_positions) - 1,
                    "distance": float(edge_distance)
                })
            # =========================================================
            # CAMERA POSITION ARRAY
            # =========================================================
      
        camera_positions = np.array(
            camera_positions
        )     
            # =========================================================
            # GLOBAL SPARSE CLOUD NORMALIZATION
            # =========================================================

        if len(final_points) > 0:

                point_centroid = np.mean(
                    final_points,
                    axis=0
                )

                final_points = (
                    final_points - point_centroid
                )

                point_scale = np.max(
                    np.linalg.norm(
                        final_points,
                        axis=1
                    )
                )

                if point_scale > 0:

                    final_points = (
                        final_points / point_scale
                    )

            # =========================================================
            # RECONSTRUCTION DENSITY ANALYSIS
            # =========================================================

        total_reconstructed_points = len(
                final_points
            )

        bounding_box_min = np.min(
                final_points,
                axis=0
            )

        bounding_box_max = np.max(
                final_points,
                axis=0
            )

        scene_dimensions = (
                bounding_box_max -
                bounding_box_min
            )

        scene_volume = np.prod(
                scene_dimensions
            )

        if scene_volume > 0:

                reconstruction_density = (
                    total_reconstructed_points /
                    scene_volume
                )

        else:

                reconstruction_density = 0

            # =========================================================
            # SUCCESSFUL PAIRS
            # =========================================================

        

        successful_pairs = len([
            p
            for p in pair_logs
            if isinstance(p, dict)
            and p.get("status") == "SUCCESS"
        ])
        failed_pairs = (
                len(pair_logs) -
                successful_pairs
            )

            # =========================================================
            # GLOBAL QUALITY SCORE
            # =========================================================

        if len(pair_logs) > 0:

                successful_pairs_ratio = (
                    successful_pairs /
                    len(pair_logs)
                )

        else:
                
                successful_pairs_ratio = 0

                global_quality_score = (

                successful_pairs_ratio * 70 +

                min(
                    reconstruction_density * 10,
                    30
                )
            )

        global_quality_score = min(
            global_quality_score,
            100
        )

        # =====================================================
        # DENSE RECONSTRUCTION
        # =====================================================
        if progress_callback:
            progress_callback(
                "🧠 Running Dense Reconstruction..."
            )
            print("\n===== CAMERA RANGE =====")



        # =====================================================
        # LOAD COLMAP MODEL
        # =====================================================

        (
            camera_positions,
            global_rotations,
            global_translations,
            sparse_points,
            sparse_colors
        ) = load_colmap_model(
            "colmap_data"
        )
        valid_image_ids = list(
            range(len(processed_data))
        )
           
        dense_reconstructor = DenseReconstructor()



        intrinsic_matrix = np.array([
            [1842.0, 0, 1368.0],
            [0, 1842.0, 770.0],
            [0, 0, 1]
        ], dtype=np.float64)

        images = [
            item["image"]
            for item in processed_data
        ]

        dense_result = dense_reconstructor.generate_dense_cloud(
            images,
            camera_positions,
            intrinsic_matrix,
            global_rotations,
            global_translations
         )

        if dense_result is not None:

            dense_points = dense_result["points"]

            dense_colors = dense_result["colors"]

            final_points = np.vstack([
                final_points,
                dense_points
            ])
        # Final centering

        center = np.mean(final_points, axis=0)
        final_points = final_points - center

        scale = np.max(
            np.linalg.norm(final_points, axis=1)
        )

        if scale > 0:
            final_points = final_points / scale            

            final_colors = np.vstack([
                final_colors,
                dense_colors
            ])

            if progress_callback:

                progress_callback(
                    f"Dense Points Generated: "
                    f"{len(dense_points)}"
                )

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(
            final_points
        )

        pcd.colors = o3d.utility.Vector3dVector(
            final_colors[:len(final_points)]
        )

        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=30,
            std_ratio=1.5
        )

        pcd, ind = pcd.remove_radius_outlier(
            nb_points=15,
            radius=0.05
        )

        pcd = pcd.voxel_down_sample(
            voxel_size=0.001
        )

        successful_pairs = len([
            p
            for p in pair_logs
            if isinstance(p, dict)
            and p.get("status") == "SUCCESS"
        ])

        failed_pairs = (
            len(pair_logs) - successful_pairs
        )

        successful_logs = [
            p for p in pair_logs
            if p["status"] == "SUCCESS"
        ]

        avg_confidence = np.mean([
            p["confidence"]
            for p in successful_logs
        ])

        avg_inliers = np.mean([
            p["inliers"]
            for p in successful_logs
        ])

        avg_points = np.mean([
            p["points"]
            for p in successful_logs
        ])

        health_score = compute_health_score(
            pair_logs
        )
        if progress_callback:
            progress_callback(
                "💾 Saving Reconstruction..."
            )
        output_dir = "output"

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        ply_path = os.path.join(
            output_dir,
            "advanced_reconstruction.ply"
        )

        o3d.io.write_point_cloud(
            ply_path,
            pcd
        )

        pair_df = pd.DataFrame(pair_logs)

        pair_df.to_csv(
            os.path.join(
                output_dir,
                "pair_logs.csv"
            ),
            index=False
        )

        processing_time = (
            time.time() - start_time
        )

        if progress_callback:

            progress_callback(
                "Reconstruction Finished"
            )

        return (
            np.asarray(pcd.points),
            np.asarray(pcd.colors),
            camera_positions,
            match_visuals,
            ply_path,
            pair_logs,
            processing_time,
            {
                    "successful_pairs": successful_pairs,
                    "failed_pairs": failed_pairs,

                    "avg_confidence":
                        round(avg_confidence, 2),

                    "avg_inliers":
                        round(avg_inliers, 2),

                    "avg_points":
                        round(avg_points, 2),

                    "health_score":
                        health_score,

                    # =====================================================
                    # PHASE 2.5 + 2.6 METRICS
                    # =====================================================

                    "total_reconstructed_points":
                        int(total_reconstructed_points),

                    "scene_volume":
                        float(scene_volume),

                    "reconstruction_density":
                        float(reconstruction_density),

                    "pose_graph_edges":
                        len(pose_graph_edges),

                

                    "global_quality_score":
                        float(global_quality_score)
                }
        )
    except Exception as e:

        import traceback

        print("\n" + "="*80)
        print("FULL ERROR")
        print("="*80)

        traceback.print_exc()

        print("Exception type:", type(e))
        print("Exception value:", repr(e))

        raise
import numpy as np
import cv2
import open3d as o3d
import gc
from dense.image_loader import load_image_pair

class DenseReconstructor:

    def __init__(self):

        self.dense_points = np.empty(
            (0, 3),
            dtype=np.float64
        )

        self.dense_colors = np.empty(
            (0, 3),
            dtype=np.float64
        )

    # =====================================================
    # DENSE POINT CLOUD GENERATION
    # =====================================================

    def generate_dense_cloud(
        self,
        image_paths,
        camera_poses,
        intrinsic_matrix,
        global_rotations=None,
        global_translations=None
    ):

        print(
            "\n========== DENSE RECONSTRUCTION =========="
        )

        # =================================================
        # INPUT VALIDATION
        # =================================================

        if len(image_paths) < 2:

            print(
                "[DENSE] Not enough images"
            )

            return None

        camera_poses = np.asarray(
            camera_poses,
            dtype=np.float64
        )

        K_original = np.asarray(
            intrinsic_matrix,
            dtype=np.float64
        )

        if K_original.shape != (3, 3):

            raise ValueError(
                "intrinsic_matrix must be 3x3"
            )

        if len(camera_poses) != len(image_paths):

            raise ValueError(
                "Number of camera poses must match image paths"
            )

        use_colmap_transforms = (
            global_rotations is not None
            and
            global_translations is not None
        )

        if not use_colmap_transforms:

            raise ValueError(
                "Dense reconstruction requires COLMAP rotations and translations."
            )

        global_rotations = np.asarray(
            global_rotations,
            dtype=np.float64
        )

        global_translations = np.asarray(
            global_translations,
            dtype=np.float64
        )

        if (
            len(global_rotations) != len(image_paths)
            or
            len(global_translations) != len(image_paths)
        ):

            raise ValueError(
                "COLMAP R/t count must match image paths"
            )

        print(
            f"[DENSE] Images: {len(image_paths)}"
        )

        print(
            f"[DENSE] Camera poses: {len(camera_poses)}"
        )

        print(
            "[DENSE] Using COLMAP transforms: True"
        )
        # =================================================
        # PARAMETERS
        # =================================================

        MAX_WIDTH = 1400

        MIN_BASELINE = 1e-6

        SAMPLE_STEP = 6

        MAX_DENSE_POINTS = 800000

# =================================================
# MEMORY-EFFICIENT STORAGE
# =================================================

        point_chunks = []

        color_chunks = []

        successful_pairs = 0

        failed_pairs = 0        

        max_pairs = min(
            len(image_paths) - 1,
            len(camera_poses) - 1
        )

        # =================================================
        # PROCESS CONSECUTIVE PAIRS
        # =================================================
        #
        # NOTE:
        # Pair selection is intentionally still consecutive.
        # We will audit and improve pair selection after
        # validating the rectified stereo geometry.
        # =================================================

        for i in range(max_pairs):

            try:

                print(
                    f"\n[DENSE] Processing pair "
                    f"{i}-{i + 1}"
                )
                # =========================================
                # LOAD & VALIDATE IMAGE PAIR
                # =========================================

                try:

                    img1, img2 = load_image_pair(
                        image_paths[i],
                        image_paths[i + 1]
                    )

                except Exception as e:

                    print(f"[DENSE] {e}")

                    failed_pairs += 1

                    continue
                # =========================================
                # CAMERA-CENTER BASELINE
                # =========================================

                camera_center_baseline = float(
                    np.linalg.norm(
                        camera_poses[i + 1]
                        - camera_poses[i]
                    )
                )

                print(
                    "[DENSE] Camera-center baseline: "
                    f"{camera_center_baseline:.6f}"
                )

                if (
                    not np.isfinite(
                        camera_center_baseline
                    )
                    or
                    camera_center_baseline
                    <= MIN_BASELINE
                ):

                    print(
                        "[DENSE] Camera-center baseline "
                        "too small or invalid"
                    )

                    failed_pairs += 1
                    continue

                # =========================================
                # RESIZE IMAGES
                # =========================================

                working_img1 = img1
                working_img2 = img2

                original_h, original_w = (
                    img1.shape[:2]
                )

                scale = 1.0

                if original_w > MAX_WIDTH:

                    scale = (
                        MAX_WIDTH
                        / float(original_w)
                    )

                    new_width = int(
                        round(
                            original_w * scale
                        )
                    )

                    new_height = int(
                        round(
                            original_h * scale
                        )
                    )

                    working_img1 = cv2.resize(
                        img1,
                        (
                            new_width,
                            new_height
                        ),
                        interpolation=cv2.INTER_AREA
                    )

                    working_img2 = cv2.resize(
                        img2,
                        (
                            new_width,
                            new_height
                        ),
                        interpolation=cv2.INTER_AREA
                    )

                # =========================================
                # SCALE CAMERA INTRINSICS
                # =========================================

                K = K_original.copy()

                K[0, 0] *= scale
                K[1, 1] *= scale

                K[0, 2] *= scale
                K[1, 2] *= scale

                fx = float(
                    K[0, 0]
                )

                fy = float(
                    K[1, 1]
                )

                if (
                    not np.isfinite(fx)
                    or
                    not np.isfinite(fy)
                    or
                    fx <= 0
                    or
                    fy <= 0
                ):

                    print(
                        "[DENSE] Invalid focal length"
                    )

                    failed_pairs += 1
                    continue

                # =========================================
                # COLMAP CAMERA EXTRINSICS
                # =========================================

                R_cam1 = np.asarray(
                    global_rotations[i],
                    dtype=np.float64
                )

                t_cam1 = np.asarray(
                    global_translations[i],
                    dtype=np.float64
                ).reshape(3)

                R_cam2 = np.asarray(
                    global_rotations[i + 1],
                    dtype=np.float64
                )

                t_cam2 = np.asarray(
                    global_translations[i + 1],
                    dtype=np.float64
                ).reshape(3)

                if (
                    R_cam1.shape != (3, 3)
                    or
                    R_cam2.shape != (3, 3)
                ):

                    raise ValueError(
                        "Invalid COLMAP rotation matrix"
                    )

                if (
                    not np.all(
                        np.isfinite(R_cam1)
                    )
                    or
                    not np.all(
                        np.isfinite(R_cam2)
                    )
                    or
                    not np.all(
                        np.isfinite(t_cam1)
                    )
                    or
                    not np.all(
                        np.isfinite(t_cam2)
                    )
                ):

                    print(
                        "[DENSE] Invalid COLMAP "
                        "camera transform"
                    )

                    failed_pairs += 1
                    continue

                # =========================================
                # RELATIVE CAMERA GEOMETRY
                # =========================================
                #
                # COLMAP:
                #
                # X1 = R1 Xw + t1
                # X2 = R2 Xw + t2
                #
                # Therefore:
                #
                # X2 =
                # (R2 R1^T) X1
                # +
                # (t2 - R2 R1^T t1)
                # =========================================

                R_relative = (
                    R_cam2
                    @ R_cam1.T
                )

                T_relative = (
                    t_cam2
                    - R_relative @ t_cam1
                )

                relative_baseline = float(
                    np.linalg.norm(
                        T_relative
                    )
                )

                print(
                    "[DENSE] Relative baseline: "
                    f"{relative_baseline:.6f}"
                )
                # =========================================
                # GEOMETRY DIAGNOSTICS
                # =========================================

                print(
                    "[DEBUG-GEOM] T_relative: "
                    f"{T_relative}"
                )

                print(
                    "[DEBUG-GEOM] R_relative:\n"
                    f"{R_relative}"
                )

                if (
                    not np.isfinite(
                        relative_baseline
                    )
                    or
                    relative_baseline
                    <= MIN_BASELINE
                ):

                    print(
                        "[DENSE] Relative baseline "
                        "too small or invalid"
                    )

                    failed_pairs += 1
                    continue

                # =========================================
                # STEREO RECTIFICATION
                # =========================================

                h, w = (
                    working_img1.shape[:2]
                )

                image_size = (
                    w,
                    h
                )

                # Stage-2 assumption:
                # zero distortion.
                #
                # Later we can pass the actual COLMAP
                # distortion coefficients when auditing
                # camera models.
                dist_coeffs = np.zeros(
                    5,
                    dtype=np.float64
                )

                (
                    R_rect1,
                    R_rect2,
                    P_rect1,
                    P_rect2,
                    Q,
                    roi1,
                    roi2
                ) = cv2.stereoRectify(

                    K,
                    dist_coeffs,

                    K,
                    dist_coeffs,

                    image_size,

                    R_relative,
                    T_relative,

                    flags=cv2.CALIB_ZERO_DISPARITY,

                    alpha=0
                )
                print(
                    "[DEBUG-GEOM] P_rect1:\n"
                    f"{P_rect1}"
                )

                print(
                    "[DEBUG-GEOM] P_rect2:\n"
                    f"{P_rect2}"
                )

                print(
                    "[DEBUG-GEOM] Q:\n"
                    f"{Q}"
                )

                # =========================================
                # DETECT RECTIFIED STEREO ORIENTATION
                # =========================================

                rectified_tx = float(
                    P_rect2[0, 3]
                )

                rectified_ty = float(
                    P_rect2[1, 3]
                )

                print(
                    "[DEBUG-GEOM] Rectified Tx term: "
                    f"{rectified_tx:.6f}"
                )

                print(
                    "[DEBUG-GEOM] Rectified Ty term: "
                    f"{rectified_ty:.6f}"
                )

                # StereoSGBM searches along image rows.
                # Therefore, for this Stage-3 test we only
                # accept horizontally rectified stereo pairs.
                is_horizontal_stereo = (
                    abs(rectified_tx)
                    >= abs(rectified_ty)
                )

                if not is_horizontal_stereo:

                    print(
                        "[DENSE] Vertical stereo pair detected; "
                        "skipping pair for Stage-3"
                    )

                    failed_pairs += 1
                    continue

                if abs(rectified_tx) <= 1e-9:

                    print(
                        "[DENSE] Invalid horizontal rectified "
                        "baseline; skipping pair"
                    )

                    failed_pairs += 1
                    continue

                print(
                    "[DENSE] Stereo orientation: HORIZONTAL"
                )

                # =========================================
                # VALIDATE RECTIFICATION MATRICES
                # =========================================

                if (
                    not np.all(
                        np.isfinite(R_rect1)
                    )
                    or
                    not np.all(
                        np.isfinite(R_rect2)
                    )
                    or
                    not np.all(
                        np.isfinite(Q)
                    )
                ):

                    print(
                        "[DENSE] Invalid rectification "
                        "matrices"
                    )

                    failed_pairs += 1
                    continue

                # =========================================
                # BUILD RECTIFICATION MAPS
                # =========================================

                map1_x, map1_y = (
                    cv2.initUndistortRectifyMap(

                        K,
                        dist_coeffs,

                        R_rect1,
                        P_rect1,

                        image_size,

                        cv2.CV_32FC1
                    )
                )

                map2_x, map2_y = (
                    cv2.initUndistortRectifyMap(

                        K,
                        dist_coeffs,

                        R_rect2,
                        P_rect2,

                        image_size,

                        cv2.CV_32FC1
                    )
                )

                # =========================================
                # RECTIFY IMAGES
                # =========================================

                rectified_img1 = cv2.remap(

                    working_img1,

                    map1_x,
                    map1_y,

                    interpolation=cv2.INTER_LINEAR,

                    borderMode=cv2.BORDER_CONSTANT
                )

                rectified_img2 = cv2.remap(

                    working_img2,

                    map2_x,
                    map2_y,

                    interpolation=cv2.INTER_LINEAR,

                    borderMode=cv2.BORDER_CONSTANT
                )

                gray1 = cv2.cvtColor(
                    rectified_img1,
                    cv2.COLOR_BGR2GRAY
                )

                gray2 = cv2.cvtColor(
                    rectified_img2,
                    cv2.COLOR_BGR2GRAY
                )

                # =========================================
                # STEREO SGBM
                # =========================================

                block_size = 5

                NUM_DISPARITIES = 96

                # Decide which disparity direction to search

                if rectified_tx < 0.0:

                    min_disparity = 0
                    expected_disparity_sign = "positive"

                else:

                    min_disparity = -NUM_DISPARITIES
                    expected_disparity_sign = "negative"

                print(
                    "[DENSE] Expected disparity sign: "
                    f"{expected_disparity_sign}"
                )

                print(
                    "[DENSE] SGBM minDisparity: "
                    f"{min_disparity}"
                )

                stereo = cv2.StereoSGBM_create(

                    minDisparity=min_disparity,

                    numDisparities=NUM_DISPARITIES,

                    blockSize=block_size,

                    P1=8 * block_size * block_size,

                    P2=32 * block_size * block_size,

                    disp12MaxDiff=1,

                    uniquenessRatio=5,

                    speckleWindowSize=50,

                    speckleRange=2,

                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                )

                disparity = stereo.compute(
                    gray1,
                    gray2
                ).astype(
                    np.float32
                ) / 16.0
                finite_disparities = disparity[
                    np.isfinite(disparity)
                ]

                if finite_disparities.size > 0:

                    print(
                        "[DEBUG-STEREO] Disparity range: "
                        f"{finite_disparities.min():.4f}"
                        " -> "
                        f"{finite_disparities.max():.4f}"
                    )

                    positive_disp_count = int(
                        np.count_nonzero(
                            finite_disparities > 0.0
                        )
                    )

                    negative_disp_count = int(
                        np.count_nonzero(
                            finite_disparities < 0.0
                        )
                    )

                    zero_disp_count = int(
                        np.count_nonzero(
                            finite_disparities == 0.0
                        )
                    )

                    print(
                        "[DEBUG-STEREO] Positive disparities: "
                        f"{positive_disp_count:,}"
                    )

                    print(
                        "[DEBUG-STEREO] Negative disparities: "
                        f"{negative_disp_count:,}"
                    )

                    print(
                        "[DEBUG-STEREO] Zero disparities: "
                        f"{zero_disp_count:,}"
                    )



                    # =========================================
                    # VALID DISPARITY USING EXPECTED SIGN
                    # =========================================

                    if rectified_tx < 0.0:

                        valid_disparity = (
                            np.isfinite(disparity)
                            &
                            (disparity > 0.0)
                        )

                    else:

                        # For negative-disparity stereo,
                        # OpenCV's invalid value is normally
                        # below minDisparity. Keep only the
                        # searched negative-disparity range.

                        valid_disparity = (
                            np.isfinite(disparity)
                            &
                            (disparity < 0.0)
                            &
                            (disparity >= min_disparity)
                        )

                valid_disparity_count = int(
                    np.count_nonzero(
                        valid_disparity
                    )
                )

                print(
                    "[DENSE] Valid disparity pixels: "
                    f"{valid_disparity_count:,}"
                )

                if valid_disparity_count == 0:

                    print(
                        "[DENSE] No valid disparity"
                    )

                    failed_pairs += 1
                    continue

                # =========================================
                # DISPARITY -> RECTIFIED 3D
                # =========================================

                points_rectified = (
                    cv2.reprojectImageTo3D(
                        disparity,
                        Q
                    )
                )

                finite_xyz = np.all(
                    np.isfinite(
                        points_rectified
                    ),
                    axis=2
                )

                rectified_depth = (
                    points_rectified[:, :, 2]
                )
                finite_depth = rectified_depth[
                    np.isfinite(rectified_depth)
                ]

                if finite_depth.size > 0:

                    print(
                        "[DEBUG-DEPTH] Z range: "
                        f"{finite_depth.min():.4f}"
                        " -> "
                        f"{finite_depth.max():.4f}"
                    )

                    positive_z_count = int(
                        np.count_nonzero(
                            finite_depth > 0.0
                        )
                    )

                    negative_z_count = int(
                        np.count_nonzero(
                            finite_depth < 0.0
                        )
                    )

                    print(
                        "[DEBUG-DEPTH] Positive Z: "
                        f"{positive_z_count:,}"
                    )

                    print(
                        "[DEBUG-DEPTH] Negative Z: "
                        f"{negative_z_count:,}"
                    )

                # =========================================
                # VALID 3D MASK
                # =========================================
                #
                # No fixed MIN_DEPTH / MAX_DEPTH here.
                #
                # COLMAP's reconstruction scale is not
                # guaranteed to be metric.
                # =========================================

                valid_3d = (
                    valid_disparity
                    &
                    finite_xyz
                    &
                    np.isfinite(
                        rectified_depth
                    )
                    &
                    (rectified_depth > 0.0)
                )

                valid_3d_count = int(
                    np.count_nonzero(
                        valid_3d
                    )
                )

                print(
                    "[DENSE] Valid 3D pixels: "
                    f"{valid_3d_count:,}"
                )

                if valid_3d_count == 0:

                    print(
                        "[DENSE] No valid 3D points"
                    )

                    failed_pairs += 1
                    continue

                valid_depth_values = (
                    rectified_depth[
                        valid_3d
                    ]
                )

                print(
                    "[DENSE] Rectified depth range: "
                    f"{valid_depth_values.min():.4f}"
                    " -> "
                    f"{valid_depth_values.max():.4f}"
                )

                # =========================================
                # GENERATE WORLD-SPACE POINTS
                # =========================================

                pair_points = []

                pair_colors = []

                for y in range(
                    0,
                    h,
                    SAMPLE_STEP
                ):

                    for x in range(
                        0,
                        w,
                        SAMPLE_STEP
                    ):

                        if not valid_3d[y, x]:
                            continue

                        # ---------------------------------
                        # RECTIFIED CAMERA-1 COORDINATES
                        # ---------------------------------

                        point_rectified = np.asarray(
                            points_rectified[
                                y,
                                x
                            ],
                            dtype=np.float64
                        )

                        if not np.all(
                            np.isfinite(
                                point_rectified
                            )
                        ):

                            continue

                        # ---------------------------------
                        # UNDO RECTIFICATION
                        # ---------------------------------
                        #
                        # X_rect =
                        # R_rect1 X_camera1
                        #
                        # Therefore:
                        #
                        # X_camera1 =
                        # R_rect1^T X_rect
                        # ---------------------------------

                        point_camera1 = (
                            R_rect1.T
                            @ point_rectified
                        )

                        # ---------------------------------
                        # CAMERA-1 -> COLMAP WORLD
                        # ---------------------------------
                        #
                        # COLMAP:
                        #
                        # X_camera1 =
                        # R_cam1 X_world + t_cam1
                        #
                        # Therefore:
                        #
                        # X_world =
                        # R_cam1^T
                        # (X_camera1 - t_cam1)
                        # ---------------------------------

                        point_world = (
                            R_cam1.T
                            @ (
                                point_camera1
                                - t_cam1
                            )
                        )

                        if not np.all(
                            np.isfinite(
                                point_world
                            )
                        ):

                            continue

                        pair_points.append(
                            point_world
                        )

                        # x/y correspond to the rectified
                        # image, so sample color there.
                        bgr = rectified_img1[
                            y,
                            x
                        ]

                        pair_colors.append([
                            float(
                                bgr[2]
                            ) / 255.0,

                            float(
                                bgr[1]
                            ) / 255.0,

                            float(
                                bgr[0]
                            ) / 255.0
                        ])

                # =========================================
                # STORE PAIR
                # =========================================

                if len(pair_points) == 0:

                    print(
                        "[DENSE] Pair produced no "
                        "usable world-space points"
                    )

                    failed_pairs += 1
                    continue

                point_chunks.append(
                    np.asarray(
                        pair_points,
                        dtype=np.float64
                    )
                )

                color_chunks.append(
                    np.asarray(
                        pair_colors,
                        dtype=np.float64
                    )
                )
                successful_pairs += 1

                print(
                    "[DENSE] Generated "
                    f"{len(pair_points):,} points"
                )

                # =========================================
                # MEMORY CLEANUP
                # =========================================

                del pair_points
                del pair_colors

                del disparity

                del gray1
                del gray2

                del rectified_img1
                del rectified_img2

                del map1_x
                del map1_y

                del map2_x
                del map2_y

                del working_img1
                del working_img2

                del stereo

                del points_rectified

                del valid_3d

                del img1
                del img2


                gc.collect()

            except cv2.error as e:

                failed_pairs += 1

                print(
                    f"[DENSE] OpenCV error for pair "
                    f"{i}-{i + 1}: {e}"
                )

            except Exception as e:

                failed_pairs += 1

                print(
                    f"[DENSE] Pair "
                    f"{i}-{i + 1} failed: {e}"
                )

        # =================================================
        # FINALIZE DENSE CLOUD
        # =================================================

        if len(point_chunks) == 0:

            print(
                "\n[DENSE] No dense points generated"
            )

            return None

        self.dense_points = np.vstack(
            point_chunks
        ).astype(np.float64)

        self.dense_colors = np.vstack(
            color_chunks
        ).astype(np.float64)

        point_chunks.clear()
        color_chunks.clear()
        # =================================================
        # REMOVE INVALID VALUES
        # =================================================

        finite_mask = (
            np.all(
                np.isfinite(
                    self.dense_points
                ),
                axis=1
            )
            &
            np.all(
                np.isfinite(
                    self.dense_colors
                ),
                axis=1
            )
        )

        self.dense_points = (
            self.dense_points[
                finite_mask
            ]
        )

        self.dense_colors = (
            self.dense_colors[
                finite_mask
            ]
        )

        self.dense_colors = np.clip(
            self.dense_colors,
            0.0,
            1.0
        )

        # =================================================
        # MEMORY LIMIT
        # =================================================

        if (
            len(self.dense_points)
            > MAX_DENSE_POINTS
        ):

            print(
                "\n[DENSE] Reducing dense cloud "
                f"from {len(self.dense_points):,} "
                f"to {MAX_DENSE_POINTS:,} points"
            )

            rng = np.random.default_rng(
                42
            )

            indices = rng.choice(
                len(self.dense_points),
                MAX_DENSE_POINTS,
                replace=False
            )

            self.dense_points = (
                self.dense_points[
                    indices
                ]
            )

            self.dense_colors = (
                self.dense_colors[
                    indices
                ]
            )

        # =================================================
        # SUMMARY
        # =================================================

        print(
            "\n========== DENSE SUMMARY =========="
        )

        print(
            "[DENSE] Successful pairs: "
            f"{successful_pairs}"
        )

        print(
            "[DENSE] Failed pairs: "
            f"{failed_pairs}"
        )

        print(
            "[DENSE] Final dense points: "
            f"{len(self.dense_points):,}"
        )

        print(
            "===================================\n"
        )

        return {

            "points":
                self.dense_points,

            "colors":
                self.dense_colors,

            "successful_pairs":
                successful_pairs,

            "failed_pairs":
                failed_pairs
        }

    # =====================================================
    # OPEN3D POINT CLOUD
    # =====================================================

    def create_open3d_cloud(self):

        if len(self.dense_points) == 0:

            return None

        pcd = (
            o3d.geometry.PointCloud()
        )

        pcd.points = (
            o3d.utility.Vector3dVector(
                self.dense_points
            )
        )

        pcd.colors = (
            o3d.utility.Vector3dVector(
                self.dense_colors
            )
        )

        return pcd
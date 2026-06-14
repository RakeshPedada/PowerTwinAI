import numpy as np
import cv2
import open3d as o3d


class DenseReconstructor:

    def __init__(self):

        self.dense_points = []
        self.dense_colors = []

    # =====================================================
    # DENSE POINT CLOUD GENERATION
    # =====================================================

    def generate_dense_cloud(
        self,
        images,
        camera_poses,
        intrinsic_matrix,
        global_rotations=None,
        global_translations=None
    ):
        print("\n===== CAMERA TRANSFORM DEBUG =====")

        if global_rotations is not None:
            print(
                "Rotations:",
                len(global_rotations)
            )

        if global_translations is not None:
            print(
                "Translations:",
                len(global_translations)
            )

        print("===============================\n")

        print("\n========== DENSE RECONSTRUCTION ==========")

        if len(images) < 2:
            print("Not enough images")
            return None

        all_points = []
        all_colors = []

        max_pairs = min(
        len(images) - 1,
        len(camera_poses) - 1
        )

        for i in range(max_pairs):

            try:

                img1 = images[i]
                img2 = images[i + 1]

                print(f"\nProcessing dense pair {i}-{i+1}")

                gray1 = cv2.cvtColor(
                    img1,
                    cv2.COLOR_BGR2GRAY
                )

                gray2 = cv2.cvtColor(
                    img2,
                    cv2.COLOR_BGR2GRAY
                )
                # =====================================
                # DOWNSAMPLE FOR DENSE RECONSTRUCTION
                # =====================================

                MAX_WIDTH = 1600

                if gray1.shape[1] > MAX_WIDTH:

                    scale = MAX_WIDTH / gray1.shape[1]

                    gray1 = cv2.resize(
                        gray1,
                        None,
                        fx=scale,
                        fy=scale
                    )

                    gray2 = cv2.resize(
                        gray2,
                        None,
                        fx=scale,
                        fy=scale
                    )

                    img1 = cv2.resize(
                        img1,
                        None,
                        fx=scale,
                        fy=scale
                    )

                    img2 = cv2.resize(
                        img2,
                        None,
                        fx=scale,
                        fy=scale
                    )

                    focal_length = (
                        intrinsic_matrix[0, 0] * scale
                    )

                else:

                    focal_length = intrinsic_matrix[0, 0]                

                # =====================================================
                # STEREO MATCHER
                # =====================================================

                stereo = cv2.StereoSGBM_create(

                    minDisparity=0,

                    numDisparities=128,

                    blockSize=5,

                    P1=8 * 3 * 5**2,

                    P2=32 * 3 * 5**2,

                    disp12MaxDiff=1,

                    uniquenessRatio=5,

                    speckleWindowSize=50,

                    speckleRange=2
                )

                disparity = stereo.compute(
                    gray1,
                    gray2
                ).astype(np.float32) / 16.0

                disparity[disparity <= 0] = np.nan

                h, w = gray1.shape

                

                baseline = np.linalg.norm(
                    camera_poses[i + 1] -
                    camera_poses[i]
                )
                print(
                    f"Pair {i}-{i+1} Baseline = {baseline:.4f}"
                )

                baseline = max(baseline, 0.01)                

                depth = (
                    focal_length * baseline
                ) / disparity
                depth = np.clip(
                        depth,
                        0.05,
                        20.0
                    )
                depth = np.nan_to_num(
                    depth,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )
                valid_depths = depth[
                    np.isfinite(depth)
                ]

                print(
                    f"Depth Range: "
                    f"{valid_depths.min():.3f} -> "
                    f"{valid_depths.max():.3f}"
                )
                points = []
                colors = []

                # =====================================================
                # GENERATE 3D POINTS
                # =====================================================

                for y in range(0, h, 4):

                    for x in range(0, w, 4):

                        z = depth[y, x]
                        if z < 0.1:
                            continue

                        if (
                            z <= 0 or
                            z > 20.0 or
                            np.isnan(z) or
                            np.isinf(z)
                        ):
                            continue

                        X = (
                            (x - intrinsic_matrix[0, 2])
                            * z / focal_length
                        )

                        Y = (
                            (y - intrinsic_matrix[1, 2])
                            * z / focal_length
                        )

                        local_point = np.array([
                            X,
                            Y,
                            z
                        ])

                        if (
                            global_rotations is not None and
                            global_translations is not None and
                            i in global_rotations and
                            i in global_translations
                        ):

                            R = global_rotations[i]

                            t = global_translations[i].flatten()

                            world_point = (
                                R @ local_point
                            ) + t

                            points.append(
                                world_point
                            )

                        else:

                            points.append(
                                local_point
                            )

                        bgr = img1[y, x]

                        rgb = [
                            bgr[2] / 255.0,
                            bgr[1] / 255.0,
                            bgr[0] / 255.0
                        ]

                        colors.append(rgb)

                if len(points) > 0:

                    all_points.extend(points)
                    all_colors.extend(colors)

                    print(
                        f"Generated {len(points)} dense points"
                    )

            except Exception as e:

                print(
                    f"Dense reconstruction error: {str(e)}"
                )

        if len(all_points) == 0:

            print("No dense points generated")
            return None

        # =====================================================
        # CONVERT TO NUMPY
        # =====================================================

        self.dense_points = np.array(all_points)

        self.dense_colors = np.array(all_colors)

        # =====================================================
        # LIMIT DENSE POINTS
        # =====================================================

        if len(self.dense_points) > 2000000:

            print("\nReducing dense cloud size...")

            indices = np.random.choice(
                len(self.dense_points),
                2000000,
                replace=False
            )

            self.dense_points = (
                self.dense_points[indices]
            )

            self.dense_colors = (
                self.dense_colors[indices]
            )

            print(
                f"Reduced to "
                f"{len(self.dense_points)} points"
            )
        print(
            f"\nTotal Dense Points: "
            f"{len(self.dense_points)}"
        )

        return {

            "points": self.dense_points,

            "colors": self.dense_colors
        }

    # =====================================================
    # OPEN3D POINT CLOUD
    # =====================================================

    def create_open3d_cloud(self):

        if len(self.dense_points) == 0:
            return None

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(
            self.dense_points
        )

        pcd.colors = o3d.utility.Vector3dVector(
            self.dense_colors
        )

        return pcd

    # =====================================================
    # POISSON MESH RECONSTRUCTION
    # =====================================================

    def create_mesh_from_cloud(self):

        if len(self.dense_points) == 0:

            print("No dense points available")

            return None

        print(
            "\n========== "
            "CREATING POISSON SURFACE "
            "=========="
        )

        try:

            # =====================================================
            # CREATE POINT CLOUD
            # =====================================================

            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(
                self.dense_points
            )

            pcd.colors = o3d.utility.Vector3dVector(
                self.dense_colors
            )

            # =====================================================
            # REMOVE OUTLIERS
            # =====================================================

            print("Removing outliers...")

            pcd, ind = pcd.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=2.0
            )

            # =====================================================
            # VOXEL DOWNSAMPLE
            # =====================================================

            print("Downsampling cloud...")

            pcd = pcd.voxel_down_sample(
                voxel_size=0.01
            )

            # =====================================================
            # NORMAL ESTIMATION
            # =====================================================

            print("Estimating normals...")

            pcd.estimate_normals(

                search_param=o3d.geometry.
                KDTreeSearchParamHybrid(

                    radius=0.05,

                    max_nn=20,
                )
            )

            pcd.orient_normals_consistent_tangent_plane(
                100
            )

            # =====================================================
            # POISSON RECONSTRUCTION
            # =====================================================

            print("Generating Poisson mesh...")

            mesh, densities = (
                o3d.geometry.TriangleMesh
                .create_from_point_cloud_poisson(

                    pcd,

                    depth=7
                )
            )

            # =====================================================
            # REMOVE LOW DENSITY AREAS
            # =====================================================

            print("Cleaning mesh...")

            densities = np.asarray(densities)

            density_threshold = np.quantile(
                densities,
                0.08
            )

            vertices_to_remove = (
                densities < density_threshold
            )

            mesh.remove_vertices_by_mask(
                vertices_to_remove
            )

            # =====================================================
            # SMOOTH MESH
            # =====================================================

            print("Smoothing mesh...")

            mesh = mesh.filter_smooth_taubin(
                number_of_iterations=10
            )

            mesh.compute_vertex_normals()

            print("\nMesh created successfully")

            return mesh

        except Exception as e:

            print(
                f"Mesh generation failed: {str(e)}"
            )

            return None
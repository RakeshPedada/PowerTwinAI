import os
import sys
import time
import shutil
import sqlite3
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml import ImagePairRanker, MLPairRankingConfig

def main():
    print("=== PowerTwinAI: ML vs Exhaustive Benchmark ===")
    img_dir = PROJECT_ROOT / "colmap_workspace" / "images"
    img_paths = sorted([str(p) for p in img_dir.glob("*.jpg")])
    total_images = len(img_paths)
    print(f"Dataset images: {total_images}")

    # 1. ML Pair Ranking
    cfg = MLPairRankingConfig(
        enabled=True,
        top_k=8,
        min_pairs_per_image=4,
        auto_bridge_components=True,
        ensure_connected=True
    )
    ranker = ImagePairRanker(cfg)
    t0 = time.perf_counter()
    rank_res = ranker.rank_pairs(img_paths)
    t_rank = time.perf_counter() - t0

    candidate_pairs = rank_res["candidate_pairs"]
    diag = rank_res["diagnostics"]
    ex_pairs = diag["exhaustive_pairs"]
    print(f"\n[1] ML Pair Ranking completed in {t_rank:.3f}s")
    print(f"    - Baseline Exhaustive Pairs: {ex_pairs}")
    print(f"    - Selected Candidate Pairs: {len(candidate_pairs)}")
    print(f"    - Reduction: {diag['pair_reduction_pct']}%")
    print(f"    - Connected Components: {diag['connected_components']}")
    print(f"    - Degree: min={diag['min_degree']}, mean={diag['avg_degree']}, max={diag['max_degree']}")

    # Save candidate pairs
    pairs_txt = PROJECT_ROOT / "temp_session" / "benchmark_ml_pairs.txt"
    pairs_txt.parent.mkdir(parents=True, exist_ok=True)
    ranker.save_colmap_pairs(candidate_pairs, str(pairs_txt))

    # 2. Setup benchmark DB using already extracted keypoints
    db_source = PROJECT_ROOT / "colmap_workspace" / "database.db"
    db_bench = PROJECT_ROOT / "temp_session" / "benchmark_ml.db"
    if db_bench.exists():
        os.remove(db_bench)
    shutil.copy2(db_source, db_bench)

    con = sqlite3.connect(db_bench)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=DELETE")
    cur.execute("DELETE FROM matches")
    cur.execute("DELETE FROM two_view_geometries")
    con.commit()
    con.close()
    print("\n[2] Cloned database.db and reset matches/geometries.")

    # 3. Run COLMAP matches_importer
    colmap_bin = r"C:\COLMAP\bin\colmap.exe"
    cmd = [
        colmap_bin, "matches_importer",
        "--database_path", str(db_bench),
        "--match_list_path", str(pairs_txt),
        "--match_type", "pairs",
        "--SiftMatching.use_gpu", "0",
        "--SiftMatching.num_threads", "4",
        "--SiftMatching.max_num_matches", "32768"
    ]
    print("\n[3] Launching COLMAP matches_importer on 386 candidate pairs...")
    t_match_start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=False)
    t_match = time.perf_counter() - t_match_start

    print(f"    - Returncode: {proc.returncode}")
    print(f"    - COLMAP Matching Duration: {t_match:.2f}s ({t_match/60.0:.2f} min)")

    # 4. Verify verified two-view geometries
    con = sqlite3.connect(db_bench)
    cur = con.cursor()
    n_matches = cur.execute("SELECT count(pair_id) FROM matches").fetchone()[0]
    n_geom = cur.execute("SELECT count(pair_id) FROM two_view_geometries").fetchone()[0]
    con.close()
    print(f"\n[4] Database Verification:")
    print(f"    - Matched pairs: {n_matches}")
    print(f"    - Geometrically verified two-view pairs: {n_geom}")

    # 5. Sparse Reconstruction (mapper) test
    sparse_out = PROJECT_ROOT / "temp_session" / "benchmark_sparse"
    if sparse_out.exists():
        shutil.rmtree(sparse_out)
    sparse_out.mkdir(parents=True, exist_ok=True)

    mapper_cmd = [
        colmap_bin, "mapper",
        "--database_path", str(db_bench),
        "--image_path", str(img_dir),
        "--output_path", str(sparse_out),
        "--Mapper.num_threads", "4",
        "--Mapper.init_min_tri_angle", "8.0",
        "--Mapper.multiple_models", "0",
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_extra_params", "1",
    ]
    print("\n[5] Launching COLMAP mapper on candidate pair match graph...")
    t_map_start = time.perf_counter()
    proc_map = subprocess.run(mapper_cmd, capture_output=False)
    t_map = time.perf_counter() - t_map_start
    print(f"    - Mapper duration: {t_map:.2f}s ({t_map/60.0:.2f} min)")
    print(f"    - Mapper returncode: {proc_map.returncode}")

    # Check model 0
    model0 = sparse_out / "0"
    if model0.exists():
        print(f"    - Model 0 created successfully at {model0}!")
        for f in model0.iterdir():
            print(f"      * {f.name} ({f.stat().st_size} bytes)")
    else:
        print(f"    - Warning: Model 0 not created. Output dirs: {[p.name for p in sparse_out.iterdir()]}")

if __name__ == "__main__":
    main()

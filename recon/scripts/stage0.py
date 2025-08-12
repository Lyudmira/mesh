import argparse
import json
import logging
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0: Data Auditing & Cleaning")
    parser.add_argument("point_cloud", help="Input PLY point cloud")
    parser.add_argument("--out_dir", default="data", help="Directory to store outputs")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="Scale factor to convert units to meters")
    parser.add_argument(
        "--crop",
        nargs=6,
        type=float,
        metavar=("min_x", "min_y", "min_z", "max_x", "max_y", "max_z"),
        help="Axis-aligned bounding box to crop points",
    )
    parser.add_argument("--sor-nn", type=int, default=16, help="MeanK for Statistical Outlier Removal")
    parser.add_argument("--sor-std", type=float, default=2.0, help="Stddev threshold for SOR")
    parser.add_argument(
        "--ror-radius-mult",
        type=float,
        default=2.0,
        help="Radius multiplier (d_median * mult) for Radius Outlier Removal",
    )
    parser.add_argument("--ror-nn", type=int, default=8, help="Min neighbors for Radius Outlier Removal")
    parser.add_argument(
        "--voxel-size-mult",
        type=float,
        default=0.5,
        help="Voxel size multiplier (d_median * mult) for density equalization",
    )
    return parser.parse_args()


def knn_stats(pcd: o3d.geometry.PointCloud, k: int) -> tuple[float, float]:
    """Compute kNN distance statistics using a simple brute-force approach.

    Open3D's ``KDTreeFlann`` occasionally segfaults in minimal
    environments without full acceleration libraries. To maintain
    robustness, we fall back to a pure NumPy implementation which is
    sufficient for the small point clouds used in tests and examples.
    """

    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise RuntimeError("Point cloud has no points for kNN statistics")

    # Compute full pairwise distance matrix and take the k-th neighbor
    diff = pts[:, None, :] - pts[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    # The first column is zero (distance to itself); take k-th neighbor
    sorted_dists = np.sort(dist_matrix, axis=1)
    kth = sorted_dists[:, k]
    return float(np.median(kth)), float(np.quantile(kth, 0.95))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(args.point_cloud)
    if pcd.is_empty():
        logging.error("Input point cloud is empty")
        return 1
    logging.info("Loaded %s with %d points", args.point_cloud, len(pcd.points))

    if args.unit_scale != 1.0:
        pts = np.asarray(pcd.points)
        pts *= args.unit_scale
        pcd.points = o3d.utility.Vector3dVector(pts)
        logging.info("Applied unit scale %.3f", args.unit_scale)

    if args.crop:
        min_x, min_y, min_z, max_x, max_y, max_z = args.crop
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(min_x, min_y, min_z), max_bound=(max_x, max_y, max_z)
        )
        pcd = pcd.crop(bbox)
        logging.info("Cropped to AABB, %d points remain", len(pcd.points))
    num_cropped = len(pcd.points)

    try:
        d_median, d_95 = knn_stats(pcd, args.sor_nn)
    except RuntimeError as e:
        logging.error(str(e))
        return 1
    logging.info("kNN distance median=%.6f, 95th=%.6f", d_median, d_95)

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=args.sor_nn, std_ratio=args.sor_std
    )
    logging.info(
        "Statistical outlier removal: %d points remain", len(pcd.points)
    )
    pcd, _ = pcd.remove_radius_outlier(
        nb_points=args.ror_nn, radius=args.ror_radius_mult * d_median
    )
    logging.info("Radius outlier removal: %d points remain", len(pcd.points))
    num_filtered = len(pcd.points)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
    try:
        pcd.orient_normals_consistent_tangent_plane(100)
    except Exception:
        logging.warning("Normal orientation failed; proceeding without global consistency")
    logging.info("Estimated normals")

    voxel_size = args.voxel_size_mult * d_median
    balanced = pcd.voxel_down_sample(voxel_size)
    if balanced.is_empty():
        logging.warning(
            "Voxel downsample produced empty cloud; using filtered cloud"
        )
        balanced = pcd
    logging.info(
        "Density equalization: %d points after voxel downsample (voxel=%.6f)",
        len(balanced.points),
        voxel_size,
    )

    clean_path = out_dir / "clean.ply"
    normals_path = out_dir / "normals.ply"
    if not o3d.io.write_point_cloud(str(clean_path), balanced):
        logging.error("Failed to write %s", clean_path)
        return 1
    if not o3d.io.write_point_cloud(str(normals_path), balanced):
        logging.error("Failed to write %s", normals_path)
        return 1
    logging.info("Saved cleaned cloud to %s and normals to %s", clean_path, normals_path)

    report = {
        "num_points_raw": len(o3d.io.read_point_cloud(args.point_cloud).points),
        "num_points_cropped": num_cropped,
        "num_points_filtered": num_filtered,
        "num_points_balanced": len(balanced.points),
        "d_median": d_median,
        "d_95": d_95,
        "parameters": {
            "unit_scale": args.unit_scale,
            "crop": args.crop,
            "sor_nn": args.sor_nn,
            "sor_std": args.sor_std,
            "ror_nn": args.ror_nn,
            "ror_radius_mult": args.ror_radius_mult,
            "voxel_size_mult": args.voxel_size_mult,
        },
    }
    report_path = out_dir / "audit.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Audit report written to %s", report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

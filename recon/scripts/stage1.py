import argparse
import json
import logging
from pathlib import Path
import os

import numpy as np
import open3d as o3d


def try_import_maxflow():
    try:
        import maxflow  # type: ignore

        return maxflow
    except Exception as e:  # pragma: no cover - import guard
        logging.warning("pymaxflow unavailable (%s); falling back to threshold labels", e)
        return None


def try_import_skimage_measure():
    try:
        from skimage import measure  # type: ignore

        return measure
    except Exception as e:  # pragma: no cover - import guard
        logging.warning(
            "scikit-image unavailable (%s); falling back to Poisson meshing", e
        )
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: Building Shell Reconstruction")
    parser.add_argument("point_cloud", help="Input PLY point cloud (cleaned)")
    parser.add_argument("--out_dir", default="outputs", help="Directory to store outputs")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.1,
        help="Voxel size for UDF grid in meters",
    )
    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=8,
        help="Depth parameter for Poisson reconstruction",
    )
    return parser.parse_args()


def compute_udf(
    pts: np.ndarray,
    voxel: float,
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    out_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a simple unsigned distance field on a regular grid.

    Uses a brute-force nearest neighbour search for robustness since
    Open3D's KDTree may not be available or stable in minimal
    environments. Returns ``(dims, udf)``.
    """

    dims = np.ceil((max_bound - min_bound) / voxel).astype(int) + 1
    if np.prod(dims) > 1_000_000:
        raise RuntimeError(
            f"UDF grid {dims.tolist()} is too large; increase voxel size"
        )
    logging.info("Computing UDF grid %s with voxel %.3f", dims.tolist(), voxel)
    udf = np.empty(dims, dtype=np.float32)
    for idx in np.ndindex(*dims):
        center = min_bound + np.array(idx) * voxel
        # brute-force distance to all points
        dist = np.linalg.norm(pts - center, axis=1)
        udf[idx] = float(dist.min()) if len(dist) else np.inf
    np.save(out_dir / "udf.npy", udf)
    logging.info("Saved UDF grid to %s", out_dir / "udf.npy")
    return dims, udf


def graph_cut_labels(udf: np.ndarray, thresh: float) -> np.ndarray:
    """Segment voxels into interior/exterior using max-flow/graph cut.

    Falls back to simple thresholding if ``pymaxflow`` is unavailable.
    """

    maxflow = try_import_maxflow()
    if maxflow is None:  # pragma: no cover - import guard
        return (udf < thresh).astype(np.uint8)

    dims = udf.shape
    num_nodes = int(np.prod(dims))
    g = maxflow.GraphFloat()
    nodes = g.add_nodes(num_nodes)

    def nid(i: int, j: int, k: int) -> int:
        return (i * dims[1] + j) * dims[2] + k

    smooth = 0.1
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                n = nid(i, j, k)
                val = float(udf[i, j, k])
                g.add_tedge(n, val, max(0.0, thresh - val))
                if i + 1 < dims[0]:
                    g.add_edge(n, nid(i + 1, j, k), smooth, smooth)
                if j + 1 < dims[1]:
                    g.add_edge(n, nid(i, j + 1, k), smooth, smooth)
                if k + 1 < dims[2]:
                    g.add_edge(n, nid(i, j, k + 1), smooth, smooth)

    logging.info("Running graph cut with %d nodes", num_nodes)
    g.maxflow()
    labels = np.zeros(dims, dtype=np.uint8)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                labels[i, j, k] = 1 if g.get_segment(nid(i, j, k)) == 0 else 0
    if labels.min() == labels.max():
        logging.warning("Graph cut produced uniform labels; using threshold fallback")
        return (udf < thresh).astype(np.uint8)
    return labels


def reconstruct_from_labels(
    labels: np.ndarray, voxel: float, min_bound: np.ndarray
) -> o3d.geometry.TriangleMesh | None:
    """Extract a mesh from voxel labels using marching cubes.

    Disabled by default for robustness; set the environment variable
    ``STAGE1_USE_MC=1`` to enable. Returns ``None`` if unavailable or on
    failure.
    """

    measure = try_import_skimage_measure()
    if measure is None or os.environ.get("STAGE1_USE_MC") != "1":
        return None
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            labels, level=0.5, spacing=(voxel, voxel, voxel)
        )
    except Exception as e:  # pragma: no cover - robustness
        logging.warning("Marching cubes failed (%s)", e)
        return None
    verts = verts + min_bound
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


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

    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
        try:
            pcd.orient_normals_consistent_tangent_plane(100)
        except Exception:
            logging.warning(
                "Normal orientation failed; proceeding without global consistency"
            )
        logging.info("Estimated normals")

    pts = np.asarray(pcd.points)
    min_bound = pts.min(axis=0)
    max_bound = pts.max(axis=0)

    try:
        dims, udf = compute_udf(
            pts, args.voxel_size, min_bound, max_bound, out_dir
        )
    except RuntimeError as e:
        logging.error(str(e))
        return 1

    thresh = args.voxel_size * 1.5
    labels = graph_cut_labels(udf, thresh)
    np.save(out_dir / "labels.npy", labels)
    logging.info("Saved graph cut labels to %s", out_dir / "labels.npy")

    mesh = reconstruct_from_labels(labels, args.voxel_size, min_bound)
    method = "marching_cubes" if mesh is not None else "poisson"
    if mesh is None:
        logging.info(
            "Falling back to Poisson reconstruction with depth=%d", args.poisson_depth
        )
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=args.poisson_depth
        )
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
    else:
        logging.info("Mesh extracted via marching cubes")

    mesh_path = out_dir / "mesh_shell.ply"
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        logging.error("Failed to write %s", mesh_path)
        return 1
    logging.info(
        "Mesh written to %s (%d verts, %d tris)",
        mesh_path,
        len(mesh.vertices),
        len(mesh.triangles),
    )

    report = {
        "num_points": len(pcd.points),
        "voxel_size": args.voxel_size,
        "poisson_depth": args.poisson_depth,
        "meshing_method": method,
        "grid_dims": [int(d) for d in dims],
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
    }
    report_path = out_dir / "stage1_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info("Report saved to %s", report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

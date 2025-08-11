import subprocess
import sys
from pathlib import Path

import open3d as o3d
import numpy as np


def test_stage1(tmp_path: Path) -> None:
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(2000)
    cloud = tmp_path / "sphere.ply"
    assert o3d.io.write_point_cloud(str(cloud), pcd)

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "recon/scripts/stage1.py",
        str(cloud),
        "--out_dir",
        str(out_dir),
        "--voxel-size",
        "0.2",
        "--poisson-depth",
        "6",
    ]
    subprocess.run(cmd, check=True)

    mesh_path = out_dir / "mesh_shell.ply"
    assert mesh_path.is_file()
    assert (out_dir / "udf.npy").is_file()
    assert (out_dir / "labels.npy").is_file()
    assert (out_dir / "stage1_report.json").is_file()

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0

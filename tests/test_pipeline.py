import os
import subprocess
from pathlib import Path

import open3d as o3d
import pymeshlab


def compile_pipeline(src: Path, exe: Path) -> None:
    conda_prefix = os.environ["CONDA_PREFIX"]
    include_dir = Path(conda_prefix, "include")

    # Flags from pkg-config and pcl_config for robust builds
    openvdb_flags = subprocess.check_output([
        "pkg-config", "--cflags", "--libs", "openvdb"
    ], text=True).split()

    try:
        pcl_cflags = subprocess.check_output(["pcl_config", "--cflags"], text=True).split()
        pcl_libs = subprocess.check_output(["pcl_config", "--libs"], text=True).split()
    except Exception:
        pcl_cflags = []
        pcl_libs = [
            "-lpcl_io", "-lpcl_common", "-lpcl_filters", "-lpcl_kdtree",
            "-lpcl_search", "-lpcl_features",
        ]

    try:
        cgal_flags = subprocess.check_output([
            "pkg-config", "--cflags", "--libs", "cgal"
        ], text=True).split()
    except Exception:
        cgal_flags = ["-lCGAL", "-lgmp", "-lmpfr"]

    compile_cmd = [
        "g++", str(src), "-std=c++17",
        f"-I{include_dir}", f"-I{include_dir/'eigen3'}",
    ] + openvdb_flags + pcl_cflags + cgal_flags + [
        f"-L{conda_prefix}/lib",
    ] + pcl_libs + ["-o", str(exe)]

    subprocess.run(compile_cmd, check=True)


def test_full_pipeline(tmp_path):
    src = Path("recon/src/pipeline.cpp")
    exe = tmp_path / "pipeline"
    compile_pipeline(src, exe)

    # Generate a simple sphere point cloud using Open3D
    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).sample_points_poisson_disk(500)
    ply_path = tmp_path / "points.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)

    out_mesh = tmp_path / "mesh.obj"
    run_proc = subprocess.run([str(exe), str(ply_path), str(out_mesh)], capture_output=True, text=True)
    assert run_proc.returncode == 0, run_proc.stderr
    print(run_proc.stdout.strip())

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(out_mesh))
    mesh = ms.current_mesh()
    print(f"Resulting mesh: {mesh.vertex_number()} vertices, {mesh.face_number()} faces")
    assert mesh.vertex_number() > 0
    assert mesh.face_number() > 0

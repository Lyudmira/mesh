import json
import subprocess
from pathlib import Path

import open3d as o3d


def test_stage0(tmp_path: Path) -> None:
    # generate synthetic sphere point cloud
    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).sample_points_poisson_disk(500)
    input_path = tmp_path / "points.ply"
    o3d.io.write_point_cloud(str(input_path), pcd)

    out_dir = tmp_path / "out"
    run = subprocess.run(
        ["python", "recon/scripts/stage0.py", str(input_path), "--out_dir", str(out_dir)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    print(run.stdout.strip())

    clean = out_dir / "clean.ply"
    normals = out_dir / "normals.ply"
    report = out_dir / "audit.json"
    assert clean.exists()
    assert normals.exists()
    assert report.exists()

    with open(report) as f:
        data = json.load(f)
    assert "d_median" in data
    assert data["num_points_balanced"] > 0

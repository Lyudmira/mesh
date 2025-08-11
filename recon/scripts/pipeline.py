"""Simple wrapper around the C++ reconstruction pipeline.

It demonstrates how the compiled binary can be invoked from Python and
uses Open3D and PyMeshLab to load the resulting mesh so the environment
covering all dependencies is exercised.
"""

import argparse
import subprocess
import open3d as o3d
import pymeshlab


def main() -> None:
    parser = argparse.ArgumentParser(description="Point cloud to mesh pipeline")
    parser.add_argument("point_cloud", help="Input PLY point cloud")
    parser.add_argument("mesh", help="Output OBJ mesh path")
    parser.add_argument("--binary", default="pipeline", help="Path to compiled C++ binary")
    args = parser.parse_args()

    subprocess.check_call([args.binary, args.point_cloud, args.mesh])

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    print(f"Open3D loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(args.mesh)
    current = ms.current_mesh()
    print(f"PyMeshLab reports {current.vertex_number()} vertices and {current.face_number()} faces")


if __name__ == "__main__":
    main()

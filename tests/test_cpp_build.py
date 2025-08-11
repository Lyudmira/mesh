import os
import subprocess
import textwrap
from pathlib import Path


def test_cpp_compilation(tmp_path):
    """Compile and run a tiny C++ program using core libraries.

    The program prints the library versions so success is obvious in the
    test logs. We assert that the expected identifiers appear in the
    output for robustness.
    """

    code = textwrap.dedent(
        """
        #include <openvdb/version.h>
        #include <pcl/point_cloud.h>
        #include <CGAL/version.h>
        #include <iostream>
        int main() {
            std::cout << "OpenVDB " << OPENVDB_LIBRARY_VERSION_STRING << std::endl;
            std::cout << "PCL " << PCL_VERSION_PRETTY << std::endl;
            std::cout << "CGAL " << CGAL_VERSION_STR << std::endl;
            return 0;
        }
        """
    )
    src = tmp_path / "check.cpp"
    src.write_text(code)
    conda_prefix = os.environ["CONDA_PREFIX"]
    include_dir = Path(conda_prefix, "include")
    pcl_include = next(include_dir.glob("pcl-*/"), None)
    compile_cmd = [
        "g++",
        str(src),
        "-std=c++17",
        f"-I{include_dir}",
        f"-I{include_dir/'eigen3'}",
    ]
    if pcl_include:
        compile_cmd.append(f"-I{pcl_include}")
    compile_cmd += [
        f"-L{conda_prefix}/lib",
        "-o",
        str(tmp_path / "check"),
    ]
    compile_proc = subprocess.run(
        compile_cmd, capture_output=True, text=True
    )
    assert compile_proc.returncode == 0, compile_proc.stderr

    run_proc = subprocess.run(
        [str(tmp_path / "check")], capture_output=True, text=True
    )
    assert run_proc.returncode == 0, run_proc.stderr
    print(run_proc.stdout.strip())
    assert "OpenVDB" in run_proc.stdout
    assert "PCL" in run_proc.stdout
    assert "CGAL" in run_proc.stdout

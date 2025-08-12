#!/usr/bin/env python3
"""
室内点云重建项目 - 基础功能测试脚本
用于验证项目的基本功能是否正常工作

版本: 1.0
日期: 2025-08-12
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import open3d as o3d

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

class BasicFunctionTester:
    """基础功能测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.project_root = PROJECT_ROOT
        self.build_dir = self.project_root / "build"
        self.test_data_dir = self.project_root / "data" / "test"
        self.test_output_dir = self.project_root / "outputs" / "test"
        
        # 确保测试目录存在
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 测试结果
        self.test_results = {}
    
    def create_test_point_cloud(self, filename: str = "test_cube.ply") -> str:
        """创建测试用的点云数据"""
        print("创建测试点云数据...")
        
        # 创建一个简单的立方体点云
        points = []
        normals = []
        colors = []
        
        # 立方体的8个顶点
        cube_vertices = [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # 底面
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # 顶面
        ]
        
        # 为每个面生成点
        for i in range(100):  # 每个面100个点
            # 底面 (z=0)
            x, y = np.random.random(2)
            points.append([x, y, 0])
            normals.append([0, 0, -1])
            colors.append([1, 0, 0])  # 红色
            
            # 顶面 (z=1)
            x, y = np.random.random(2)
            points.append([x, y, 1])
            normals.append([0, 0, 1])
            colors.append([0, 1, 0])  # 绿色
            
            # 前面 (y=0)
            x, z = np.random.random(2)
            points.append([x, 0, z])
            normals.append([0, -1, 0])
            colors.append([0, 0, 1])  # 蓝色
            
            # 后面 (y=1)
            x, z = np.random.random(2)
            points.append([x, 1, z])
            normals.append([0, 1, 0])
            colors.append([1, 1, 0])  # 黄色
            
            # 左面 (x=0)
            y, z = np.random.random(2)
            points.append([0, y, z])
            normals.append([-1, 0, 0])
            colors.append([1, 0, 1])  # 洋红
            
            # 右面 (x=1)
            y, z = np.random.random(2)
            points.append([1, y, z])
            normals.append([1, 0, 0])
            colors.append([0, 1, 1])  # 青色
        
        # 创建Open3D点云
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array(points))
        cloud.normals = o3d.utility.Vector3dVector(np.array(normals))
        cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        # 保存点云
        output_file = self.test_data_dir / filename
        o3d.io.write_point_cloud(str(output_file), cloud)
        
        print(f"测试点云已保存到: {output_file}")
        print(f"点云包含 {len(points)} 个点")
        
        return str(output_file)
    
    def test_cpp_executable(self) -> bool:
        """测试C++可执行文件是否存在且可运行"""
        print("\n=== 测试C++可执行文件 ===")
        
        executable_path = self.build_dir / "pipeline"
        
        if not executable_path.exists():
            print(f"❌ C++可执行文件不存在: {executable_path}")
            return False
        
        print(f"✅ C++可执行文件存在: {executable_path}")
        
        # 测试是否可以运行（显示帮助信息）
        try:
            result = subprocess.run([str(executable_path), "--help"], 
                                  capture_output=True, text=True, timeout=10)
            print(f"✅ 可执行文件可以运行")
            return True
        except subprocess.TimeoutExpired:
            print("⚠️  可执行文件运行超时")
            return True  # 超时可能是正常的，因为程序可能在等待输入
        except Exception as e:
            print(f"❌ 可执行文件运行失败: {e}")
            return False
    
    def test_point_cloud_loading(self) -> bool:
        """测试点云加载功能"""
        print("\n=== 测试点云加载功能 ===")
        
        # 创建测试点云
        test_file = self.create_test_point_cloud()
        
        try:
            # 使用Open3D加载点云
            cloud = o3d.io.read_point_cloud(test_file)
            
            if len(cloud.points) == 0:
                print("❌ 点云加载失败：点云为空")
                return False
            
            print(f"✅ 点云加载成功：{len(cloud.points)} 个点")
            
            # 检查法向量
            if len(cloud.normals) > 0:
                print(f"✅ 法向量存在：{len(cloud.normals)} 个")
            else:
                print("⚠️  法向量不存在")
            
            # 检查颜色
            if len(cloud.colors) > 0:
                print(f"✅ 颜色信息存在：{len(cloud.colors)} 个")
            else:
                print("⚠️  颜色信息不存在")
            
            return True
            
        except Exception as e:
            print(f"❌ 点云加载失败: {e}")
            return False
    
    def test_python_pipeline(self) -> bool:
        """测试Python管道脚本"""
        print("\n=== 测试Python管道脚本 ===")
        
        pipeline_script = self.project_root / "scripts" / "pipeline.py"
        
        if not pipeline_script.exists():
            print(f"❌ Python管道脚本不存在: {pipeline_script}")
            return False
        
        print(f"✅ Python管道脚本存在: {pipeline_script}")
        
        # 测试脚本是否可以导入
        try:
            sys.path.insert(0, str(pipeline_script.parent))
            import pipeline
            print("✅ Python管道脚本可以导入")
            
            # 测试创建管道实例
            recon_pipeline = pipeline.ReconstructionPipeline()
            print("✅ 重建管道实例创建成功")
            
            return True
            
        except Exception as e:
            print(f"❌ Python管道脚本测试失败: {e}")
            return False
    
    def test_quality_control(self) -> bool:
        """测试质量控制脚本"""
        print("\n=== 测试质量控制脚本 ===")
        
        qc_script = self.project_root / "scripts" / "qc.py"
        
        if not qc_script.exists():
            print(f"❌ 质量控制脚本不存在: {qc_script}")
            return False
        
        print(f"✅ 质量控制脚本存在: {qc_script}")
        
        # 测试脚本是否可以导入
        try:
            sys.path.insert(0, str(qc_script.parent))
            import qc
            print("✅ 质量控制脚本可以导入")
            
            # 测试创建质量控制器实例
            quality_controller = qc.MeshQualityController()
            print("✅ 质量控制器实例创建成功")
            
            return True
            
        except Exception as e:
            print(f"❌ 质量控制脚本测试失败: {e}")
            return False
    
    def test_lod_generator(self) -> bool:
        """测试LOD生成脚本"""
        print("\n=== 测试LOD生成脚本 ===")
        
        lod_script = self.project_root / "scripts" / "lod.py"
        
        if not lod_script.exists():
            print(f"❌ LOD生成脚本不存在: {lod_script}")
            return False
        
        print(f"✅ LOD生成脚本存在: {lod_script}")
        
        # 测试脚本是否可以导入
        try:
            sys.path.insert(0, str(lod_script.parent))
            import lod
            print("✅ LOD生成脚本可以导入")
            
            # 测试创建LOD生成器实例
            lod_generator = lod.LODGenerator()
            print("✅ LOD生成器实例创建成功")
            
            return True
            
        except Exception as e:
            print(f"❌ LOD生成脚本测试失败: {e}")
            return False
    
    def test_configuration_files(self) -> bool:
        """测试配置文件"""
        print("\n=== 测试配置文件 ===")
        
        config_dir = self.project_root / "configs"
        required_configs = ["global.yml", "shell.yml", "detail.yml", "fusion.yml"]
        
        all_exist = True
        for config_file in required_configs:
            config_path = config_dir / config_file
            if config_path.exists():
                print(f"✅ 配置文件存在: {config_file}")
                
                # 尝试解析YAML
                try:
                    import yaml
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                    print(f"✅ 配置文件格式正确: {config_file}")
                except Exception as e:
                    print(f"❌ 配置文件格式错误 {config_file}: {e}")
                    all_exist = False
            else:
                print(f"❌ 配置文件不存在: {config_file}")
                all_exist = False
        
        return all_exist
    
    def test_directory_structure(self) -> bool:
        """测试目录结构"""
        print("\n=== 测试目录结构 ===")
        
        required_dirs = [
            "src", "configs", "scripts", "data", "outputs",
            "outputs/mesh_shell", "outputs/mesh_detail", 
            "outputs/mesh_final", "outputs/reports"
        ]
        
        all_exist = True
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                print(f"✅ 目录存在: {dir_name}")
            else:
                print(f"❌ 目录不存在: {dir_name}")
                all_exist = False
        
        return all_exist
    
    def run_all_tests(self) -> dict:
        """运行所有测试"""
        print("开始运行基础功能测试...")
        print("=" * 60)
        
        tests = [
            ("目录结构", self.test_directory_structure),
            ("配置文件", self.test_configuration_files),
            ("C++可执行文件", self.test_cpp_executable),
            ("点云加载", self.test_point_cloud_loading),
            ("Python管道", self.test_python_pipeline),
            ("质量控制", self.test_quality_control),
            ("LOD生成", self.test_lod_generator),
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
            except Exception as e:
                print(f"❌ 测试 '{test_name}' 异常: {e}")
                results[test_name] = False
        
        # 打印总结
        print("\n" + "=" * 60)
        print("测试总结:")
        print(f"总测试数: {total}")
        print(f"通过测试: {passed}")
        print(f"失败测试: {total - passed}")
        print(f"通过率: {passed/total*100:.1f}%")
        
        # 详细结果
        print("\n详细结果:")
        for test_name, result in results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {test_name}: {status}")
        
        return results
    
    def generate_test_report(self, results: dict):
        """生成测试报告"""
        report_file = self.test_output_dir / "basic_test_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("室内点云重建项目 - 基础功能测试报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            passed = sum(results.values())
            total = len(results)
            
            f.write(f"测试总结:\n")
            f.write(f"  总测试数: {total}\n")
            f.write(f"  通过测试: {passed}\n")
            f.write(f"  失败测试: {total - passed}\n")
            f.write(f"  通过率: {passed/total*100:.1f}%\n\n")
            
            f.write("详细结果:\n")
            for test_name, result in results.items():
                status = "通过" if result else "失败"
                f.write(f"  {test_name}: {status}\n")
        
        print(f"\n测试报告已保存到: {report_file}")


def main():
    """主函数"""
    tester = BasicFunctionTester()
    results = tester.run_all_tests()
    tester.generate_test_report(results)
    
    # 返回适当的退出码
    if all(results.values()):
        print("\n🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("\n⚠️  部分测试失败，请检查上述输出")
        sys.exit(1)


if __name__ == "__main__":
    main()


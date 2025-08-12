#!/usr/bin/env python3
"""
室内点云重建项目 - 主管道脚本
基于双层混合管道的点云到网格转换

版本: 1.0
日期: 2025-08-12
"""

import os
import sys
import argparse
import subprocess
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
try:
    import pymeshlab
    PYMESHLAB_AVAILABLE = True
except ImportError:
    PYMESHLAB_AVAILABLE = False
    print("警告: PyMeshLab未安装，某些功能可能不可用")

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

class ReconstructionPipeline:
    """室内点云重建管道类"""
    
    def __init__(self, config_dir: str = None):
        """
        初始化重建管道
        
        Args:
            config_dir: 配置文件目录路径
        """
        self.config_dir = Path(config_dir) if config_dir else PROJECT_ROOT / "configs"
        self.build_dir = PROJECT_ROOT / "build"
        self.output_dir = PROJECT_ROOT / "outputs"
        
        # 加载配置
        self.global_config = self._load_config("global.yml")
        self.shell_config = self._load_config("shell.yml")
        self.detail_config = self._load_config("detail.yml")
        self.fusion_config = self._load_config("fusion.yml")
        
        # 设置日志
        self._setup_logging()
        
        # 统计信息
        self.stats = {
            'start_time': None,
            'end_time': None,
            'stages': {},
            'input_points': 0,
            'output_vertices': 0,
            'output_triangles': 0
        }
    
    def _load_config(self, filename: str) -> Dict:
        """加载YAML配置文件"""
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"配置文件未找到: {config_path}")
            return {}
        except yaml.YAMLError as e:
            self.logger.error(f"配置文件解析错误: {e}")
            return {}
    
    def _setup_logging(self):
        """设置日志系统"""
        log_level = self.global_config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.output_dir / "reports" / "pipeline.log")
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_full_pipeline(self, input_file: str, output_file: str) -> bool:
        """
        运行完整的重建管道
        
        Args:
            input_file: 输入点云文件路径
            output_file: 输出网格文件路径
            
        Returns:
            成功返回True
        """
        self.stats['start_time'] = time.time()
        self.logger.info(f"开始点云重建管道: {input_file} -> {output_file}")
        
        try:
            # 1. 验证输入
            if not self._validate_input(input_file):
                return False
            
            # 2. 预处理和外壳重建
            shell_mesh_path = self.output_dir / "mesh_shell" / "shell.obj"
            if not self._run_shell_reconstruction(input_file, str(shell_mesh_path)):
                return False
            
            # 3. 细节重建
            detail_mesh_path = self.output_dir / "mesh_detail" / "detail.obj"
            if not self._run_detail_reconstruction(input_file, str(shell_mesh_path), str(detail_mesh_path)):
                return False
            
            # 4. 融合
            fused_mesh_path = self.output_dir / "mesh_final" / "fused.obj"
            if not self._run_fusion(str(shell_mesh_path), str(detail_mesh_path), str(fused_mesh_path)):
                return False
            
            # 5. LOD生成
            if not self._run_lod_generation(str(fused_mesh_path)):
                return False
            
            # 6. 复制最终结果
            if not self._copy_final_result(str(fused_mesh_path), output_file):
                return False
            
            # 7. 生成报告
            self._generate_report(output_file)
            
            self.stats['end_time'] = time.time()
            total_time = self.stats['end_time'] - self.stats['start_time']
            self.logger.info(f"重建完成，总耗时: {total_time:.2f}秒")
            
            return True
            
        except Exception as e:
            self.logger.error(f"管道执行失败: {e}")
            return False
    
    def _validate_input(self, input_file: str) -> bool:
        """验证输入文件"""
        if not os.path.exists(input_file):
            self.logger.error(f"输入文件不存在: {input_file}")
            return False
        
        # 尝试加载点云
        try:
            cloud = o3d.io.read_point_cloud(input_file)
            if len(cloud.points) == 0:
                self.logger.error("输入点云为空")
                return False
            
            self.stats['input_points'] = len(cloud.points)
            self.logger.info(f"输入点云包含 {len(cloud.points)} 个点")
            return True
            
        except Exception as e:
            self.logger.error(f"无法加载输入点云: {e}")
            return False
    
    def _run_shell_reconstruction(self, input_file: str, output_file: str) -> bool:
        """运行外壳重建"""
        stage_start = time.time()
        self.logger.info("开始外壳重建...")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 构建命令
        cpp_executable = self.build_dir / "pipeline"
        if not cpp_executable.exists():
            self.logger.error(f"C++可执行文件不存在: {cpp_executable}")
            return False
        
        cmd = [str(cpp_executable), input_file, output_file]
        
        try:
            # 运行C++程序
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                self.logger.error(f"外壳重建失败: {result.stderr}")
                return False
            
            # 验证输出
            if not os.path.exists(output_file):
                self.logger.error(f"外壳重建输出文件未生成: {output_file}")
                return False
            
            stage_time = time.time() - stage_start
            self.stats['stages']['shell_reconstruction'] = stage_time
            self.logger.info(f"外壳重建完成，耗时: {stage_time:.2f}秒")
            
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("外壳重建超时")
            return False
        except Exception as e:
            self.logger.error(f"外壳重建异常: {e}")
            return False
    
    def _run_detail_reconstruction(self, input_file: str, shell_file: str, output_file: str) -> bool:
        """运行细节重建"""
        stage_start = time.time()
        self.logger.info("开始细节重建...")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if not PYMESHLAB_AVAILABLE:
            self.logger.warning("PyMeshLab不可用，跳过细节重建")
            # 复制外壳网格作为细节网格
            import shutil
            shutil.copy2(shell_file, output_file)
            return True
        
        try:
            # 使用PyMeshLab进行细节重建
            ms = pymeshlab.MeshSet()
            
            # 加载原始点云
            ms.load_new_mesh(input_file)
            
            # 应用RIMLS重建
            ms.apply_filter('compute_curvature_and_color_rimls_per_vertex', 
                           h=self.detail_config.get('rimls', {}).get('voxel_size', 0.01))
            
            ms.apply_filter('generate_marching_cubes_rimls', 
                           decimation=0, 
                           ml_smoothing_step=self.detail_config.get('rimls', {}).get('smoothing_steps', 5))
            
            # 保存结果
            ms.save_current_mesh(output_file)
            
            stage_time = time.time() - stage_start
            self.stats['stages']['detail_reconstruction'] = stage_time
            self.logger.info(f"细节重建完成，耗时: {stage_time:.2f}秒")
            
            return True
            
        except Exception as e:
            self.logger.error(f"细节重建失败: {e}")
            # 后备方案：复制外壳网格
            import shutil
            shutil.copy2(shell_file, output_file)
            return True
    
    def _run_fusion(self, shell_file: str, detail_file: str, output_file: str) -> bool:
        """运行网格融合"""
        stage_start = time.time()
        self.logger.info("开始网格融合...")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            # 使用Open3D进行简单融合
            shell_mesh = o3d.io.read_triangle_mesh(shell_file)
            detail_mesh = o3d.io.read_triangle_mesh(detail_file)
            
            if len(shell_mesh.vertices) == 0:
                self.logger.error("外壳网格为空")
                return False
            
            # 简单融合：优先使用外壳网格
            if len(detail_mesh.vertices) > 0:
                # 合并顶点和三角形
                combined_vertices = np.vstack([
                    np.asarray(shell_mesh.vertices),
                    np.asarray(detail_mesh.vertices)
                ])
                
                shell_triangles = np.asarray(shell_mesh.triangles)
                detail_triangles = np.asarray(detail_mesh.triangles) + len(shell_mesh.vertices)
                combined_triangles = np.vstack([shell_triangles, detail_triangles])
                
                # 创建融合网格
                fused_mesh = o3d.geometry.TriangleMesh()
                fused_mesh.vertices = o3d.utility.Vector3dVector(combined_vertices)
                fused_mesh.triangles = o3d.utility.Vector3iVector(combined_triangles)
                
                # 移除重复顶点
                fused_mesh.remove_duplicated_vertices()
                fused_mesh.remove_duplicated_triangles()
                fused_mesh.remove_degenerate_triangles()
                fused_mesh.remove_unreferenced_vertices()
            else:
                fused_mesh = shell_mesh
            
            # 保存结果
            o3d.io.write_triangle_mesh(output_file, fused_mesh)
            
            self.stats['output_vertices'] = len(fused_mesh.vertices)
            self.stats['output_triangles'] = len(fused_mesh.triangles)
            
            stage_time = time.time() - stage_start
            self.stats['stages']['fusion'] = stage_time
            self.logger.info(f"网格融合完成，耗时: {stage_time:.2f}秒")
            self.logger.info(f"融合网格: {len(fused_mesh.vertices)} 顶点, {len(fused_mesh.triangles)} 三角形")
            
            return True
            
        except Exception as e:
            self.logger.error(f"网格融合失败: {e}")
            return False
    
    def _run_lod_generation(self, input_file: str) -> bool:
        """运行LOD生成"""
        stage_start = time.time()
        self.logger.info("开始LOD生成...")
        
        try:
            # 加载网格
            mesh = o3d.io.read_triangle_mesh(input_file)
            if len(mesh.vertices) == 0:
                self.logger.error("输入网格为空")
                return False
            
            # 生成不同级别的LOD
            lod_configs = self.fusion_config.get('lod_generation', {}).get('target_counts', {})
            
            for lod_name, target_count in lod_configs.items():
                if lod_name == 'full':
                    continue
                
                # 简化网格
                simplified = mesh.simplify_quadric_decimation(target_count)
                
                # 保存LOD
                lod_file = self.output_dir / "mesh_final" / f"mesh_{lod_name}.obj"
                o3d.io.write_triangle_mesh(str(lod_file), simplified)
                
                self.logger.info(f"生成 {lod_name}: {len(simplified.vertices)} 顶点, {len(simplified.triangles)} 三角形")
            
            stage_time = time.time() - stage_start
            self.stats['stages']['lod_generation'] = stage_time
            self.logger.info(f"LOD生成完成，耗时: {stage_time:.2f}秒")
            
            return True
            
        except Exception as e:
            self.logger.error(f"LOD生成失败: {e}")
            return False
    
    def _copy_final_result(self, source_file: str, target_file: str) -> bool:
        """复制最终结果到指定位置"""
        try:
            import shutil
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            shutil.copy2(source_file, target_file)
            self.logger.info(f"最终结果已保存到: {target_file}")
            return True
        except Exception as e:
            self.logger.error(f"复制最终结果失败: {e}")
            return False
    
    def _generate_report(self, output_file: str):
        """生成重建报告"""
        try:
            report_file = self.output_dir / "reports" / "reconstruction_report.txt"
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            total_time = self.stats['end_time'] - self.stats['start_time']
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("室内点云重建报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"输入点数: {self.stats['input_points']}\n")
                f.write(f"输出顶点数: {self.stats['output_vertices']}\n")
                f.write(f"输出三角形数: {self.stats['output_triangles']}\n")
                f.write(f"总耗时: {total_time:.2f}秒\n\n")
                
                f.write("各阶段耗时:\n")
                for stage, duration in self.stats['stages'].items():
                    f.write(f"  {stage}: {duration:.2f}秒\n")
                
                f.write(f"\n最终输出文件: {output_file}\n")
            
            self.logger.info(f"重建报告已保存到: {report_file}")
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="室内点云重建管道")
    parser.add_argument("input", help="输入点云文件(.ply)")
    parser.add_argument("output", help="输出网格文件(.obj)")
    parser.add_argument("--config-dir", help="配置文件目录")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 创建管道实例
    pipeline = ReconstructionPipeline(args.config_dir)
    
    # 运行重建
    success = pipeline.run_full_pipeline(args.input, args.output)
    
    if success:
        print("重建成功完成！")
        sys.exit(0)
    else:
        print("重建失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()


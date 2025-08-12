#!/usr/bin/env python3
"""
室内点云重建项目 - LOD生成脚本
专门用于生成多级细节(Level of Detail)网格

版本: 1.0
日期: 2025-08-12
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import open3d as o3d

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

class LODGenerator:
    """LOD生成器"""
    
    def __init__(self, output_dir: str = None):
        """
        初始化LOD生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs" / "mesh_final"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 默认LOD配置
        self.lod_config = {
            'lod1': {'target_triangles': 200000, 'reduction_ratio': 0.5},
            'lod2': {'target_triangles': 50000, 'reduction_ratio': 0.25},
            'lod3': {'target_triangles': 10000, 'reduction_ratio': 0.1}
        }
        
        # 简化参数
        self.simplification_params = {
            'preserve_boundary': True,
            'boundary_weight': 1000.0,
            'preserve_feature': True,
            'feature_angle': 30.0  # 度
        }
    
    def generate_lod_hierarchy(self, input_mesh_file: str, base_name: str = None) -> Dict:
        """
        生成完整的LOD层次结构
        
        Args:
            input_mesh_file: 输入网格文件
            base_name: 基础文件名
            
        Returns:
            LOD生成结果字典
        """
        print(f"开始生成LOD层次结构: {input_mesh_file}")
        
        # 加载原始网格
        original_mesh = o3d.io.read_triangle_mesh(input_mesh_file)
        if len(original_mesh.vertices) == 0:
            return {'error': '无法加载输入网格或网格为空'}
        
        if base_name is None:
            base_name = Path(input_mesh_file).stem
        
        # 预处理原始网格
        original_mesh = self._preprocess_mesh(original_mesh)
        
        # 保存原始网格（作为LOD0）
        lod0_file = self.output_dir / f"{base_name}_lod0.obj"
        o3d.io.write_triangle_mesh(str(lod0_file), original_mesh)
        
        # 生成结果
        results = {
            'input_file': input_mesh_file,
            'base_name': base_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'original_stats': self._get_mesh_stats(original_mesh),
            'lod_files': {'lod0': str(lod0_file)},
            'lod_stats': {'lod0': self._get_mesh_stats(original_mesh)},
            'generation_times': {},
            'quality_metrics': {}
        }
        
        # 生成各级LOD
        current_mesh = original_mesh
        for lod_name, config in self.lod_config.items():
            start_time = time.time()
            
            print(f"生成 {lod_name}...")
            
            # 简化网格
            simplified_mesh = self._simplify_mesh(current_mesh, config)
            
            if simplified_mesh is None:
                print(f"警告: {lod_name} 生成失败")
                continue
            
            # 后处理
            simplified_mesh = self._postprocess_mesh(simplified_mesh)
            
            # 保存LOD
            lod_file = self.output_dir / f"{base_name}_{lod_name}.obj"
            o3d.io.write_triangle_mesh(str(lod_file), simplified_mesh)
            
            # 记录统计信息
            generation_time = time.time() - start_time
            mesh_stats = self._get_mesh_stats(simplified_mesh)
            quality_metrics = self._compute_quality_metrics(original_mesh, simplified_mesh)
            
            results['lod_files'][lod_name] = str(lod_file)
            results['lod_stats'][lod_name] = mesh_stats
            results['generation_times'][lod_name] = generation_time
            results['quality_metrics'][lod_name] = quality_metrics
            
            print(f"{lod_name} 完成: {mesh_stats['vertices']} 顶点, "
                  f"{mesh_stats['triangles']} 三角形, "
                  f"耗时 {generation_time:.2f}秒")
            
            # 为下一级LOD准备
            current_mesh = simplified_mesh
        
        # 保存生成报告
        self._save_generation_report(results)
        
        print("LOD层次结构生成完成!")
        return results
    
    def _preprocess_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """预处理网格"""
        # 移除重复顶点和三角形
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        
        # 计算法向量
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # 统一方向
        try:
            mesh.orient_triangles()
        except:
            pass  # 如果失败就跳过
        
        return mesh
    
    def _simplify_mesh(self, mesh: o3d.geometry.TriangleMesh, config: Dict) -> Optional[o3d.geometry.TriangleMesh]:
        """简化网格"""
        target_triangles = config['target_triangles']
        current_triangles = len(mesh.triangles)
        
        if current_triangles <= target_triangles:
            print(f"当前三角形数 ({current_triangles}) 已小于目标数 ({target_triangles})，跳过简化")
            return mesh.copy()
        
        try:
            # 使用二次边坍塌简化
            simplified = mesh.simplify_quadric_decimation(target_triangles)
            
            # 如果简化失败或结果不理想，尝试其他方法
            if len(simplified.triangles) == 0 or len(simplified.triangles) > target_triangles * 1.5:
                print("二次边坍塌简化效果不佳，尝试顶点聚类简化")
                
                # 计算体素大小
                bbox = mesh.get_axis_aligned_bounding_box()
                bbox_size = bbox.get_extent()
                diagonal = np.linalg.norm(bbox_size)
                
                # 根据目标三角形数估算体素大小
                reduction_ratio = target_triangles / current_triangles
                voxel_size = diagonal * (1 - reduction_ratio) * 0.01
                
                simplified = mesh.simplify_vertex_clustering(voxel_size)
            
            return simplified
            
        except Exception as e:
            print(f"网格简化失败: {e}")
            return None
    
    def _postprocess_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """后处理网格"""
        # 清理网格
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        
        # 重新计算法向量
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        # 可选的平滑操作（轻微）
        try:
            mesh = mesh.filter_smooth_simple(number_of_iterations=1)
        except:
            pass
        
        return mesh
    
    def _get_mesh_stats(self, mesh: o3d.geometry.TriangleMesh) -> Dict:
        """获取网格统计信息"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # 基本统计
        stats = {
            'vertices': len(vertices),
            'triangles': len(triangles),
            'edges': len(triangles) * 3 // 2  # 近似值
        }
        
        # 几何统计
        try:
            bbox = mesh.get_axis_aligned_bounding_box()
            stats['bounding_box'] = bbox.get_extent().tolist()
            stats['surface_area'] = mesh.get_surface_area()
            
            try:
                stats['volume'] = mesh.get_volume()
            except:
                stats['volume'] = 0.0
                
        except Exception as e:
            print(f"计算几何统计失败: {e}")
        
        return stats
    
    def _compute_quality_metrics(self, original: o3d.geometry.TriangleMesh, 
                                simplified: o3d.geometry.TriangleMesh) -> Dict:
        """计算质量指标"""
        metrics = {}
        
        try:
            # 简化比例
            orig_triangles = len(original.triangles)
            simp_triangles = len(simplified.triangles)
            
            if orig_triangles > 0:
                metrics['reduction_ratio'] = simp_triangles / orig_triangles
                metrics['compression_ratio'] = 1.0 - metrics['reduction_ratio']
            
            # 几何误差（简化版本）
            orig_area = original.get_surface_area()
            simp_area = simplified.get_surface_area()
            
            if orig_area > 0:
                metrics['area_preservation'] = simp_area / orig_area
            
            # 体积保持（如果可计算）
            try:
                orig_volume = original.get_volume()
                simp_volume = simplified.get_volume()
                
                if abs(orig_volume) > 1e-10:
                    metrics['volume_preservation'] = simp_volume / orig_volume
            except:
                pass
            
            # 拓扑保持
            metrics['topology_preserved'] = (
                original.is_watertight() == simplified.is_watertight() and
                original.is_orientable() == simplified.is_orientable()
            )
            
        except Exception as e:
            print(f"计算质量指标失败: {e}")
        
        return metrics
    
    def _save_generation_report(self, results: Dict):
        """保存生成报告"""
        try:
            # JSON报告
            json_file = self.output_dir / f"{results['base_name']}_lod_report.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 文本报告
            text_file = self.output_dir / f"{results['base_name']}_lod_report.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(self._generate_text_report(results))
            
            print(f"生成报告已保存: {json_file}")
            
        except Exception as e:
            print(f"保存报告失败: {e}")
    
    def _generate_text_report(self, results: Dict) -> str:
        """生成文本报告"""
        report = []
        report.append("LOD生成报告")
        report.append("=" * 50)
        report.append(f"输入文件: {results['input_file']}")
        report.append(f"基础名称: {results['base_name']}")
        report.append(f"生成时间: {results['timestamp']}")
        report.append("")
        
        # 原始网格统计
        orig_stats = results['original_stats']
        report.append("原始网格统计:")
        report.append(f"  顶点数: {orig_stats['vertices']:,}")
        report.append(f"  三角形数: {orig_stats['triangles']:,}")
        if 'surface_area' in orig_stats:
            report.append(f"  表面积: {orig_stats['surface_area']:.3f}")
        report.append("")
        
        # 各级LOD统计
        report.append("LOD层次统计:")
        for lod_name in ['lod0', 'lod1', 'lod2', 'lod3']:
            if lod_name in results['lod_stats']:
                stats = results['lod_stats'][lod_name]
                report.append(f"  {lod_name.upper()}:")
                report.append(f"    顶点数: {stats['vertices']:,}")
                report.append(f"    三角形数: {stats['triangles']:,}")
                
                if lod_name in results['generation_times']:
                    time_taken = results['generation_times'][lod_name]
                    report.append(f"    生成时间: {time_taken:.2f}秒")
                
                if lod_name in results['quality_metrics']:
                    metrics = results['quality_metrics'][lod_name]
                    if 'reduction_ratio' in metrics:
                        report.append(f"    简化比例: {metrics['reduction_ratio']:.3f}")
                    if 'area_preservation' in metrics:
                        report.append(f"    面积保持: {metrics['area_preservation']:.3f}")
                
                report.append("")
        
        # 文件列表
        report.append("生成的文件:")
        for lod_name, file_path in results['lod_files'].items():
            report.append(f"  {lod_name.upper()}: {file_path}")
        
        return "\n".join(report)
    
    def batch_generate_lods(self, input_dir: str, pattern: str = "*.obj") -> List[Dict]:
        """批量生成LOD"""
        input_path = Path(input_dir)
        mesh_files = list(input_path.glob(pattern))
        
        if not mesh_files:
            print(f"在 {input_dir} 中未找到匹配 {pattern} 的文件")
            return []
        
        results = []
        for mesh_file in mesh_files:
            print(f"\n处理文件: {mesh_file}")
            try:
                result = self.generate_lod_hierarchy(str(mesh_file))
                if 'error' not in result:
                    results.append(result)
                else:
                    print(f"处理失败: {result['error']}")
            except Exception as e:
                print(f"处理异常: {e}")
        
        return results
    
    def compare_lod_quality(self, lod_files: Dict) -> Dict:
        """比较LOD质量"""
        comparison = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'comparisons': {}
        }
        
        # 加载所有LOD
        lod_meshes = {}
        for lod_name, file_path in lod_files.items():
            try:
                mesh = o3d.io.read_triangle_mesh(file_path)
                if len(mesh.vertices) > 0:
                    lod_meshes[lod_name] = mesh
            except Exception as e:
                print(f"加载 {lod_name} 失败: {e}")
        
        # 两两比较
        lod_names = list(lod_meshes.keys())
        for i in range(len(lod_names)):
            for j in range(i + 1, len(lod_names)):
                name1, name2 = lod_names[i], lod_names[j]
                mesh1, mesh2 = lod_meshes[name1], lod_meshes[name2]
                
                # 计算比较指标
                comp_key = f"{name1}_vs_{name2}"
                comparison['comparisons'][comp_key] = self._compute_quality_metrics(mesh1, mesh2)
        
        return comparison


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LOD生成器")
    parser.add_argument("input", help="输入网格文件或目录")
    parser.add_argument("--output-dir", help="输出目录")
    parser.add_argument("--base-name", help="基础文件名")
    parser.add_argument("--batch", action="store_true", help="批量处理模式")
    parser.add_argument("--pattern", default="*.obj", help="批量处理文件模式")
    parser.add_argument("--config", help="LOD配置文件")
    
    args = parser.parse_args()
    
    # 创建LOD生成器
    generator = LODGenerator(args.output_dir)
    
    # 加载自定义配置
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                import yaml
                custom_config = yaml.safe_load(f)
                generator.lod_config.update(custom_config.get('lod_config', {}))
                generator.simplification_params.update(custom_config.get('simplification_params', {}))
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    # 执行生成
    if args.batch:
        results = generator.batch_generate_lods(args.input, args.pattern)
        print(f"\n批量处理完成，成功处理 {len(results)} 个文件")
    else:
        if not os.path.exists(args.input):
            print(f"输入文件不存在: {args.input}")
            sys.exit(1)
        
        result = generator.generate_lod_hierarchy(args.input, args.base_name)
        
        if 'error' in result:
            print(f"生成失败: {result['error']}")
            sys.exit(1)
        else:
            print("LOD生成成功完成!")


if __name__ == "__main__":
    main()


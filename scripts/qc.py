#!/usr/bin/env python3
"""
室内点云重建项目 - 质量控制脚本
用于验证和评估重建结果的质量

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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

class MeshQualityController:
    """网格质量控制器"""
    
    def __init__(self, output_dir: str = None):
        """
        初始化质量控制器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "outputs" / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 质量阈值
        self.quality_thresholds = {
            'min_triangle_area': 1e-6,
            'max_edge_length': 0.2,
            'min_angle': 5.0,  # 度
            'max_angle': 175.0,  # 度
            'aspect_ratio_threshold': 10.0,
            'manifold_required': True,
            'watertight_required': False  # 室内重建可能不是完全水密的
        }
    
    def evaluate_mesh_quality(self, mesh_file: str) -> Dict:
        """
        评估网格质量
        
        Args:
            mesh_file: 网格文件路径
            
        Returns:
            质量评估结果字典
        """
        print(f"评估网格质量: {mesh_file}")
        
        # 加载网格
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        if len(mesh.vertices) == 0:
            return {'error': '无法加载网格或网格为空'}
        
        # 基本统计
        basic_stats = self._compute_basic_statistics(mesh)
        
        # 几何质量
        geometric_quality = self._evaluate_geometric_quality(mesh)
        
        # 拓扑质量
        topological_quality = self._evaluate_topological_quality(mesh)
        
        # 视觉质量
        visual_quality = self._evaluate_visual_quality(mesh)
        
        # 综合评分
        overall_score = self._compute_overall_score(geometric_quality, topological_quality, visual_quality)
        
        # 汇总结果
        quality_report = {
            'mesh_file': mesh_file,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'basic_statistics': basic_stats,
            'geometric_quality': geometric_quality,
            'topological_quality': topological_quality,
            'visual_quality': visual_quality,
            'overall_score': overall_score,
            'quality_grade': self._get_quality_grade(overall_score),
            'recommendations': self._generate_recommendations(geometric_quality, topological_quality, visual_quality)
        }
        
        return quality_report
    
    def _compute_basic_statistics(self, mesh: o3d.geometry.TriangleMesh) -> Dict:
        """计算基本统计信息"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # 边界框
        bbox = mesh.get_axis_aligned_bounding_box()
        bbox_size = bbox.get_extent()
        
        # 表面积和体积
        surface_area = mesh.get_surface_area()
        try:
            volume = mesh.get_volume()
        except:
            volume = 0.0
        
        return {
            'vertex_count': len(vertices),
            'triangle_count': len(triangles),
            'edge_count': len(triangles) * 3 // 2,  # 近似值
            'bounding_box_size': bbox_size.tolist(),
            'surface_area': surface_area,
            'volume': volume,
            'vertex_density': len(vertices) / surface_area if surface_area > 0 else 0
        }
    
    def _evaluate_geometric_quality(self, mesh: o3d.geometry.TriangleMesh) -> Dict:
        """评估几何质量"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # 计算三角形面积
        triangle_areas = self._compute_triangle_areas(vertices, triangles)
        
        # 计算边长
        edge_lengths = self._compute_edge_lengths(vertices, triangles)
        
        # 计算角度
        angles = self._compute_triangle_angles(vertices, triangles)
        
        # 计算长宽比
        aspect_ratios = self._compute_aspect_ratios(vertices, triangles)
        
        # 统计信息
        geometric_stats = {
            'triangle_areas': {
                'min': float(np.min(triangle_areas)),
                'max': float(np.max(triangle_areas)),
                'mean': float(np.mean(triangle_areas)),
                'std': float(np.std(triangle_areas)),
                'median': float(np.median(triangle_areas))
            },
            'edge_lengths': {
                'min': float(np.min(edge_lengths)),
                'max': float(np.max(edge_lengths)),
                'mean': float(np.mean(edge_lengths)),
                'std': float(np.std(edge_lengths)),
                'median': float(np.median(edge_lengths))
            },
            'angles': {
                'min': float(np.min(angles)),
                'max': float(np.max(angles)),
                'mean': float(np.mean(angles)),
                'std': float(np.std(angles))
            },
            'aspect_ratios': {
                'min': float(np.min(aspect_ratios)),
                'max': float(np.max(aspect_ratios)),
                'mean': float(np.mean(aspect_ratios)),
                'std': float(np.std(aspect_ratios))
            }
        }
        
        # 质量检查
        quality_checks = {
            'degenerate_triangles': int(np.sum(triangle_areas < self.quality_thresholds['min_triangle_area'])),
            'oversized_edges': int(np.sum(edge_lengths > self.quality_thresholds['max_edge_length'])),
            'acute_angles': int(np.sum(angles < self.quality_thresholds['min_angle'])),
            'obtuse_angles': int(np.sum(angles > self.quality_thresholds['max_angle'])),
            'high_aspect_ratio': int(np.sum(aspect_ratios > self.quality_thresholds['aspect_ratio_threshold']))
        }
        
        # 几何质量评分 (0-100)
        total_triangles = len(triangles)
        geometric_score = 100.0
        
        if total_triangles > 0:
            geometric_score -= (quality_checks['degenerate_triangles'] / total_triangles) * 30
            geometric_score -= (quality_checks['oversized_edges'] / (total_triangles * 3)) * 20
            geometric_score -= (quality_checks['acute_angles'] / (total_triangles * 3)) * 15
            geometric_score -= (quality_checks['obtuse_angles'] / (total_triangles * 3)) * 15
            geometric_score -= (quality_checks['high_aspect_ratio'] / total_triangles) * 20
        
        geometric_score = max(0.0, geometric_score)
        
        return {
            'statistics': geometric_stats,
            'quality_checks': quality_checks,
            'score': geometric_score
        }
    
    def _evaluate_topological_quality(self, mesh: o3d.geometry.TriangleMesh) -> Dict:
        """评估拓扑质量"""
        # 检查流形性
        is_edge_manifold = mesh.is_edge_manifold()
        is_vertex_manifold = mesh.is_vertex_manifold()
        is_manifold = is_edge_manifold and is_vertex_manifold
        
        # 检查方向一致性
        is_orientable = mesh.is_orientable()
        
        # 检查水密性
        is_watertight = mesh.is_watertight()
        
        # 检查自相交（简化检查）
        has_self_intersections = self._check_self_intersections(mesh)
        
        # 边界信息
        boundary_info = self._analyze_boundaries(mesh)
        
        # 拓扑质量评分
        topological_score = 100.0
        
        if not is_manifold:
            topological_score -= 40
        if not is_orientable:
            topological_score -= 20
        if self.quality_thresholds['watertight_required'] and not is_watertight:
            topological_score -= 20
        if has_self_intersections:
            topological_score -= 30
        
        topological_score = max(0.0, topological_score)
        
        return {
            'is_manifold': is_manifold,
            'is_edge_manifold': is_edge_manifold,
            'is_vertex_manifold': is_vertex_manifold,
            'is_orientable': is_orientable,
            'is_watertight': is_watertight,
            'has_self_intersections': has_self_intersections,
            'boundary_info': boundary_info,
            'score': topological_score
        }
    
    def _evaluate_visual_quality(self, mesh: o3d.geometry.TriangleMesh) -> Dict:
        """评估视觉质量"""
        # 计算法向量
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        
        # 法向量一致性
        normal_consistency = self._compute_normal_consistency(mesh)
        
        # 表面平滑度
        surface_smoothness = self._compute_surface_smoothness(mesh)
        
        # 特征保持度（基于曲率变化）
        feature_preservation = self._compute_feature_preservation(mesh)
        
        # 视觉质量评分
        visual_score = (normal_consistency + surface_smoothness + feature_preservation) / 3.0 * 100
        
        return {
            'normal_consistency': normal_consistency,
            'surface_smoothness': surface_smoothness,
            'feature_preservation': feature_preservation,
            'score': visual_score
        }
    
    def _compute_triangle_areas(self, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """计算三角形面积"""
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        
        # 使用叉积计算面积
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        return areas
    
    def _compute_edge_lengths(self, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """计算边长"""
        edges = []
        for tri in triangles:
            edges.extend([(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])])
        
        edge_lengths = []
        for edge in edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            length = np.linalg.norm(v2 - v1)
            edge_lengths.append(length)
        
        return np.array(edge_lengths)
    
    def _compute_triangle_angles(self, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """计算三角形内角"""
        angles = []
        
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            
            # 计算三个内角
            for i in range(3):
                if i == 0:
                    a, b, c = v0, v1, v2
                elif i == 1:
                    a, b, c = v1, v2, v0
                else:
                    a, b, c = v2, v0, v1
                
                # 向量
                ba = a - b
                bc = c - b
                
                # 角度
                cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180.0 / np.pi
                angles.append(angle)
        
        return np.array(angles)
    
    def _compute_aspect_ratios(self, vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
        """计算三角形长宽比"""
        aspect_ratios = []
        
        for tri in triangles:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            
            # 计算三边长
            edge_lengths = [
                np.linalg.norm(v1 - v0),
                np.linalg.norm(v2 - v1),
                np.linalg.norm(v0 - v2)
            ]
            
            # 长宽比 = 最长边 / 最短边
            max_edge = max(edge_lengths)
            min_edge = min(edge_lengths)
            
            if min_edge > 1e-10:
                aspect_ratio = max_edge / min_edge
            else:
                aspect_ratio = float('inf')
            
            aspect_ratios.append(aspect_ratio)
        
        return np.array(aspect_ratios)
    
    def _check_self_intersections(self, mesh: o3d.geometry.TriangleMesh) -> bool:
        """检查自相交（简化版本）"""
        # 这里使用简化的检查方法
        # 实际应用中可能需要更复杂的算法
        try:
            # 尝试使用Open3D的内置方法
            return not mesh.is_self_intersecting()
        except:
            # 如果方法不可用，返回False
            return False
    
    def _analyze_boundaries(self, mesh: o3d.geometry.TriangleMesh) -> Dict:
        """分析边界信息"""
        # 简化的边界分析
        triangles = np.asarray(mesh.triangles)
        
        # 统计边的出现次数
        edge_count = {}
        for tri in triangles:
            edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for edge in edges:
                # 标准化边（小索引在前）
                edge = tuple(sorted(edge))
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # 边界边（只出现一次的边）
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        return {
            'total_edges': len(edge_count),
            'boundary_edges': len(boundary_edges),
            'is_closed': len(boundary_edges) == 0
        }
    
    def _compute_normal_consistency(self, mesh: o3d.geometry.TriangleMesh) -> float:
        """计算法向量一致性"""
        # 简化实现：基于相邻三角形法向量的一致性
        try:
            mesh.compute_triangle_normals()
            normals = np.asarray(mesh.triangle_normals)
            
            # 计算法向量的方差作为一致性的度量
            normal_variance = np.var(normals, axis=0).mean()
            consistency = 1.0 / (1.0 + normal_variance)
            
            return consistency
        except:
            return 0.5  # 默认值
    
    def _compute_surface_smoothness(self, mesh: o3d.geometry.TriangleMesh) -> float:
        """计算表面平滑度"""
        # 简化实现：基于顶点法向量的变化
        try:
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)
            
            # 计算相邻顶点法向量的差异
            # 这里使用简化的方法
            normal_diff = np.std(normals, axis=0).mean()
            smoothness = 1.0 / (1.0 + normal_diff)
            
            return smoothness
        except:
            return 0.5  # 默认值
    
    def _compute_feature_preservation(self, mesh: o3d.geometry.TriangleMesh) -> float:
        """计算特征保持度"""
        # 简化实现：基于曲率变化
        try:
            # 这里使用简化的特征检测
            vertices = np.asarray(mesh.vertices)
            
            # 计算顶点的局部变化
            vertex_variance = np.var(vertices, axis=0).mean()
            feature_score = min(1.0, vertex_variance * 10)  # 归一化
            
            return feature_score
        except:
            return 0.5  # 默认值
    
    def _compute_overall_score(self, geometric: Dict, topological: Dict, visual: Dict) -> float:
        """计算综合评分"""
        # 加权平均
        weights = {'geometric': 0.4, 'topological': 0.4, 'visual': 0.2}
        
        overall_score = (
            weights['geometric'] * geometric['score'] +
            weights['topological'] * topological['score'] +
            weights['visual'] * visual['score']
        )
        
        return overall_score
    
    def _get_quality_grade(self, score: float) -> str:
        """根据评分获取质量等级"""
        if score >= 90:
            return 'A (优秀)'
        elif score >= 80:
            return 'B (良好)'
        elif score >= 70:
            return 'C (一般)'
        elif score >= 60:
            return 'D (较差)'
        else:
            return 'F (失败)'
    
    def _generate_recommendations(self, geometric: Dict, topological: Dict, visual: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 几何质量建议
        if geometric['score'] < 80:
            if geometric['quality_checks']['degenerate_triangles'] > 0:
                recommendations.append("移除退化三角形以改善几何质量")
            if geometric['quality_checks']['high_aspect_ratio'] > 0:
                recommendations.append("优化三角形形状，减少高长宽比三角形")
        
        # 拓扑质量建议
        if topological['score'] < 80:
            if not topological['is_manifold']:
                recommendations.append("修复非流形几何以改善拓扑质量")
            if topological['has_self_intersections']:
                recommendations.append("解决自相交问题")
        
        # 视觉质量建议
        if visual['score'] < 80:
            recommendations.append("应用平滑滤波以改善视觉质量")
            recommendations.append("优化法向量计算")
        
        if not recommendations:
            recommendations.append("网格质量良好，无需特殊改进")
        
        return recommendations
    
    def generate_quality_report(self, quality_data: Dict, output_file: str = None) -> str:
        """生成质量报告"""
        if output_file is None:
            output_file = self.output_dir / "quality_report.json"
        
        # 保存JSON报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(quality_data, f, indent=2, ensure_ascii=False)
        
        # 生成文本报告
        text_report = self._generate_text_report(quality_data)
        text_file = str(output_file).replace('.json', '.txt')
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        # 生成可视化报告
        self._generate_visual_report(quality_data, str(output_file).replace('.json', '.pdf'))
        
        return str(output_file)
    
    def _generate_text_report(self, quality_data: Dict) -> str:
        """生成文本格式的质量报告"""
        report = []
        report.append("网格质量评估报告")
        report.append("=" * 50)
        report.append(f"文件: {quality_data['mesh_file']}")
        report.append(f"时间: {quality_data['timestamp']}")
        report.append(f"综合评分: {quality_data['overall_score']:.1f}/100")
        report.append(f"质量等级: {quality_data['quality_grade']}")
        report.append("")
        
        # 基本统计
        stats = quality_data['basic_statistics']
        report.append("基本统计信息:")
        report.append(f"  顶点数: {stats['vertex_count']:,}")
        report.append(f"  三角形数: {stats['triangle_count']:,}")
        report.append(f"  表面积: {stats['surface_area']:.3f}")
        report.append(f"  体积: {stats['volume']:.3f}")
        report.append("")
        
        # 几何质量
        geo = quality_data['geometric_quality']
        report.append(f"几何质量评分: {geo['score']:.1f}/100")
        report.append(f"  退化三角形: {geo['quality_checks']['degenerate_triangles']}")
        report.append(f"  过大边缘: {geo['quality_checks']['oversized_edges']}")
        report.append(f"  高长宽比三角形: {geo['quality_checks']['high_aspect_ratio']}")
        report.append("")
        
        # 拓扑质量
        topo = quality_data['topological_quality']
        report.append(f"拓扑质量评分: {topo['score']:.1f}/100")
        report.append(f"  是否流形: {'是' if topo['is_manifold'] else '否'}")
        report.append(f"  是否可定向: {'是' if topo['is_orientable'] else '否'}")
        report.append(f"  是否水密: {'是' if topo['is_watertight'] else '否'}")
        report.append("")
        
        # 改进建议
        report.append("改进建议:")
        for i, rec in enumerate(quality_data['recommendations'], 1):
            report.append(f"  {i}. {rec}")
        
        return "\n".join(report)
    
    def _generate_visual_report(self, quality_data: Dict, output_file: str):
        """生成可视化报告"""
        try:
            with PdfPages(output_file) as pdf:
                # 综合评分图
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
                
                # 评分雷达图
                categories = ['几何质量', '拓扑质量', '视觉质量']
                scores = [
                    quality_data['geometric_quality']['score'],
                    quality_data['topological_quality']['score'],
                    quality_data['visual_quality']['score']
                ]
                
                ax1.bar(categories, scores, color=['blue', 'green', 'orange'])
                ax1.set_ylim(0, 100)
                ax1.set_title('质量评分')
                ax1.set_ylabel('评分')
                
                # 基本统计饼图
                stats = quality_data['basic_statistics']
                ax2.pie([stats['vertex_count'], stats['triangle_count']], 
                       labels=['顶点', '三角形'], autopct='%1.1f%%')
                ax2.set_title('网格组成')
                
                # 几何质量问题分布
                geo_checks = quality_data['geometric_quality']['quality_checks']
                problems = list(geo_checks.keys())
                counts = list(geo_checks.values())
                
                ax3.bar(problems, counts, color='red', alpha=0.7)
                ax3.set_title('几何质量问题')
                ax3.set_ylabel('数量')
                ax3.tick_params(axis='x', rotation=45)
                
                # 综合评分仪表盘
                overall_score = quality_data['overall_score']
                ax4.pie([overall_score, 100-overall_score], 
                       labels=['得分', ''], 
                       colors=['green', 'lightgray'],
                       startangle=90,
                       counterclock=False)
                ax4.set_title(f'综合评分: {overall_score:.1f}/100')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"生成可视化报告失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="网格质量控制")
    parser.add_argument("mesh_file", help="要评估的网格文件")
    parser.add_argument("--output-dir", help="输出目录")
    parser.add_argument("--report-format", choices=['json', 'txt', 'pdf', 'all'], 
                       default='all', help="报告格式")
    
    args = parser.parse_args()
    
    # 创建质量控制器
    qc = MeshQualityController(args.output_dir)
    
    # 评估质量
    quality_data = qc.evaluate_mesh_quality(args.mesh_file)
    
    if 'error' in quality_data:
        print(f"错误: {quality_data['error']}")
        sys.exit(1)
    
    # 生成报告
    report_file = qc.generate_quality_report(quality_data)
    
    print(f"质量评估完成!")
    print(f"综合评分: {quality_data['overall_score']:.1f}/100")
    print(f"质量等级: {quality_data['quality_grade']}")
    print(f"报告已保存到: {report_file}")


if __name__ == "__main__":
    main()


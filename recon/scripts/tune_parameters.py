#!/usr/bin/env python3
"""
室内点云重建项目 - 参数调优脚本
用于自动调优重建参数以获得最佳效果

版本: 1.0
日期: 2025-08-12
"""

import os
import sys
import yaml
import json
import time
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import open3d as o3d

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

class ParameterTuner:
    """参数调优器"""
    
    def __init__(self, config_dir: str = None):
        """
        初始化参数调优器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir) if config_dir else PROJECT_ROOT / "configs"
        self.output_dir = PROJECT_ROOT / "outputs" / "tuning"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载当前配置
        self.current_configs = self._load_all_configs()
        
        # 调优历史
        self.tuning_history = []
        
        # 参数搜索空间
        self.parameter_space = self._define_parameter_space()
    
    def _load_all_configs(self) -> Dict:
        """加载所有配置文件"""
        configs = {}
        config_files = ["global.yml", "shell.yml", "detail.yml", "fusion.yml"]
        
        for config_file in config_files:
            config_path = self.config_dir / config_file
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    configs[config_file.replace('.yml', '')] = yaml.safe_load(f)
            except FileNotFoundError:
                print(f"警告: 配置文件未找到: {config_path}")
                configs[config_file.replace('.yml', '')] = {}
        
        return configs
    
    def _define_parameter_space(self) -> Dict:
        """定义参数搜索空间"""
        return {
            # 外壳重建参数
            'shell': {
                'voxel_size': [0.005, 0.01, 0.02, 0.03],  # 体素大小
                'truncation_distance': [0.04, 0.08, 0.12, 0.16],  # 截断距离
                'use_gaussian_filter': [True, False],  # 是否使用高斯滤波
                'graph_cut': {
                    'inside_cost_multiplier': [0.5, 1.0, 1.5, 2.0],
                    'visibility_weight': [0.1, 0.3, 0.5, 0.7],
                    'base_weight': [0.5, 0.8, 1.0, 1.2]
                }
            },
            
            # 细节重建参数
            'detail': {
                'offset_distance': [0.04, 0.06, 0.08, 0.10],  # 偏移带距离
                'gp3': {
                    'edge_length_multiplier': [2.0, 2.5, 3.0, 3.5],
                    'mu': [2.0, 2.5, 2.8, 3.0],
                    'angle_threshold_planar': [30, 45, 60, 75],  # 度
                    'max_nearest_neighbors': [50, 75, 100, 125]
                },
                'rimls': {
                    'bandwidth': [1.0, 1.3, 1.6, 2.0],
                    'voxel_size': [0.005, 0.01, 0.015, 0.02],
                    'smoothing_steps': [3, 5, 7, 10]
                }
            },
            
            # 融合参数
            'fusion': {
                'detail_threshold': [0.003, 0.005, 0.008, 0.01],  # 细节阈值
                'welding_threshold': [0.001, 0.002, 0.003, 0.005],  # 焊接阈值
                'color': {
                    'shell_weight': [0.5, 0.6, 0.7, 0.8],
                    'detail_weight': [0.2, 0.3, 0.4, 0.5],
                    'transition_distance': [0.005, 0.01, 0.015, 0.02]
                }
            }
        }
    
    def generate_parameter_combinations(self, max_combinations: int = 50) -> List[Dict]:
        """生成参数组合"""
        print(f"生成参数组合（最多{max_combinations}个）...")
        
        # 使用网格搜索生成组合
        combinations = []
        
        # 为了避免组合爆炸，我们分别对每个模块进行采样
        shell_params = self._sample_module_params('shell', max_combinations // 3)
        detail_params = self._sample_module_params('detail', max_combinations // 3)
        fusion_params = self._sample_module_params('fusion', max_combinations // 3)
        
        # 组合不同模块的参数
        for i in range(min(max_combinations, len(shell_params), len(detail_params), len(fusion_params))):
            combination = {
                'shell': shell_params[i % len(shell_params)],
                'detail': detail_params[i % len(detail_params)],
                'fusion': fusion_params[i % len(fusion_params)]
            }
            combinations.append(combination)
        
        print(f"生成了 {len(combinations)} 个参数组合")
        return combinations
    
    def _sample_module_params(self, module: str, max_samples: int) -> List[Dict]:
        """为单个模块采样参数"""
        module_space = self.parameter_space[module]
        
        # 获取所有参数的键值对
        param_items = []
        for key, values in module_space.items():
            if isinstance(values, dict):
                # 嵌套参数
                for sub_key, sub_values in values.items():
                    param_items.append((f"{key}.{sub_key}", sub_values))
            else:
                param_items.append((key, values))
        
        # 随机采样组合
        np.random.seed(42)  # 确保可重现性
        combinations = []
        
        for _ in range(max_samples):
            combination = {}
            for param_name, param_values in param_items:
                value = np.random.choice(param_values)
                
                # 处理嵌套参数
                if '.' in param_name:
                    main_key, sub_key = param_name.split('.', 1)
                    if main_key not in combination:
                        combination[main_key] = {}
                    combination[main_key][sub_key] = value
                else:
                    combination[param_name] = value
            
            combinations.append(combination)
        
        return combinations
    
    def evaluate_parameters(self, params: Dict, test_data: str) -> Dict:
        """评估参数组合的效果"""
        print(f"评估参数组合...")
        
        # 创建临时配置文件
        temp_configs = self._create_temp_configs(params)
        
        # 模拟重建过程（实际应用中会调用真实的重建管道）
        metrics = self._simulate_reconstruction(params, test_data)
        
        # 清理临时文件
        self._cleanup_temp_configs(temp_configs)
        
        return metrics
    
    def _create_temp_configs(self, params: Dict) -> List[Path]:
        """创建临时配置文件"""
        temp_files = []
        
        for module, module_params in params.items():
            # 复制当前配置
            current_config = self.current_configs.get(module, {}).copy()
            
            # 更新参数
            self._update_nested_dict(current_config, module_params)
            
            # 保存临时配置
            temp_file = self.output_dir / f"temp_{module}.yml"
            with open(temp_file, 'w', encoding='utf-8') as f:
                yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)
            
            temp_files.append(temp_file)
        
        return temp_files
    
    def _update_nested_dict(self, target: Dict, source: Dict):
        """递归更新嵌套字典"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._update_nested_dict(target[key], value)
            else:
                target[key] = value
    
    def _cleanup_temp_configs(self, temp_files: List[Path]):
        """清理临时配置文件"""
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except FileNotFoundError:
                pass
    
    def _simulate_reconstruction(self, params: Dict, test_data: str) -> Dict:
        """模拟重建过程并计算质量指标"""
        # 这里是模拟的评估过程
        # 实际应用中应该调用真实的重建管道
        
        # 基于参数计算模拟的质量指标
        shell_params = params.get('shell', {})
        detail_params = params.get('detail', {})
        fusion_params = params.get('fusion', {})
        
        # 模拟质量指标计算
        voxel_size = shell_params.get('voxel_size', 0.01)
        offset_distance = detail_params.get('offset_distance', 0.08)
        detail_threshold = fusion_params.get('detail_threshold', 0.005)
        
        # 简化的质量评估模型
        # 实际应用中应该基于真实的重建结果
        completeness = max(0, min(1, 1.0 - abs(voxel_size - 0.01) * 10))
        accuracy = max(0, min(1, 1.0 - abs(offset_distance - 0.06) * 5))
        detail_quality = max(0, min(1, 1.0 - abs(detail_threshold - 0.005) * 100))
        
        # 添加一些随机噪声来模拟真实情况
        noise_factor = 0.1
        completeness += np.random.normal(0, noise_factor)
        accuracy += np.random.normal(0, noise_factor)
        detail_quality += np.random.normal(0, noise_factor)
        
        # 确保在[0,1]范围内
        completeness = max(0, min(1, completeness))
        accuracy = max(0, min(1, accuracy))
        detail_quality = max(0, min(1, detail_quality))
        
        # 计算综合评分
        overall_score = (completeness * 0.4 + accuracy * 0.4 + detail_quality * 0.2)
        
        # 模拟处理时间
        processing_time = 10 + np.random.exponential(5)  # 秒
        
        return {
            'completeness': completeness,
            'accuracy': accuracy,
            'detail_quality': detail_quality,
            'overall_score': overall_score,
            'processing_time': processing_time,
            'parameters': params
        }
    
    def run_parameter_tuning(self, test_data: str, max_iterations: int = 20) -> Dict:
        """运行参数调优"""
        print("开始参数调优...")
        print("=" * 60)
        
        # 生成参数组合
        param_combinations = self.generate_parameter_combinations(max_iterations)
        
        best_result = None
        best_score = -1
        
        for i, params in enumerate(param_combinations):
            print(f"\n进度: {i+1}/{len(param_combinations)}")
            print(f"测试参数组合 {i+1}...")
            
            # 评估当前参数组合
            result = self.evaluate_parameters(params, test_data)
            
            # 记录结果
            result['iteration'] = i + 1
            result['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            self.tuning_history.append(result)
            
            # 更新最佳结果
            if result['overall_score'] > best_score:
                best_score = result['overall_score']
                best_result = result.copy()
                print(f"🎉 发现更好的参数组合！评分: {best_score:.3f}")
            
            print(f"当前评分: {result['overall_score']:.3f}")
            print(f"  完整性: {result['completeness']:.3f}")
            print(f"  准确性: {result['accuracy']:.3f}")
            print(f"  细节质量: {result['detail_quality']:.3f}")
            print(f"  处理时间: {result['processing_time']:.1f}秒")
        
        # 保存调优结果
        self._save_tuning_results(best_result)
        
        print("\n" + "=" * 60)
        print("参数调优完成！")
        print(f"最佳评分: {best_score:.3f}")
        print(f"最佳参数已保存到配置文件")
        
        return best_result
    
    def _save_tuning_results(self, best_result: Dict):
        """保存调优结果"""
        # 保存最佳参数到配置文件
        best_params = best_result['parameters']
        
        for module, module_params in best_params.items():
            config_file = self.config_dir / f"{module}.yml"
            
            # 加载当前配置
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    current_config = yaml.safe_load(f)
            except FileNotFoundError:
                current_config = {}
            
            # 更新参数
            self._update_nested_dict(current_config, module_params)
            
            # 保存更新后的配置
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)
        
        # 保存调优历史
        history_file = self.output_dir / "tuning_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.tuning_history, f, indent=2, ensure_ascii=False)
        
        # 保存最佳结果
        best_result_file = self.output_dir / "best_parameters.json"
        with open(best_result_file, 'w', encoding='utf-8') as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
        
        # 生成调优报告
        self._generate_tuning_report(best_result)
    
    def _generate_tuning_report(self, best_result: Dict):
        """生成调优报告"""
        report_file = self.output_dir / "tuning_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("参数调优报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"调优时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试组合数: {len(self.tuning_history)}\n\n")
            
            f.write("最佳结果:\n")
            f.write(f"  综合评分: {best_result['overall_score']:.3f}\n")
            f.write(f"  完整性: {best_result['completeness']:.3f}\n")
            f.write(f"  准确性: {best_result['accuracy']:.3f}\n")
            f.write(f"  细节质量: {best_result['detail_quality']:.3f}\n")
            f.write(f"  处理时间: {best_result['processing_time']:.1f}秒\n\n")
            
            f.write("最佳参数:\n")
            for module, params in best_result['parameters'].items():
                f.write(f"  {module}:\n")
                self._write_params_recursive(f, params, indent=4)
            
            f.write("\n调优历史统计:\n")
            scores = [result['overall_score'] for result in self.tuning_history]
            f.write(f"  平均评分: {np.mean(scores):.3f}\n")
            f.write(f"  最高评分: {np.max(scores):.3f}\n")
            f.write(f"  最低评分: {np.min(scores):.3f}\n")
            f.write(f"  标准差: {np.std(scores):.3f}\n")
        
        print(f"调优报告已保存到: {report_file}")
    
    def _write_params_recursive(self, f, params: Dict, indent: int = 0):
        """递归写入参数"""
        for key, value in params.items():
            if isinstance(value, dict):
                f.write(" " * indent + f"{key}:\n")
                self._write_params_recursive(f, value, indent + 2)
            else:
                f.write(" " * indent + f"{key}: {value}\n")
    
    def analyze_parameter_sensitivity(self) -> Dict:
        """分析参数敏感性"""
        print("分析参数敏感性...")
        
        if len(self.tuning_history) < 5:
            print("调优历史数据不足，无法进行敏感性分析")
            return {}
        
        # 提取所有参数和对应的评分
        param_effects = {}
        
        for result in self.tuning_history:
            score = result['overall_score']
            params = result['parameters']
            
            # 展平参数字典
            flat_params = self._flatten_dict(params)
            
            for param_name, param_value in flat_params.items():
                if param_name not in param_effects:
                    param_effects[param_name] = []
                param_effects[param_name].append((param_value, score))
        
        # 计算每个参数的影响
        sensitivity_results = {}
        for param_name, value_score_pairs in param_effects.items():
            if len(set(pair[0] for pair in value_score_pairs)) > 1:  # 参数有变化
                values = [pair[0] for pair in value_score_pairs]
                scores = [pair[1] for pair in value_score_pairs]
                
                # 计算相关系数
                correlation = np.corrcoef(values, scores)[0, 1] if len(values) > 1 else 0
                
                sensitivity_results[param_name] = {
                    'correlation': correlation,
                    'score_range': max(scores) - min(scores),
                    'param_range': max(values) - min(values) if isinstance(values[0], (int, float)) else 'categorical'
                }
        
        # 保存敏感性分析结果
        sensitivity_file = self.output_dir / "parameter_sensitivity.json"
        with open(sensitivity_file, 'w', encoding='utf-8') as f:
            json.dump(sensitivity_results, f, indent=2, ensure_ascii=False)
        
        print(f"参数敏感性分析结果已保存到: {sensitivity_file}")
        return sensitivity_results
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """展平嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="参数调优工具")
    parser.add_argument("--test-data", default="data/test/test_cube.ply", help="测试数据文件")
    parser.add_argument("--max-iterations", type=int, default=20, help="最大迭代次数")
    parser.add_argument("--config-dir", help="配置文件目录")
    
    args = parser.parse_args()
    
    # 创建参数调优器
    tuner = ParameterTuner(args.config_dir)
    
    # 运行参数调优
    best_result = tuner.run_parameter_tuning(args.test_data, args.max_iterations)
    
    # 分析参数敏感性
    sensitivity_results = tuner.analyze_parameter_sensitivity()
    
    print("\n🎉 参数调优完成！")
    print(f"最佳综合评分: {best_result['overall_score']:.3f}")


if __name__ == "__main__":
    main()


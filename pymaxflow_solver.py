#!/usr/bin/env python3
"""
PyMaxflow求解器接口
为C++代码提供高性能的图割求解服务
"""

import maxflow
import json
import sys
import time
from typing import Dict, List, Tuple, Any

class PyMaxflowSolver:
    """
    PyMaxflow求解器包装类
    提供与C++代码兼容的接口
    """
    
    def __init__(self):
        self.graph = None
        self.num_nodes = 0
        self.solve_time = 0.0
        self.flow_value = 0.0
        
    def create_graph(self, num_nodes: int) -> bool:
        """
        创建图结构
        
        Args:
            num_nodes: 节点数量
            
        Returns:
            bool: 创建是否成功
        """
        try:
            self.graph = maxflow.Graph[float]()
            self.num_nodes = num_nodes
            
            # 添加节点
            self.nodes = self.graph.add_nodes(num_nodes)
            
            return True
        except Exception as e:
            print(f"Error creating graph: {e}", file=sys.stderr)
            return False
    
    def add_terminal_edge(self, node_id: int, source_cap: float, sink_cap: float) -> bool:
        """
        添加源汇边（数据项）
        
        Args:
            node_id: 节点ID
            source_cap: 到源的容量
            sink_cap: 到汇的容量
            
        Returns:
            bool: 添加是否成功
        """
        try:
            if node_id < 0 or node_id >= self.num_nodes:
                return False
                
            self.graph.add_tedge(self.nodes[node_id], source_cap, sink_cap)
            return True
        except Exception as e:
            print(f"Error adding terminal edge: {e}", file=sys.stderr)
            return False
    
    def add_edge(self, from_node: int, to_node: int, capacity: float, rev_capacity: float = None) -> bool:
        """
        添加普通边（平滑项）
        
        Args:
            from_node: 起始节点ID
            to_node: 目标节点ID
            capacity: 正向容量
            rev_capacity: 反向容量（如果为None，则等于正向容量）
            
        Returns:
            bool: 添加是否成功
        """
        try:
            if (from_node < 0 or from_node >= self.num_nodes or 
                to_node < 0 or to_node >= self.num_nodes):
                return False
                
            if rev_capacity is None:
                rev_capacity = capacity
                
            self.graph.add_edge(self.nodes[from_node], self.nodes[to_node], 
                              capacity, rev_capacity)
            return True
        except Exception as e:
            print(f"Error adding edge: {e}", file=sys.stderr)
            return False
    
    def solve(self) -> float:
        """
        求解最大流
        
        Returns:
            float: 最大流值
        """
        try:
            start_time = time.time()
            self.flow_value = self.graph.maxflow()
            self.solve_time = time.time() - start_time
            
            return self.flow_value
        except Exception as e:
            print(f"Error solving maxflow: {e}", file=sys.stderr)
            return -1.0
    
    def get_segment(self, node_id: int) -> int:
        """
        获取节点的分割结果
        
        Args:
            node_id: 节点ID
            
        Returns:
            int: 0表示源侧，1表示汇侧，-1表示错误
        """
        try:
            if node_id < 0 or node_id >= self.num_nodes:
                return -1
                
            return self.graph.get_segment(self.nodes[node_id])
        except Exception as e:
            print(f"Error getting segment: {e}", file=sys.stderr)
            return -1
    
    def get_all_segments(self) -> List[int]:
        """
        获取所有节点的分割结果
        
        Returns:
            List[int]: 分割结果列表
        """
        try:
            segments = []
            for i in range(self.num_nodes):
                segments.append(self.graph.get_segment(self.nodes[i]))
            return segments
        except Exception as e:
            print(f"Error getting all segments: {e}", file=sys.stderr)
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取求解统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            'num_nodes': self.num_nodes,
            'flow_value': self.flow_value,
            'solve_time': self.solve_time,
            'nodes_per_second': self.num_nodes / self.solve_time if self.solve_time > 0 else 0
        }

def process_json_request(request_data: Dict) -> Dict:
    """
    处理JSON格式的求解请求
    
    Args:
        request_data: 请求数据
        
    Returns:
        Dict: 响应数据
    """
    try:
        command = request_data.get('command', '')
        
        if command == 'solve_graph':
            # 完整的图割求解请求
            num_nodes = request_data['num_nodes']
            terminal_edges = request_data['terminal_edges']  # [(node_id, source_cap, sink_cap), ...]
            edges = request_data['edges']  # [(from, to, cap, rev_cap), ...]
            
            # 创建求解器
            solver = PyMaxflowSolver()
            
            # 创建图
            if not solver.create_graph(num_nodes):
                return {'success': False, 'error': 'Failed to create graph'}
            
            # 添加源汇边
            for node_id, source_cap, sink_cap in terminal_edges:
                if not solver.add_terminal_edge(node_id, source_cap, sink_cap):
                    return {'success': False, 'error': f'Failed to add terminal edge for node {node_id}'}
            
            # 添加普通边
            for edge_data in edges:
                if len(edge_data) == 3:
                    from_node, to_node, capacity = edge_data
                    rev_capacity = capacity
                else:
                    from_node, to_node, capacity, rev_capacity = edge_data
                    
                if not solver.add_edge(from_node, to_node, capacity, rev_capacity):
                    return {'success': False, 'error': f'Failed to add edge {from_node}->{to_node}'}
            
            # 求解
            flow_value = solver.solve()
            if flow_value < 0:
                return {'success': False, 'error': 'Failed to solve maxflow'}
            
            # 获取结果
            segments = solver.get_all_segments()
            statistics = solver.get_statistics()
            
            return {
                'success': True,
                'flow_value': flow_value,
                'segments': segments,
                'statistics': statistics
            }
        
        else:
            return {'success': False, 'error': f'Unknown command: {command}'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    """
    主函数：处理命令行参数或标准输入
    """
    if len(sys.argv) > 1:
        # 从命令行参数读取JSON文件
        json_file = sys.argv[1]
        try:
            with open(json_file, 'r') as f:
                request_data = json.load(f)
        except Exception as e:
            print(json.dumps({'success': False, 'error': f'Failed to read JSON file: {e}'}))
            return
    else:
        # 从标准输入读取JSON数据
        try:
            request_data = json.load(sys.stdin)
        except Exception as e:
            print(json.dumps({'success': False, 'error': f'Failed to parse JSON input: {e}'}))
            return
    
    # 处理请求
    response = process_json_request(request_data)
    
    # 输出响应
    print(json.dumps(response))

if __name__ == '__main__':
    main()


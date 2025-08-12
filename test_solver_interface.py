#!/usr/bin/env python3
"""
测试PyMaxflow求解器接口
"""

import json
import subprocess
import time

def test_solver_interface():
    """
    测试求解器接口
    """
    print("测试PyMaxflow求解器接口")
    print("=" * 40)
    
    # 创建测试数据
    test_request = {
        'command': 'solve_graph',
        'num_nodes': 4,
        'terminal_edges': [
            [0, 1.0, 0.5],  # 节点0：更倾向于源
            [1, 0.5, 1.0],  # 节点1：更倾向于汇
            [2, 0.8, 0.8],  # 节点2：中性
            [3, 0.3, 1.2]   # 节点3：更倾向于汇
        ],
        'edges': [
            [0, 1, 0.4, 0.4],  # 0-1边
            [1, 2, 0.3, 0.3],  # 1-2边
            [2, 3, 0.5, 0.5],  # 2-3边
            [0, 2, 0.2, 0.2]   # 0-2边
        ]
    }
    
    print("测试数据:")
    print(f"  节点数: {test_request['num_nodes']}")
    print(f"  源汇边数: {len(test_request['terminal_edges'])}")
    print(f"  平滑边数: {len(test_request['edges'])}")
    
    # 调用求解器
    start_time = time.time()
    
    try:
        # 通过subprocess调用Python求解器
        process = subprocess.Popen(
            ['python3', 'pymaxflow_solver.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=json.dumps(test_request))
        
        if process.returncode != 0:
            print(f"✗ 求解器执行失败: {stderr}")
            return False
        
        # 解析响应
        response = json.loads(stdout)
        
        end_time = time.time()
        
        if response['success']:
            print("✓ 求解成功!")
            print(f"  最大流值: {response['flow_value']:.3f}")
            print(f"  分割结果: {response['segments']}")
            print(f"  求解时间: {response['statistics']['solve_time']:.6f}s")
            print(f"  总时间: {end_time - start_time:.6f}s")
            print(f"  求解速度: {response['statistics']['nodes_per_second']:.0f} 节点/秒")
            return True
        else:
            print(f"✗ 求解失败: {response['error']}")
            return False
            
    except Exception as e:
        print(f"✗ 接口测试失败: {e}")
        return False

def test_large_graph():
    """
    测试大规模图
    """
    print("\n测试大规模图")
    print("=" * 40)
    
    # 创建较大的测试图
    num_nodes = 1000
    terminal_edges = []
    edges = []
    
    # 添加源汇边
    import random
    for i in range(num_nodes):
        source_cap = random.uniform(0.1, 2.0)
        sink_cap = random.uniform(0.1, 2.0)
        terminal_edges.append([i, source_cap, sink_cap])
    
    # 添加随机边
    num_edges = num_nodes // 2
    for _ in range(num_edges):
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)
        if i != j:
            weight = random.uniform(0.1, 1.0)
            edges.append([i, j, weight, weight])
    
    test_request = {
        'command': 'solve_graph',
        'num_nodes': num_nodes,
        'terminal_edges': terminal_edges,
        'edges': edges
    }
    
    print(f"测试数据: {num_nodes} 节点, {len(edges)} 边")
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            ['python3', 'pymaxflow_solver.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(input=json.dumps(test_request))
        
        if process.returncode != 0:
            print(f"✗ 大规模测试失败: {stderr}")
            return False
        
        response = json.loads(stdout)
        end_time = time.time()
        
        if response['success']:
            print("✓ 大规模测试成功!")
            print(f"  最大流值: {response['flow_value']:.3f}")
            print(f"  求解时间: {response['statistics']['solve_time']:.6f}s")
            print(f"  总时间: {end_time - start_time:.6f}s")
            print(f"  求解速度: {response['statistics']['nodes_per_second']:.0f} 节点/秒")
            
            # 统计分割结果
            segments = response['segments']
            source_count = segments.count(0)
            sink_count = segments.count(1)
            print(f"  源侧节点: {source_count}, 汇侧节点: {sink_count}")
            
            return True
        else:
            print(f"✗ 大规模测试失败: {response['error']}")
            return False
            
    except Exception as e:
        print(f"✗ 大规模测试异常: {e}")
        return False

if __name__ == '__main__':
    print("PyMaxflow求解器接口测试")
    print("=" * 50)
    
    # 基础功能测试
    success1 = test_solver_interface()
    
    # 大规模测试
    success2 = test_large_graph()
    
    print("\n测试总结:")
    if success1 and success2:
        print("✓ 所有测试通过，接口工作正常")
        print("✓ 可以开始集成到C++代码中")
    else:
        print("✗ 部分测试失败，需要修复问题")


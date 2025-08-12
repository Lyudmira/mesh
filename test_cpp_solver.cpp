#include "recon/src/graph_cut/pymaxflow_solver.h"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "测试PyMaxflow C++接口" << std::endl;
    std::cout << "=====================" << std::endl;
    
    // 基础功能测试
    std::cout << "\n1. 基础功能测试" << std::endl;
    
    try {
        // 创建求解器
        recon::PyMaxflowSolver solver(4, "./pymaxflow_solver.py");
        
        // 添加源汇边
        solver.addTerminalEdge(0, 1.0f, 0.5f);  // 更倾向于源
        solver.addTerminalEdge(1, 0.5f, 1.0f);  // 更倾向于汇
        solver.addTerminalEdge(2, 0.8f, 0.8f);  // 中性
        solver.addTerminalEdge(3, 0.3f, 1.2f);  // 更倾向于汇
        
        // 添加平滑边
        solver.addEdge(0, 1, 0.4f, 0.4f);
        solver.addEdge(1, 2, 0.3f, 0.3f);
        solver.addEdge(2, 3, 0.5f, 0.5f);
        solver.addEdge(0, 2, 0.2f, 0.2f);
        
        // 求解
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = solver.solve();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        if (result.success) {
            std::cout << "✓ 基础测试成功!" << std::endl;
            std::cout << "  最大流值: " << result.flow_value << std::endl;
            std::cout << "  求解时间: " << result.solve_time << "s" << std::endl;
            std::cout << "  总时间: " << total_time << "s" << std::endl;
            
            std::cout << "  分割结果: ";
            for (size_t i = 0; i < result.segments.size(); ++i) {
                std::cout << result.segments[i];
                if (i < result.segments.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            
            std::cout << "  isSourceSide检查: ";
            for (int i = 0; i < 4; ++i) {
                std::cout << i << "=" << (solver.isSourceSide(i) ? "源" : "汇");
                if (i < 3) std::cout << ", ";
            }
            std::cout << std::endl;
            
            // 获取统计信息
            auto stats = solver.getStatistics(result);
            std::cout << "  统计信息:" << std::endl;
            std::cout << "    节点数: " << stats.num_nodes << std::endl;
            std::cout << "    源汇边数: " << stats.num_terminal_edges << std::endl;
            std::cout << "    平滑边数: " << stats.num_edges << std::endl;
            std::cout << "    源侧节点: " << stats.source_nodes << std::endl;
            std::cout << "    汇侧节点: " << stats.sink_nodes << std::endl;
            
        } else {
            std::cout << "✗ 基础测试失败: " << result.error_message << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "✗ 基础测试异常: " << e.what() << std::endl;
        return 1;
    }
    
    // 性能测试
    std::cout << "\n2. 性能测试" << std::endl;
    
    try {
        const int num_nodes = 1000;
        recon::PyMaxflowSolver solver(num_nodes, "./pymaxflow_solver.py");
        
        // 添加随机源汇边
        for (int i = 0; i < num_nodes; ++i) {
            float source_cap = 0.5f + static_cast<float>(rand()) / RAND_MAX;
            float sink_cap = 0.5f + static_cast<float>(rand()) / RAND_MAX;
            solver.addTerminalEdge(i, source_cap, sink_cap);
        }
        
        // 添加随机平滑边
        const int num_edges = num_nodes / 2;
        for (int i = 0; i < num_edges; ++i) {
            int from = rand() % num_nodes;
            int to = rand() % num_nodes;
            if (from != to) {
                float weight = 0.1f + static_cast<float>(rand()) / RAND_MAX * 0.9f;
                solver.addEdge(from, to, weight, weight);
            }
        }
        
        std::cout << "  测试规模: " << num_nodes << " 节点, ~" << num_edges << " 边" << std::endl;
        
        // 求解
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = solver.solve();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        if (result.success) {
            std::cout << "✓ 性能测试成功!" << std::endl;
            std::cout << "  最大流值: " << result.flow_value << std::endl;
            std::cout << "  求解时间: " << result.solve_time << "s" << std::endl;
            std::cout << "  总时间: " << total_time << "s" << std::endl;
            
            auto stats = solver.getStatistics(result);
            std::cout << "  源侧节点: " << stats.source_nodes << std::endl;
            std::cout << "  汇侧节点: " << stats.sink_nodes << std::endl;
            
            if (result.solve_time > 0) {
                std::cout << "  求解速度: " << static_cast<int>(num_nodes / result.solve_time) << " 节点/秒" << std::endl;
            }
            
        } else {
            std::cout << "✗ 性能测试失败: " << result.error_message << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cout << "✗ 性能测试异常: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n测试总结:" << std::endl;
    std::cout << "✓ 所有测试通过" << std::endl;
    std::cout << "✓ PyMaxflow C++接口工作正常" << std::endl;
    std::cout << "✓ 可以集成到图割优化器中" << std::endl;
    
    return 0;
}


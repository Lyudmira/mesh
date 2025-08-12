/**
 * @file test_thread_safe_solver.cpp
 * @brief 线程安全PyMaxflow求解器测试程序
 */

#include "recon/src/graph_cut/thread_safe_pymaxflow_solver.h"
#include "recon/src/base/types.h"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <random>

using namespace recon::algorithms;
using namespace recon::core;

void testBasicFunctionality() {
    std::cout << "\n=== 基础功能测试 ===" << std::endl;
    
    // 检查依赖
    auto [deps_ok, deps_error] = GraphCutSolverFactory::checkDependencies();
    std::cout << "依赖检查: " << (deps_ok ? "通过" : "失败") << std::endl;
    if (!deps_ok) {
        std::cout << "错误: " << deps_error << std::endl;
        return;
    }
    
    // 创建求解器
    GraphCutConfig config;
    config.enable_detailed_logging = true;
    config.validate_graph = true;
    
    auto solver = GraphCutSolverFactory::create(4, config);
    
    // 构建简单图
    solver->addTerminalEdge(0, 10, 0);  // 源点
    solver->addTerminalEdge(3, 0, 10);  // 汇点
    solver->addEdge(0, 1, 5, 0);
    solver->addEdge(1, 2, 3, 0);
    solver->addEdge(2, 3, 7, 0);
    solver->addEdge(0, 2, 8, 0);
    
    // 获取图统计
    auto stats = solver->getGraphStats();
    std::cout << "图统计:" << std::endl;
    std::cout << "  节点数: " << stats.num_nodes << std::endl;
    std::cout << "  终端边数: " << stats.num_terminal_edges << std::endl;
    std::cout << "  普通边数: " << stats.num_edges << std::endl;
    std::cout << "  内存使用: " << stats.memory_usage_bytes << " 字节" << std::endl;
    std::cout << "  图有效性: " << (stats.is_valid ? "有效" : "无效") << std::endl;
    
    // 验证图结构
    auto [valid, error_msg] = solver->validateGraph();
    std::cout << "图验证: " << (valid ? "通过" : "失败") << std::endl;
    if (!valid) {
        std::cout << "验证错误: " << error_msg << std::endl;
    }
    
    // 求解
    std::cout << "\n开始求解..." << std::endl;
    auto result = solver->solve();
    
    if (result.success()) {
        std::cout << "求解成功!" << std::endl;
        std::cout << "  最大流值: " << result.flow_value << std::endl;
        std::cout << "  源侧节点数: " << result.source_nodes << std::endl;
        std::cout << "  汇侧节点数: " << result.sink_nodes << std::endl;
        std::cout << "  求解时间: " << result.performance.total_time << "s" << std::endl;
        
        std::cout << "  分割结果: ";
        for (size_t i = 0; i < result.segments.size(); ++i) {
            std::cout << "节点" << i << "=" << (result.segments[i] == 0 ? "源侧" : "汇侧") << " ";
        }
        std::cout << std::endl;
        
        // 验证isSourceSide方法
        std::cout << "  验证isSourceSide: ";
        for (int i = 0; i < 4; ++i) {
            bool is_source = solver->isSourceSide(i);
            bool expected = (result.segments[i] == 0);
            std::cout << "节点" << i << "=" << (is_source ? "源" : "汇") 
                     << (is_source == expected ? "✓" : "✗") << " ";
        }
        std::cout << std::endl;
        
    } else {
        std::cout << "求解失败!" << std::endl;
        std::cout << "  错误代码: " << static_cast<int>(result.error_code) << std::endl;
        std::cout << "  错误信息: " << result.error_message << std::endl;
    }
}

void testThreadSafety() {
    std::cout << "\n=== 线程安全测试 ===" << std::endl;
    
    // 检查依赖
    auto [deps_ok, deps_error] = GraphCutSolverFactory::checkDependencies();
    if (!deps_ok) {
        std::cout << "跳过线程安全测试，依赖不满足: " << deps_error << std::endl;
        return;
    }
    
    const int num_threads = 4;
    const int num_nodes = 100;
    
    GraphCutConfig config;
    config.enable_detailed_logging = false;  // 减少日志输出
    config.validate_graph = false;  // 提升性能
    
    auto solver = GraphCutSolverFactory::create(num_nodes, config);
    
    // 构建随机图
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> node_dist(0, num_nodes - 1);
    std::uniform_real_distribution<> cap_dist(1.0, 10.0);
    
    // 添加终端边
    solver->addTerminalEdge(0, 100, 0);  // 源点
    solver->addTerminalEdge(num_nodes - 1, 0, 100);  // 汇点
    
    // 添加随机边
    for (int i = 0; i < num_nodes * 2; ++i) {
        int from = node_dist(gen);
        int to = node_dist(gen);
        if (from != to) {
            solver->addEdge(from, to, cap_dist(gen), 0);
        }
    }
    
    std::cout << "构建了包含 " << num_nodes << " 个节点的随机图" << std::endl;
    
    // 多线程并发测试
    std::vector<std::thread> threads;
    std::vector<GraphCutResult> results(num_threads);
    std::atomic<int> completed_threads(0);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            // 每个线程执行求解
            results[i] = solver->solve();
            completed_threads++;
            
            std::cout << "线程 " << i << " 完成求解" << std::endl;
        });
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    std::cout << "\n线程安全测试结果:" << std::endl;
    std::cout << "  总时间: " << total_time << "s" << std::endl;
    std::cout << "  平均每线程时间: " << total_time / num_threads << "s" << std::endl;
    
    // 验证结果一致性
    bool all_success = true;
    bool results_consistent = true;
    float first_flow = -1;
    
    for (int i = 0; i < num_threads; ++i) {
        if (!results[i].success()) {
            all_success = false;
            std::cout << "  线程 " << i << " 失败: " << results[i].error_message << std::endl;
        } else {
            if (first_flow < 0) {
                first_flow = results[i].flow_value;
            } else if (std::abs(results[i].flow_value - first_flow) > 1e-6) {
                results_consistent = false;
                std::cout << "  线程 " << i << " 流值不一致: " << results[i].flow_value 
                         << " vs " << first_flow << std::endl;
            }
        }
    }
    
    std::cout << "  所有线程成功: " << (all_success ? "是" : "否") << std::endl;
    std::cout << "  结果一致性: " << (results_consistent ? "一致" : "不一致") << std::endl;
    
    if (all_success && results_consistent) {
        std::cout << "  ✅ 线程安全测试通过!" << std::endl;
    } else {
        std::cout << "  ❌ 线程安全测试失败!" << std::endl;
    }
}

void testPerformance() {
    std::cout << "\n=== 性能测试 ===" << std::endl;
    
    // 检查依赖
    auto [deps_ok, deps_error] = GraphCutSolverFactory::checkDependencies();
    if (!deps_ok) {
        std::cout << "跳过性能测试，依赖不满足: " << deps_error << std::endl;
        return;
    }
    
    std::vector<int> test_sizes = {100, 500, 1000, 2000};
    
    for (int size : test_sizes) {
        std::cout << "\n测试规模: " << size << " 节点" << std::endl;
        
        GraphCutConfig config = GraphCutSolverFactory::getRecommendedConfig(size, true);
        config.enable_detailed_logging = false;
        
        auto solver = GraphCutSolverFactory::create(size, config);
        
        // 构建密集图
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> node_dist(0, size - 1);
        std::uniform_real_distribution<> cap_dist(1.0, 10.0);
        
        // 添加终端边
        solver->addTerminalEdge(0, 100, 0);
        solver->addTerminalEdge(size - 1, 0, 100);
        
        // 添加边（每个节点平均连接4个其他节点）
        int num_edges = size * 4;
        for (int i = 0; i < num_edges; ++i) {
            int from = node_dist(gen);
            int to = node_dist(gen);
            if (from != to) {
                solver->addEdge(from, to, cap_dist(gen), 0);
            }
        }
        
        // 执行求解
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = solver->solve();
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (result.success()) {
            double solve_time = std::chrono::duration<double>(end_time - start_time).count();
            double throughput = size / solve_time;
            
            std::cout << "  求解时间: " << solve_time << "s" << std::endl;
            std::cout << "  处理速度: " << static_cast<int>(throughput) << " 节点/秒" << std::endl;
            std::cout << "  最大流值: " << result.flow_value << std::endl;
            std::cout << "  源侧/汇侧: " << result.source_nodes << "/" << result.sink_nodes << std::endl;
        } else {
            std::cout << "  求解失败: " << result.error_message << std::endl;
        }
    }
}

void testErrorHandling() {
    std::cout << "\n=== 错误处理测试 ===" << std::endl;
    
    GraphCutConfig config;
    config.enable_detailed_logging = true;
    
    // 测试1: 空图
    {
        std::cout << "\n测试1: 空图" << std::endl;
        auto solver = GraphCutSolverFactory::create(5, config);
        auto result = solver->solve();
        
        std::cout << "  结果: " << (result.success() ? "成功" : "失败") << std::endl;
        std::cout << "  错误: " << result.error_message << std::endl;
    }
    
    // 测试2: 无效节点索引
    {
        std::cout << "\n测试2: 无效节点索引" << std::endl;
        auto solver = GraphCutSolverFactory::create(5, config);
        
        bool result1 = solver->addTerminalEdge(-1, 10, 0);  // 负索引
        bool result2 = solver->addTerminalEdge(10, 10, 0);  // 超出范围
        bool result3 = solver->addEdge(0, 10, 5, 0);        // 超出范围
        
        std::cout << "  负索引终端边: " << (result1 ? "成功" : "失败") << std::endl;
        std::cout << "  超范围终端边: " << (result2 ? "成功" : "失败") << std::endl;
        std::cout << "  超范围普通边: " << (result3 ? "成功" : "失败") << std::endl;
    }
    
    // 测试3: 负容量
    {
        std::cout << "\n测试3: 负容量" << std::endl;
        auto solver = GraphCutSolverFactory::create(5, config);
        
        bool result1 = solver->addTerminalEdge(0, -5, 10);  // 负源容量
        bool result2 = solver->addEdge(0, 1, -3, 0);        // 负边容量
        
        std::cout << "  负源容量: " << (result1 ? "成功" : "失败") << std::endl;
        std::cout << "  负边容量: " << (result2 ? "成功" : "失败") << std::endl;
    }
    
    // 测试4: 自环
    {
        std::cout << "\n测试4: 自环" << std::endl;
        auto solver = GraphCutSolverFactory::create(5, config);
        
        bool result = solver->addEdge(2, 2, 5, 0);  // 自环
        std::cout << "  自环边: " << (result ? "成功" : "失败") << std::endl;
    }
}

int main() {
    std::cout << "=== 线程安全PyMaxflow求解器测试程序 ===" << std::endl;
    
    try {
        testBasicFunctionality();
        testThreadSafety();
        testPerformance();
        testErrorHandling();
        
        std::cout << "\n=== 所有测试完成 ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}


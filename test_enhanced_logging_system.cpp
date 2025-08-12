/**
 * @file test_enhanced_logging_system.cpp
 * @brief 增强logging和数据验证系统测试程序
 */

#include "recon/src/utils/logger.h"
#include "recon/src/utils/data_validator.h"
#include "recon/src/utils/performance_monitor.h"
#include "recon/src/base/types.h"
#include <iostream>
#include <thread>
#include <random>

using namespace recon::utils;
using namespace recon::core;

void testLoggingSystem() {
    std::cout << "\n=== 测试日志系统 ===" << std::endl;
    
    // 配置日志系统
    LogManager& logger = LogManager::getInstance();
    
    // 添加文件输出器
    logger.addFileAppender("logs/test_text.log", false, LogLevel::DEBUG);
    logger.addFileAppender("logs/test_json.log", true, LogLevel::INFO);
    
    // 设置模块日志级别
    logger.setModuleLevel("TestModule", LogLevel::DEBUG);
    logger.setModuleLevel("QuietModule", LogLevel::ERROR);
    
    std::cout << "配置完成，开始测试各种日志级别..." << std::endl;
    
    // 测试各种日志级别
    LOG_DEBUG("TestModule", "这是一条调试信息");
    LOG_INFO("TestModule", "这是一条信息");
    LOG_WARN("TestModule", "这是一条警告");
    LOG_ERROR("TestModule", "这是一条错误");
    LOG_FATAL("TestModule", "这是一条致命错误");
    
    // 测试带元数据的日志
    std::map<std::string, std::string> metadata;
    metadata["user_id"] = "12345";
    metadata["operation"] = "mesh_reconstruction";
    metadata["input_size"] = "50000";
    
    LOG_WITH_METADATA(LogLevel::INFO, "TestModule", "开始网格重建操作", metadata);
    
    // 测试被过滤的日志（QuietModule设置为ERROR级别）
    LOG_DEBUG("QuietModule", "这条调试信息不会显示");
    LOG_INFO("QuietModule", "这条信息也不会显示");
    LOG_ERROR("QuietModule", "只有这条错误会显示");
    
    // 测试多线程日志
    std::cout << "测试多线程日志..." << std::endl;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back([i]() {
            for (int j = 0; j < 5; ++j) {
                LOG_INFO("Thread" + std::to_string(i), 
                        "线程 " + std::to_string(i) + " 消息 " + std::to_string(j));
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // 刷新日志
    logger.flush();
    
    std::cout << "日志系统测试完成，检查 logs/ 目录下的日志文件" << std::endl;
}

void testDataValidation() {
    std::cout << "\n=== 测试数据验证系统 ===" << std::endl;
    
    // 创建测试点云
    PointCloud cloud;
    
    // 添加正常点
    for (int i = 0; i < 1000; ++i) {
        ColoredPoint point;
        point.position = Point3D(
            static_cast<float>(i % 10),
            static_cast<float>((i / 10) % 10),
            static_cast<float>(i / 100)
        );
        point.r = 0.5f + 0.5f * std::sin(i * 0.1f);
        point.g = 0.5f + 0.5f * std::cos(i * 0.1f);
        point.b = 0.5f;
        point.normal = Point3D(0, 0, 1);  // 简单的法向量
        
        cloud.addPoint(point);
    }
    
    // 添加一些问题点
    // 重复点
    cloud.addPoint(cloud.points[0]);
    cloud.addPoint(cloud.points[0]);
    
    // 异常颜色值
    ColoredPoint bad_color_point;
    bad_color_point.position = Point3D(100, 100, 100);  // 异常位置
    bad_color_point.r = 2.0f;  // 超出范围
    bad_color_point.g = -0.5f; // 超出范围
    bad_color_point.b = 0.5f;
    bad_color_point.normal = Point3D(0, 0, 0);  // 零法向量
    cloud.addPoint(bad_color_point);
    
    std::cout << "创建了包含 " << cloud.size() << " 个点的测试点云" << std::endl;
    
    // 验证点云
    auto cloud_report = PointCloudValidator::validate(cloud);
    
    std::cout << "\n点云验证结果:" << std::endl;
    std::cout << "  总体评分: " << cloud_report.getOverallScore() << std::endl;
    std::cout << "  验证通过: " << (cloud_report.passed ? "是" : "否") << std::endl;
    std::cout << "  有警告: " << (cloud_report.has_warnings ? "是" : "否") << std::endl;
    std::cout << "  有严重错误: " << (cloud_report.has_critical_failures ? "是" : "否") << std::endl;
    std::cout << "  问题数量: " << cloud_report.issues.size() << std::endl;
    
    // 显示问题详情
    for (const auto& issue : cloud_report.issues) {
        std::string severity_str;
        switch (issue.severity) {
            case ValidationResult::WARNING: severity_str = "警告"; break;
            case ValidationResult::FAIL: severity_str = "失败"; break;
            case ValidationResult::CRITICAL_FAIL: severity_str = "严重失败"; break;
            default: severity_str = "通过"; break;
        }
        
        std::cout << "  [" << severity_str << "] " << issue.category 
                 << ": " << issue.description << std::endl;
        
        if (!issue.suggestion.empty()) {
            std::cout << "    建议: " << issue.suggestion << std::endl;
        }
        
        for (const auto& [key, value] : issue.metrics) {
            std::cout << "    " << key << ": " << value << std::endl;
        }
    }
    
    // 创建测试网格
    Mesh mesh;
    
    // 添加顶点
    for (int i = 0; i < 4; ++i) {
        ColoredPoint vertex;
        vertex.position = Point3D(
            static_cast<float>(i % 2),
            static_cast<float>(i / 2),
            0.0f
        );
        vertex.r = vertex.g = vertex.b = 0.5f;
        mesh.addVertex(vertex);
    }
    
    // 添加面
    mesh.addTriangle(Triangle(0, 1, 2));
    mesh.addTriangle(Triangle(1, 2, 3));
    
    // 添加退化面
    mesh.addTriangle(Triangle(0, 0, 1));  // 退化面
    
    std::cout << "\n创建了包含 " << mesh.vertexCount() << " 个顶点和 " 
             << mesh.faceCount() << " 个面的测试网格" << std::endl;
    
    // 验证网格
    auto mesh_report = MeshValidator::validate(mesh);
    
    std::cout << "\n网格验证结果:" << std::endl;
    std::cout << "  总体评分: " << mesh_report.getOverallScore() << std::endl;
    std::cout << "  验证通过: " << (mesh_report.passed ? "是" : "否") << std::endl;
    std::cout << "  问题数量: " << mesh_report.issues.size() << std::endl;
    
    for (const auto& issue : mesh_report.issues) {
        std::string severity_str;
        switch (issue.severity) {
            case ValidationResult::WARNING: severity_str = "警告"; break;
            case ValidationResult::FAIL: severity_str = "失败"; break;
            case ValidationResult::CRITICAL_FAIL: severity_str = "严重失败"; break;
            default: severity_str = "通过"; break;
        }
        
        std::cout << "  [" << severity_str << "] " << issue.category 
                 << ": " << issue.description << std::endl;
    }
    
    std::cout << "数据验证系统测试完成" << std::endl;
}

void testPerformanceMonitoring() {
    std::cout << "\n=== 测试性能监控系统 ===" << std::endl;
    
    PerformanceMonitor& monitor = PerformanceMonitor::getInstance();
    
    // 启动实时监控
    monitor.startMonitoring();
    monitor.setMonitoringInterval(std::chrono::seconds(1));
    
    std::cout << "开始性能监控..." << std::endl;
    
    // 测试计时器
    {
        PERF_TIMER("test_operation");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // 嵌套计时器
        {
            PERF_TIMER_CATEGORY("nested_operation", "algorithm");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    // 测试计数器
    PERF_COUNTER("processed_points", 1000);
    PERF_COUNTER("processed_points", 2000);
    PERF_COUNTER("processed_points", 1500);
    
    // 测试瞬时值
    PERF_GAUGE("current_memory_mb", 256.5);
    PERF_GAUGE("current_cpu_percent", 45.2);
    
    // 测试速率
    PERF_RATE("points_per_second", 50000);
    PERF_RATE("triangles_per_second", 25000);
    
    // 记录内存使用
    PERF_MEMORY("test_start");
    
    // 模拟一些工作负载
    std::vector<int> large_vector(1000000, 42);
    PERF_MEMORY("after_allocation");
    
    // 模拟算法性能测试
    std::cout << "模拟算法性能测试..." << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.001, 0.1);
    
    for (int i = 0; i < 10; ++i) {
        auto timer = monitor.createTimer("algorithm_iteration_" + std::to_string(i), "algorithm");
        
        // 模拟算法工作
        double work_time = dis(gen);
        std::this_thread::sleep_for(std::chrono::duration<double>(work_time));
        
        // 记录一些指标
        monitor.recordCounter("iterations_completed", 1, "count", "algorithm");
        monitor.recordRate("processing_rate", 1000.0 / work_time, "items/sec", "algorithm");
    }
    
    // 等待一段时间让监控收集数据
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // 获取聚合指标
    auto aggregated = monitor.getAggregatedMetrics();
    
    std::cout << "\n聚合性能指标:" << std::endl;
    for (const auto& [name, value] : aggregated) {
        std::cout << "  " << name << ": " << value << std::endl;
    }
    
    // 生成报告
    monitor.generateReport("logs/performance_report.txt");
    monitor.generateJsonReport("logs/performance_report.json");
    
    // 停止监控
    monitor.stopMonitoring();
    
    std::cout << "性能监控测试完成，检查 logs/ 目录下的报告文件" << std::endl;
}

void testIntegratedSystem() {
    std::cout << "\n=== 测试集成系统 ===" << std::endl;
    
    // 模拟完整的重建流程
    LOG_INFO("IntegratedTest", "开始集成系统测试");
    
    {
        PERF_TIMER_CATEGORY("integrated_reconstruction", "reconstruction");
        
        // 阶段1: 数据加载
        {
            PERF_TIMER_CATEGORY("data_loading", "io");
            LOG_INFO("IntegratedTest", "加载点云数据");
            
            // 创建测试数据
            PointCloud cloud;
            for (int i = 0; i < 5000; ++i) {
                ColoredPoint point;
                point.position = Point3D(
                    static_cast<float>(std::sin(i * 0.01) * 10),
                    static_cast<float>(std::cos(i * 0.01) * 10),
                    static_cast<float>(i * 0.001)
                );
                point.r = point.g = point.b = 0.7f;
                point.normal = Point3D(0, 0, 1);
                cloud.addPoint(point);
            }
            
            PERF_COUNTER("loaded_points", cloud.size());
            PERF_MEMORY("after_data_loading");
            
            // 验证输入数据
            auto validation_report = PointCloudValidator::validate(cloud);
            LOG_INFO("IntegratedTest", "数据验证完成，评分: " + 
                    std::to_string(validation_report.getOverallScore()));
            
            if (!validation_report.passed) {
                LOG_WARN("IntegratedTest", "输入数据存在问题，但继续处理");
                for (const auto& issue : validation_report.issues) {
                    if (issue.severity == ValidationResult::CRITICAL_FAIL) {
                        LOG_ERROR("IntegratedTest", "严重问题: " + issue.description);
                    }
                }
            }
        }
        
        // 阶段2: 预处理
        {
            PERF_TIMER_CATEGORY("preprocessing", "algorithm");
            LOG_INFO("IntegratedTest", "执行数据预处理");
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            PERF_COUNTER("preprocessed_points", 4800);
            PERF_MEMORY("after_preprocessing");
        }
        
        // 阶段3: 网格重建
        {
            PERF_TIMER_CATEGORY("mesh_reconstruction", "algorithm");
            LOG_INFO("IntegratedTest", "执行网格重建");
            
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            
            // 创建结果网格
            Mesh result_mesh;
            for (int i = 0; i < 100; ++i) {
                ColoredPoint vertex;
                vertex.position = Point3D(
                    static_cast<float>(i % 10),
                    static_cast<float>(i / 10),
                    0.0f
                );
                vertex.r = vertex.g = vertex.b = 0.8f;
                result_mesh.addVertex(vertex);
            }
            
            // 添加一些三角形
            for (int i = 0; i < 90; ++i) {
                if (i % 10 < 9 && i / 10 < 9) {
                    result_mesh.addTriangle(Triangle(i, i + 1, i + 10));
                    result_mesh.addTriangle(Triangle(i + 1, i + 11, i + 10));
                }
            }
            
            PERF_COUNTER("generated_vertices", result_mesh.vertexCount());
            PERF_COUNTER("generated_triangles", result_mesh.faceCount());
            PERF_MEMORY("after_mesh_reconstruction");
            
            // 验证输出网格
            auto mesh_validation = MeshValidator::validate(result_mesh);
            LOG_INFO("IntegratedTest", "网格验证完成，评分: " + 
                    std::to_string(mesh_validation.getOverallScore()));
            
            if (mesh_validation.has_warnings || !mesh_validation.passed) {
                LOG_WARN("IntegratedTest", "输出网格存在质量问题");
                for (const auto& issue : mesh_validation.issues) {
                    LOG_WARN("IntegratedTest", issue.category + ": " + issue.description);
                }
            }
        }
        
        // 阶段4: 后处理
        {
            PERF_TIMER_CATEGORY("postprocessing", "algorithm");
            LOG_INFO("IntegratedTest", "执行后处理");
            
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            PERF_MEMORY("after_postprocessing");
        }
    }
    
    LOG_INFO("IntegratedTest", "集成系统测试完成");
    
    // 生成最终报告
    PerformanceMonitor::getInstance().generateReport("logs/integrated_test_report.txt");
    LogManager::getInstance().flush();
    
    std::cout << "集成系统测试完成，检查日志文件获取详细信息" << std::endl;
}

int main() {
    std::cout << "=== 增强Logging和数据验证系统测试程序 ===" << std::endl;
    
    // 创建日志目录
    system("mkdir -p logs");
    
    try {
        testLoggingSystem();
        testDataValidation();
        testPerformanceMonitoring();
        testIntegratedSystem();
        
        std::cout << "\n=== 所有测试完成 ===" << std::endl;
        std::cout << "检查 logs/ 目录下的输出文件:" << std::endl;
        std::cout << "  - test_text.log: 文本格式日志" << std::endl;
        std::cout << "  - test_json.log: JSON格式日志" << std::endl;
        std::cout << "  - performance_report.txt: 性能报告" << std::endl;
        std::cout << "  - performance_report.json: JSON性能报告" << std::endl;
        std::cout << "  - integrated_test_report.txt: 集成测试报告" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}


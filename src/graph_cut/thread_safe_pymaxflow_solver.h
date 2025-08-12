/**
 * @file thread_safe_pymaxflow_solver.h
 * @brief 线程安全的PyMaxflow求解器
 * 
 * 本文件提供了线程安全的图割求解器实现，解决了原版本中的
 * 并发访问问题，并增强了错误处理和性能监控能力。
 * 
 * @version 2.0
 * @date 2025-08-12
 */

#ifndef RECON_GRAPH_CUT_THREAD_SAFE_PYMAXFLOW_SOLVER_H
#define RECON_GRAPH_CUT_THREAD_SAFE_PYMAXFLOW_SOLVER_H

#include "../base/types.h"
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <memory>
#include <chrono>

namespace recon {
namespace algorithms {

/**
 * @brief 图割求解错误类型
 */
enum class GraphCutError {
    SUCCESS,
    EMPTY_GRAPH,
    INVALID_NODE_INDEX,
    NO_TERMINAL_EDGES,
    PYTHON_SOLVER_NOT_FOUND,
    PYTHON_EXECUTION_FAILED,
    JSON_PARSING_FAILED,
    MEMORY_ALLOCATION_FAILED,
    GRAPH_VALIDATION_FAILED,
    TIMEOUT_EXCEEDED
};

/**
 * @brief 图割求解结果
 */
struct GraphCutResult {
    GraphCutError error_code = GraphCutError::SUCCESS;
    std::string error_message;
    
    float flow_value = 0.0f;
    std::vector<int> segments;  // 0=源侧, 1=汇侧
    
    core::PerformanceStats performance;
    
    // 统计信息
    int source_nodes = 0;
    int sink_nodes = 0;
    
    bool success() const { return error_code == GraphCutError::SUCCESS; }
    
    std::string getErrorDescription() const {
        switch (error_code) {
            case GraphCutError::SUCCESS:
                return "Success";
            case GraphCutError::EMPTY_GRAPH:
                return "Graph has no nodes or edges";
            case GraphCutError::INVALID_NODE_INDEX:
                return "Invalid node index provided";
            case GraphCutError::NO_TERMINAL_EDGES:
                return "Graph has no terminal edges (source/sink connections)";
            case GraphCutError::PYTHON_SOLVER_NOT_FOUND:
                return "Python solver script not found";
            case GraphCutError::PYTHON_EXECUTION_FAILED:
                return "Python solver execution failed";
            case GraphCutError::JSON_PARSING_FAILED:
                return "Failed to parse JSON response from solver";
            case GraphCutError::MEMORY_ALLOCATION_FAILED:
                return "Memory allocation failed";
            case GraphCutError::GRAPH_VALIDATION_FAILED:
                return "Graph structure validation failed";
            case GraphCutError::TIMEOUT_EXCEEDED:
                return "Solver execution timeout exceeded";
            default:
                return "Unknown error";
        }
    }
};

/**
 * @brief 终端边（源汇边）
 */
struct TerminalEdge {
    int node;
    float source_capacity;
    float sink_capacity;
    
    TerminalEdge(int n, float s_cap, float t_cap) 
        : node(n), source_capacity(s_cap), sink_capacity(t_cap) {}
};

/**
 * @brief 图割求解器配置
 */
struct GraphCutConfig {
    std::string python_script_path;
    int timeout_seconds = 300;  // 5分钟超时
    bool use_persistent_process = false;  // 是否使用持久进程
    bool validate_graph = true;  // 是否验证图结构
    bool enable_detailed_logging = false;
    
    // 性能优化选项
    bool use_shared_memory = false;
    bool use_pipe_communication = false;
    size_t json_buffer_size = 1024 * 1024;  // 1MB JSON缓冲区
};

/**
 * @brief 线程安全的PyMaxflow求解器
 * 
 * 这个类提供了线程安全的图割求解功能，支持：
 * - 多线程并发访问
 * - 详细的错误处理和诊断
 * - 性能监控和统计
 * - 图结构验证
 * - 超时控制
 */
class ThreadSafePyMaxflowSolver {
public:
    /**
     * @brief 构造函数
     * @param num_nodes 图中节点数量
     * @param config 求解器配置
     */
    explicit ThreadSafePyMaxflowSolver(int num_nodes, const GraphCutConfig& config = GraphCutConfig());
    
    /**
     * @brief 析构函数
     */
    ~ThreadSafePyMaxflowSolver();
    
    // 禁用拷贝构造和赋值
    ThreadSafePyMaxflowSolver(const ThreadSafePyMaxflowSolver&) = delete;
    ThreadSafePyMaxflowSolver& operator=(const ThreadSafePyMaxflowSolver&) = delete;
    
    /**
     * @brief 添加终端边（源汇边）
     * @param node 节点索引
     * @param source_cap 从源点到该节点的容量
     * @param sink_cap 从该节点到汇点的容量
     * @return 是否添加成功
     */
    bool addTerminalEdge(int node, float source_cap, float sink_cap);
    
    /**
     * @brief 添加普通边
     * @param from 起始节点
     * @param to 目标节点
     * @param capacity 正向容量
     * @param rev_capacity 反向容量（默认为0）
     * @return 是否添加成功
     */
    bool addEdge(int from, int to, float capacity, float rev_capacity = 0.0f);
    
    /**
     * @brief 求解最大流最小割
     * @return 求解结果
     */
    GraphCutResult solve();
    
    /**
     * @brief 检查节点是否在源侧
     * @param node 节点索引
     * @return 是否在源侧（线程安全）
     */
    bool isSourceSide(int node) const;
    
    /**
     * @brief 清空图数据
     */
    void clear();
    
    /**
     * @brief 获取图统计信息
     * @return 图统计信息
     */
    struct GraphStats {
        int num_nodes;
        int num_terminal_edges;
        int num_edges;
        size_t memory_usage_bytes;
        bool is_valid;
    };
    
    GraphStats getGraphStats() const;
    
    /**
     * @brief 验证图结构
     * @return 验证结果和错误信息
     */
    std::pair<bool, std::string> validateGraph() const;
    
    /**
     * @brief 获取性能统计
     * @return 累积的性能统计信息
     */
    core::PerformanceStats getPerformanceStats() const;
    
private:
    // 配置和状态
    GraphCutConfig config_;
    int num_nodes_;
    
    // 图数据（线程安全保护）
    mutable std::mutex graph_mutex_;
    std::vector<TerminalEdge> terminal_edges_;
    std::vector<core::Edge> edges_;
    std::vector<bool> source_side_;  // 最后一次求解的结果
    
    // 性能统计（原子操作）
    mutable std::mutex stats_mutex_;
    core::PerformanceStats cumulative_stats_;
    std::atomic<int> solve_count_{0};
    
    // 持久进程管理
    struct PersistentProcess;
    std::unique_ptr<PersistentProcess> persistent_process_;
    
    // 内部方法
    GraphCutResult solveInternal();
    GraphCutResult solveWithFileIO();
    GraphCutResult solveWithPersistentProcess();
    GraphCutResult solveWithSharedMemory();
    
    std::string createJsonRequest() const;
    GraphCutResult parseJsonResponse(const std::string& json_response) const;
    
    bool validateNodeIndex(int node) const;
    bool validateGraphStructure() const;
    
    std::string findPythonScript() const;
    bool initializePersistentProcess();
    void cleanupPersistentProcess();
    
    // 错误处理辅助方法
    GraphCutResult makeError(GraphCutError error_code, const std::string& details = "") const;
    void logError(const std::string& message) const;
    void logInfo(const std::string& message) const;
    
    // 性能监控
    void updatePerformanceStats(const core::PerformanceStats& stats);
    
    // JSON处理（使用nlohmann/json库）
    std::string serializeToJson() const;
    GraphCutResult deserializeFromJson(const std::string& json) const;
};

/**
 * @brief 图割求解器工厂
 */
class GraphCutSolverFactory {
public:
    /**
     * @brief 创建求解器实例
     * @param num_nodes 节点数量
     * @param config 配置参数
     * @return 求解器智能指针
     */
    static std::unique_ptr<ThreadSafePyMaxflowSolver> create(
        int num_nodes, 
        const GraphCutConfig& config = GraphCutConfig());
    
    /**
     * @brief 检查系统依赖
     * @return 依赖检查结果和错误信息
     */
    static std::pair<bool, std::string> checkDependencies();
    
    /**
     * @brief 获取推荐配置
     * @param num_nodes 节点数量
     * @param performance_priority 是否优先性能（vs稳定性）
     * @return 推荐的配置
     */
    static GraphCutConfig getRecommendedConfig(int num_nodes, bool performance_priority = true);
};

} // namespace algorithms
} // namespace recon

#endif // RECON_GRAPH_CUT_THREAD_SAFE_PYMAXFLOW_SOLVER_H


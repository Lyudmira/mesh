/**
 * @file thread_safe_pymaxflow_solver.cpp
 * @brief 线程安全的PyMaxflow求解器实现
 * 
 * @version 2.0
 * @date 2025-08-12
 */

#include "thread_safe_pymaxflow_solver.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <algorithm>
#include <thread>

using json = nlohmann::json;

namespace recon {
namespace algorithms {

// ============================================================================
// 持久进程管理结构
// ============================================================================

struct ThreadSafePyMaxflowSolver::PersistentProcess {
    pid_t pid = -1;
    int input_pipe[2] = {-1, -1};
    int output_pipe[2] = {-1, -1};
    bool is_running = false;
    std::mutex process_mutex;
    
    ~PersistentProcess() {
        cleanup();
    }
    
    void cleanup() {
        if (is_running && pid > 0) {
            kill(pid, SIGTERM);
            waitpid(pid, nullptr, 0);
        }
        
        if (input_pipe[0] != -1) close(input_pipe[0]);
        if (input_pipe[1] != -1) close(input_pipe[1]);
        if (output_pipe[0] != -1) close(output_pipe[0]);
        if (output_pipe[1] != -1) close(output_pipe[1]);
        
        is_running = false;
        pid = -1;
    }
};

// ============================================================================
// 构造函数和析构函数
// ============================================================================

ThreadSafePyMaxflowSolver::ThreadSafePyMaxflowSolver(int num_nodes, const GraphCutConfig& config)
    : config_(config), num_nodes_(num_nodes), source_side_(num_nodes, false) {
    
    if (num_nodes <= 0) {
        throw std::invalid_argument("Number of nodes must be positive");
    }
    
    // 查找Python脚本
    if (config_.python_script_path.empty()) {
        config_.python_script_path = findPythonScript();
    }
    
    // 初始化持久进程（如果启用）
    if (config_.use_persistent_process) {
        if (!initializePersistentProcess()) {
            logError("Failed to initialize persistent process, falling back to file I/O");
            config_.use_persistent_process = false;
        }
    }
    
    logInfo("ThreadSafePyMaxflowSolver initialized with " + std::to_string(num_nodes) + " nodes");
}

ThreadSafePyMaxflowSolver::~ThreadSafePyMaxflowSolver() {
    cleanupPersistentProcess();
}

// ============================================================================
// 图构建方法
// ============================================================================

bool ThreadSafePyMaxflowSolver::addTerminalEdge(int node, float source_cap, float sink_cap) {
    if (!validateNodeIndex(node)) {
        logError("Invalid node index: " + std::to_string(node));
        return false;
    }
    
    if (source_cap < 0 || sink_cap < 0) {
        logError("Terminal edge capacities must be non-negative");
        return false;
    }
    
    std::lock_guard<std::mutex> lock(graph_mutex_);
    terminal_edges_.emplace_back(node, source_cap, sink_cap);
    return true;
}

bool ThreadSafePyMaxflowSolver::addEdge(int from, int to, float capacity, float rev_capacity) {
    if (!validateNodeIndex(from) || !validateNodeIndex(to)) {
        logError("Invalid node indices: from=" + std::to_string(from) + ", to=" + std::to_string(to));
        return false;
    }
    
    if (from == to) {
        logError("Self-loops are not allowed: node=" + std::to_string(from));
        return false;
    }
    
    if (capacity < 0 || rev_capacity < 0) {
        logError("Edge capacities must be non-negative");
        return false;
    }
    
    std::lock_guard<std::mutex> lock(graph_mutex_);
    edges_.emplace_back(from, to, capacity, rev_capacity);
    return true;
}

// ============================================================================
// 求解方法
// ============================================================================

GraphCutResult ThreadSafePyMaxflowSolver::solve() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 验证图结构
    if (config_.validate_graph) {
        auto [valid, error_msg] = validateGraph();
        if (!valid) {
            return makeError(GraphCutError::GRAPH_VALIDATION_FAILED, error_msg);
        }
    }
    
    // 选择求解方法
    GraphCutResult result;
    if (config_.use_shared_memory) {
        result = solveWithSharedMemory();
    } else if (config_.use_persistent_process && persistent_process_) {
        result = solveWithPersistentProcess();
    } else {
        result = solveWithFileIO();
    }
    
    // 更新性能统计
    auto end_time = std::chrono::high_resolution_clock::now();
    result.performance.total_time = std::chrono::duration<double>(end_time - start_time).count();
    result.performance.input_size = num_nodes_;
    result.performance.output_size = result.segments.size();
    
    if (result.success()) {
        // 更新源侧标记（线程安全）
        {
            std::lock_guard<std::mutex> lock(graph_mutex_);
            if (result.segments.size() == static_cast<size_t>(num_nodes_)) {
                for (int i = 0; i < num_nodes_; ++i) {
                    source_side_[i] = (result.segments[i] == 0);
                }
            }
        }
        
        // 计算统计信息
        result.source_nodes = std::count(result.segments.begin(), result.segments.end(), 0);
        result.sink_nodes = result.segments.size() - result.source_nodes;
        
        solve_count_++;
        updatePerformanceStats(result.performance);
        
        logInfo("Graph cut solved successfully: flow=" + std::to_string(result.flow_value) + 
                ", time=" + std::to_string(result.performance.total_time) + "s");
    } else {
        logError("Graph cut solve failed: " + result.error_message);
    }
    
    return result;
}

GraphCutResult ThreadSafePyMaxflowSolver::solveWithFileIO() {
    try {
        // 创建JSON请求
        std::string json_request = serializeToJson();
        
        // 创建临时文件
        std::string temp_input = "/tmp/pymaxflow_input_" + std::to_string(getpid()) + "_" + 
                                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json";
        std::string temp_output = "/tmp/pymaxflow_output_" + std::to_string(getpid()) + "_" + 
                                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json";
        
        // 写入请求
        {
            std::ofstream input_file(temp_input);
            if (!input_file) {
                return makeError(GraphCutError::PYTHON_EXECUTION_FAILED, "Failed to create input file: " + temp_input);
            }
            input_file << json_request;
        }
        
        // 构建命令
        std::string command = "timeout " + std::to_string(config_.timeout_seconds) + 
                             " python3 \"" + config_.python_script_path + "\" \"" + temp_input + "\" > \"" + temp_output + "\" 2>&1";
        
        // 执行命令
        int exit_code = std::system(command.c_str());
        
        // 清理输入文件
        std::filesystem::remove(temp_input);
        
        if (exit_code != 0) {
            // 读取错误信息
            std::string error_output;
            std::ifstream error_file(temp_output);
            if (error_file) {
                std::string line;
                while (std::getline(error_file, line)) {
                    error_output += line + "\n";
                }
            }
            std::filesystem::remove(temp_output);
            
            if (WEXITSTATUS(exit_code) == 124) {  // timeout命令的超时退出码
                return makeError(GraphCutError::TIMEOUT_EXCEEDED, 
                               "Solver timeout after " + std::to_string(config_.timeout_seconds) + " seconds");
            } else {
                return makeError(GraphCutError::PYTHON_EXECUTION_FAILED, 
                               "Python solver failed with exit code " + std::to_string(exit_code) + 
                               "\nError output: " + error_output);
            }
        }
        
        // 读取响应
        std::ifstream output_file(temp_output);
        if (!output_file) {
            std::filesystem::remove(temp_output);
            return makeError(GraphCutError::PYTHON_EXECUTION_FAILED, "Failed to read solver output");
        }
        
        std::string json_response;
        std::string line;
        while (std::getline(output_file, line)) {
            json_response += line;
        }
        
        // 清理输出文件
        std::filesystem::remove(temp_output);
        
        // 解析响应
        return deserializeFromJson(json_response);
        
    } catch (const std::exception& e) {
        return makeError(GraphCutError::PYTHON_EXECUTION_FAILED, 
                        "Exception in solveWithFileIO: " + std::string(e.what()));
    }
}

GraphCutResult ThreadSafePyMaxflowSolver::solveWithPersistentProcess() {
    if (!persistent_process_ || !persistent_process_->is_running) {
        return makeError(GraphCutError::PYTHON_EXECUTION_FAILED, "Persistent process not available");
    }
    
    std::lock_guard<std::mutex> lock(persistent_process_->process_mutex);
    
    try {
        // 创建JSON请求
        std::string json_request = serializeToJson();
        json_request += "\n";  // 添加换行符作为结束标记
        
        // 发送请求
        ssize_t bytes_written = write(persistent_process_->input_pipe[1], json_request.c_str(), json_request.length());
        if (bytes_written != static_cast<ssize_t>(json_request.length())) {
            return makeError(GraphCutError::PYTHON_EXECUTION_FAILED, "Failed to write to persistent process");
        }
        
        // 读取响应（带超时）
        std::string response;
        char buffer[4096];
        
        auto start_time = std::chrono::steady_clock::now();
        while (true) {
            // 检查超时
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            if (elapsed > config_.timeout_seconds) {
                return makeError(GraphCutError::TIMEOUT_EXCEEDED, "Persistent process timeout");
            }
            
            // 非阻塞读取
            ssize_t bytes_read = read(persistent_process_->output_pipe[0], buffer, sizeof(buffer) - 1);
            if (bytes_read > 0) {
                buffer[bytes_read] = '\0';
                response += buffer;
                
                // 检查是否收到完整响应（以换行符结束）
                if (response.back() == '\n') {
                    response.pop_back();  // 移除换行符
                    break;
                }
            } else if (bytes_read == 0) {
                return makeError(GraphCutError::PYTHON_EXECUTION_FAILED, "Persistent process closed connection");
            } else {
                // 短暂休眠避免忙等待
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        // 解析响应
        return deserializeFromJson(response);
        
    } catch (const std::exception& e) {
        return makeError(GraphCutError::PYTHON_EXECUTION_FAILED, 
                        "Exception in solveWithPersistentProcess: " + std::string(e.what()));
    }
}

GraphCutResult ThreadSafePyMaxflowSolver::solveWithSharedMemory() {
    // 共享内存实现（暂时返回错误，需要更复杂的实现）
    return makeError(GraphCutError::PYTHON_EXECUTION_FAILED, "Shared memory communication not yet implemented");
}

// ============================================================================
// JSON序列化/反序列化
// ============================================================================

std::string ThreadSafePyMaxflowSolver::serializeToJson() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    json j;
    j["command"] = "solve_graph";
    j["num_nodes"] = num_nodes_;
    
    // 序列化终端边
    json terminal_edges = json::array();
    for (const auto& te : terminal_edges_) {
        terminal_edges.push_back({te.node, te.source_capacity, te.sink_capacity});
    }
    j["terminal_edges"] = terminal_edges;
    
    // 序列化普通边
    json edges = json::array();
    for (const auto& e : edges_) {
        edges.push_back({e.from, e.to, e.capacity, e.rev_capacity});
    }
    j["edges"] = edges;
    
    return j.dump();
}

GraphCutResult ThreadSafePyMaxflowSolver::deserializeFromJson(const std::string& json_str) const {
    try {
        json j = json::parse(json_str);
        
        GraphCutResult result;
        
        if (j.contains("success") && j["success"].is_boolean()) {
            if (j["success"]) {
                result.error_code = GraphCutError::SUCCESS;
                
                if (j.contains("flow_value") && j["flow_value"].is_number()) {
                    result.flow_value = j["flow_value"];
                }
                
                if (j.contains("segments") && j["segments"].is_array()) {
                    result.segments = j["segments"].get<std::vector<int>>();
                }
                
                if (j.contains("solve_time") && j["solve_time"].is_number()) {
                    result.performance.algorithm_time = j["solve_time"];
                }
            } else {
                result.error_code = GraphCutError::PYTHON_EXECUTION_FAILED;
                if (j.contains("error") && j["error"].is_string()) {
                    result.error_message = j["error"];
                }
            }
        } else {
            result.error_code = GraphCutError::JSON_PARSING_FAILED;
            result.error_message = "Invalid JSON response format";
        }
        
        return result;
        
    } catch (const json::exception& e) {
        return makeError(GraphCutError::JSON_PARSING_FAILED, 
                        "JSON parsing error: " + std::string(e.what()));
    }
}

// ============================================================================
// 辅助方法
// ============================================================================

bool ThreadSafePyMaxflowSolver::isSourceSide(int node) const {
    if (!validateNodeIndex(node)) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(graph_mutex_);
    return source_side_[node];
}

void ThreadSafePyMaxflowSolver::clear() {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    terminal_edges_.clear();
    edges_.clear();
    std::fill(source_side_.begin(), source_side_.end(), false);
}

ThreadSafePyMaxflowSolver::GraphStats ThreadSafePyMaxflowSolver::getGraphStats() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    GraphStats stats;
    stats.num_nodes = num_nodes_;
    stats.num_terminal_edges = static_cast<int>(terminal_edges_.size());
    stats.num_edges = static_cast<int>(edges_.size());
    stats.memory_usage_bytes = sizeof(*this) + 
                              terminal_edges_.capacity() * sizeof(TerminalEdge) +
                              edges_.capacity() * sizeof(core::Edge) +
                              source_side_.capacity() * sizeof(bool);
    stats.is_valid = validateGraphStructure();
    
    return stats;
}

std::pair<bool, std::string> ThreadSafePyMaxflowSolver::validateGraph() const {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    
    // 检查是否有节点
    if (num_nodes_ <= 0) {
        return {false, "Graph has no nodes"};
    }
    
    // 检查是否有终端边
    if (terminal_edges_.empty()) {
        return {false, "Graph has no terminal edges (source/sink connections)"};
    }
    
    // 检查节点连通性
    std::vector<bool> connected(num_nodes_, false);
    
    // 标记有终端边的节点
    for (const auto& te : terminal_edges_) {
        if (te.node >= 0 && te.node < num_nodes_) {
            connected[te.node] = true;
        }
    }
    
    // 标记有普通边的节点
    for (const auto& e : edges_) {
        if (e.from >= 0 && e.from < num_nodes_) {
            connected[e.from] = true;
        }
        if (e.to >= 0 && e.to < num_nodes_) {
            connected[e.to] = true;
        }
    }
    
    // 检查是否有孤立节点
    int isolated_nodes = 0;
    for (int i = 0; i < num_nodes_; ++i) {
        if (!connected[i]) {
            isolated_nodes++;
        }
    }
    
    if (isolated_nodes > 0) {
        return {false, "Graph has " + std::to_string(isolated_nodes) + " isolated nodes"};
    }
    
    return {true, "Graph structure is valid"};
}

bool ThreadSafePyMaxflowSolver::validateNodeIndex(int node) const {
    return node >= 0 && node < num_nodes_;
}

bool ThreadSafePyMaxflowSolver::validateGraphStructure() const {
    return validateGraph().first;
}

std::string ThreadSafePyMaxflowSolver::findPythonScript() const {
    std::vector<std::string> search_paths = {
        "./pymaxflow_solver.py",
        "../pymaxflow_solver.py",
        "../../pymaxflow_solver.py",
        "./recon/pymaxflow_solver.py",
        "./scripts/pymaxflow_solver.py",
        "/usr/local/bin/pymaxflow_solver.py",
        "/opt/recon/pymaxflow_solver.py"
    };
    
    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            return std::filesystem::absolute(path);
        }
    }
    
    logError("Python solver script not found in any search path");
    return "./pymaxflow_solver.py";  // 返回默认路径
}

GraphCutResult ThreadSafePyMaxflowSolver::makeError(GraphCutError error_code, const std::string& details) const {
    GraphCutResult result;
    result.error_code = error_code;
    result.error_message = result.getErrorDescription();
    if (!details.empty()) {
        result.error_message += ": " + details;
    }
    return result;
}

void ThreadSafePyMaxflowSolver::logError(const std::string& message) const {
    if (config_.enable_detailed_logging) {
        std::cerr << "[ThreadSafePyMaxflowSolver ERROR] " << message << std::endl;
    }
}

void ThreadSafePyMaxflowSolver::logInfo(const std::string& message) const {
    if (config_.enable_detailed_logging) {
        std::cout << "[ThreadSafePyMaxflowSolver INFO] " << message << std::endl;
    }
}

core::PerformanceStats ThreadSafePyMaxflowSolver::getPerformanceStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return cumulative_stats_;
}

void ThreadSafePyMaxflowSolver::updatePerformanceStats(const core::PerformanceStats& stats) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    cumulative_stats_.total_time += stats.total_time;
    cumulative_stats_.algorithm_time += stats.algorithm_time;
    cumulative_stats_.input_size += stats.input_size;
    cumulative_stats_.output_size += stats.output_size;
    
    // 计算平均值
    int count = solve_count_.load();
    if (count > 0) {
        cumulative_stats_.throughput = cumulative_stats_.input_size / cumulative_stats_.total_time;
    }
}

// ============================================================================
// 持久进程管理
// ============================================================================

bool ThreadSafePyMaxflowSolver::initializePersistentProcess() {
    persistent_process_ = std::make_unique<PersistentProcess>();
    
    // 创建管道
    if (pipe(persistent_process_->input_pipe) != 0 || pipe(persistent_process_->output_pipe) != 0) {
        logError("Failed to create pipes for persistent process");
        return false;
    }
    
    // 创建子进程
    persistent_process_->pid = fork();
    if (persistent_process_->pid == -1) {
        logError("Failed to fork persistent process");
        return false;
    }
    
    if (persistent_process_->pid == 0) {
        // 子进程：重定向标准输入输出到管道
        dup2(persistent_process_->input_pipe[0], STDIN_FILENO);
        dup2(persistent_process_->output_pipe[1], STDOUT_FILENO);
        
        // 关闭不需要的管道端
        close(persistent_process_->input_pipe[1]);
        close(persistent_process_->output_pipe[0]);
        
        // 执行Python脚本
        execl("/usr/bin/python3", "python3", config_.python_script_path.c_str(), "--persistent", nullptr);
        
        // 如果execl失败
        std::cerr << "Failed to execute Python script: " << config_.python_script_path << std::endl;
        exit(1);
    } else {
        // 父进程：关闭不需要的管道端
        close(persistent_process_->input_pipe[0]);
        close(persistent_process_->output_pipe[1]);
        
        persistent_process_->is_running = true;
        logInfo("Persistent process initialized with PID: " + std::to_string(persistent_process_->pid));
        return true;
    }
}

void ThreadSafePyMaxflowSolver::cleanupPersistentProcess() {
    if (persistent_process_) {
        persistent_process_->cleanup();
        persistent_process_.reset();
    }
}

// ============================================================================
// 工厂方法
// ============================================================================

std::unique_ptr<ThreadSafePyMaxflowSolver> GraphCutSolverFactory::create(
    int num_nodes, const GraphCutConfig& config) {
    
    // 检查依赖
    auto [deps_ok, deps_error] = checkDependencies();
    if (!deps_ok) {
        throw std::runtime_error("Dependencies check failed: " + deps_error);
    }
    
    return std::make_unique<ThreadSafePyMaxflowSolver>(num_nodes, config);
}

std::pair<bool, std::string> GraphCutSolverFactory::checkDependencies() {
    // 检查Python3
    if (std::system("which python3 > /dev/null 2>&1") != 0) {
        return {false, "Python3 not found. Please install: sudo apt-get install python3"};
    }
    
    // 检查PyMaxflow
    if (std::system("python3 -c 'import maxflow' > /dev/null 2>&1") != 0) {
        return {false, "PyMaxflow not found. Please install: pip3 install PyMaxflow"};
    }
    
    return {true, "All dependencies satisfied"};
}

GraphCutConfig GraphCutSolverFactory::getRecommendedConfig(int num_nodes, bool performance_priority) {
    GraphCutConfig config;
    
    if (performance_priority) {
        // 性能优先配置
        config.use_persistent_process = (num_nodes > 1000);
        config.use_pipe_communication = true;
        config.timeout_seconds = 600;  // 10分钟
        config.validate_graph = false;  // 跳过验证以提升性能
    } else {
        // 稳定性优先配置
        config.use_persistent_process = false;
        config.use_pipe_communication = false;
        config.timeout_seconds = 300;  // 5分钟
        config.validate_graph = true;
        config.enable_detailed_logging = true;
    }
    
    return config;
}

} // namespace algorithms
} // namespace recon


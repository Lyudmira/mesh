#include "pymaxflow_solver.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <filesystem>
#include <chrono>
#include <unistd.h>  // for getpid()

// 简单的JSON处理（避免引入额外依赖）
namespace {

std::string escapeJsonString(const std::string& str) {
    std::string result;
    for (char c : str) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c; break;
        }
    }
    return result;
}

std::string vectorToJsonArray(const std::vector<int>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << ",";
        oss << vec[i];
    }
    oss << "]";
    return oss.str();
}

// 简单的JSON解析（只解析我们需要的字段）
class SimpleJsonParser {
private:
    std::string json_;
    size_t pos_;
    
public:
    explicit SimpleJsonParser(const std::string& json) : json_(json), pos_(0) {}
    
    bool getBool(const std::string& key) {
        std::string pattern = "\"" + key + "\":";
        size_t start = json_.find(pattern);
        if (start == std::string::npos) return false;
        
        start += pattern.length();
        while (start < json_.length() && std::isspace(json_[start])) start++;
        
        return json_.substr(start, 4) == "true";
    }
    
    float getFloat(const std::string& key) {
        std::string pattern = "\"" + key + "\":";
        size_t start = json_.find(pattern);
        if (start == std::string::npos) return 0.0f;
        
        start += pattern.length();
        while (start < json_.length() && std::isspace(json_[start])) start++;
        
        size_t end = start;
        while (end < json_.length() && 
               (std::isdigit(json_[end]) || json_[end] == '.' || json_[end] == '-' || json_[end] == 'e' || json_[end] == 'E')) {
            end++;
        }
        
        return std::stof(json_.substr(start, end - start));
    }
    
    double getDouble(const std::string& key) {
        return static_cast<double>(getFloat(key));
    }
    
    std::string getString(const std::string& key) {
        std::string pattern = "\"" + key + "\":\"";
        size_t start = json_.find(pattern);
        if (start == std::string::npos) return "";
        
        start += pattern.length();
        size_t end = json_.find("\"", start);
        if (end == std::string::npos) return "";
        
        return json_.substr(start, end - start);
    }
    
    std::vector<int> getIntArray(const std::string& key) {
        std::string pattern = "\"" + key + "\":";
        size_t start = json_.find(pattern);
        if (start == std::string::npos) return {};
        
        start += pattern.length();
        
        // 跳过空白字符
        while (start < json_.length() && std::isspace(json_[start])) start++;
        
        // 检查是否是数组开始
        if (start >= json_.length() || json_[start] != '[') return {};
        start++; // 跳过 '['
        
        size_t end = json_.find("]", start);
        if (end == std::string::npos) return {};
        
        std::string array_content = json_.substr(start, end - start);
        std::vector<int> result;
        
        // 简单的数组解析
        std::string current_number;
        for (char c : array_content) {
            if (std::isdigit(c) || c == '-') {
                current_number += c;
            } else if (c == ',' || c == ' ') {
                if (!current_number.empty()) {
                    result.push_back(std::stoi(current_number));
                    current_number.clear();
                }
            }
        }
        // 处理最后一个数字
        if (!current_number.empty()) {
            result.push_back(std::stoi(current_number));
        }
        
        return result;
    }
};

} // anonymous namespace

namespace recon {

PyMaxflowSolver::PyMaxflowSolver(int num_nodes, const std::string& python_script_path)
    : num_nodes_(num_nodes), python_script_path_(python_script_path), source_side_(num_nodes, false) {
    
    if (python_script_path_.empty()) {
        python_script_path_ = findPythonScript();
    }
}

void PyMaxflowSolver::addTerminalEdge(int node, float source_cap, float sink_cap) {
    if (node >= 0 && node < num_nodes_) {
        terminal_edges_.emplace_back(node, source_cap, sink_cap);
    }
}

void PyMaxflowSolver::addEdge(int from, int to, float capacity, float rev_capacity) {
    if (from >= 0 && from < num_nodes_ && to >= 0 && to < num_nodes_ && from != to) {
        edges_.emplace_back(from, to, capacity, rev_capacity);
    }
}

PyMaxflowSolver::SolveResult PyMaxflowSolver::solve() {
    SolveResult result;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // 创建JSON请求
        std::string json_request = createJsonRequest();
        
        // 调用Python求解器
        std::string json_response = callPythonSolver(json_request);
        
        // 解析响应
        result = parseJsonResponse(json_response);
        
        // 更新source_side_数组
        if (result.success && result.segments.size() == static_cast<size_t>(num_nodes_)) {
            for (int i = 0; i < num_nodes_; ++i) {
                source_side_[i] = (result.segments[i] == 0);
            }
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("Exception in solve(): ") + e.what();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    // 如果Python求解器没有返回时间，使用总时间
    if (result.solve_time <= 0) {
        result.solve_time = total_time;
    }
    
    return result;
}

bool PyMaxflowSolver::isSourceSide(int node) const {
    if (node >= 0 && node < num_nodes_) {
        return source_side_[node];
    }
    return false;
}

void PyMaxflowSolver::clear() {
    terminal_edges_.clear();
    edges_.clear();
    std::fill(source_side_.begin(), source_side_.end(), false);
}

PyMaxflowSolver::Statistics PyMaxflowSolver::getStatistics(const SolveResult& result) const {
    Statistics stats;
    stats.num_nodes = num_nodes_;
    stats.num_terminal_edges = static_cast<int>(terminal_edges_.size());
    stats.num_edges = static_cast<int>(edges_.size());
    stats.flow_value = result.flow_value;
    stats.solve_time = result.solve_time;
    
    stats.source_nodes = 0;
    stats.sink_nodes = 0;
    for (int segment : result.segments) {
        if (segment == 0) {
            stats.source_nodes++;
        } else {
            stats.sink_nodes++;
        }
    }
    
    return stats;
}

std::string PyMaxflowSolver::createJsonRequest() const {
    std::ostringstream oss;
    
    oss << "{";
    oss << "\"command\":\"solve_graph\",";
    oss << "\"num_nodes\":" << num_nodes_ << ",";
    
    // 添加源汇边
    oss << "\"terminal_edges\":[";
    for (size_t i = 0; i < terminal_edges_.size(); ++i) {
        if (i > 0) oss << ",";
        const auto& te = terminal_edges_[i];
        oss << "[" << te.node << "," << te.source_capacity << "," << te.sink_capacity << "]";
    }
    oss << "],";
    
    // 添加普通边
    oss << "\"edges\":[";
    for (size_t i = 0; i < edges_.size(); ++i) {
        if (i > 0) oss << ",";
        const auto& e = edges_[i];
        oss << "[" << e.from << "," << e.to << "," << e.capacity << "," << e.rev_capacity << "]";
    }
    oss << "]";
    
    oss << "}";
    
    return oss.str();
}

PyMaxflowSolver::SolveResult PyMaxflowSolver::parseJsonResponse(const std::string& json_response) const {
    SolveResult result;
    
    try {
        SimpleJsonParser parser(json_response);
        
        result.success = parser.getBool("success");
        
        if (result.success) {
            result.flow_value = parser.getFloat("flow_value");
            result.segments = parser.getIntArray("segments");
            result.solve_time = parser.getDouble("solve_time");
        } else {
            result.error_message = parser.getString("error");
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("JSON parsing error: ") + e.what();
    }
    
    return result;
}

std::string PyMaxflowSolver::callPythonSolver(const std::string& json_request) const {
    // 创建临时文件来传递数据
    std::string temp_input = "/tmp/pymaxflow_input_" + std::to_string(getpid()) + ".json";
    std::string temp_output = "/tmp/pymaxflow_output_" + std::to_string(getpid()) + ".json";
    
    try {
        // 写入请求数据
        {
            std::ofstream input_file(temp_input);
            if (!input_file) {
                throw std::runtime_error("Failed to create temporary input file");
            }
            input_file << json_request;
        }
        
        // 构建命令
        std::string command = "python3 \"" + python_script_path_ + "\" \"" + temp_input + "\" > \"" + temp_output + "\" 2>&1";
        
        // 执行命令
        int exit_code = std::system(command.c_str());
        
        if (exit_code != 0) {
            // 读取错误信息
            std::ifstream error_file(temp_output);
            std::string error_msg;
            if (error_file) {
                std::getline(error_file, error_msg);
            }
            throw std::runtime_error("Python solver failed with exit code " + std::to_string(exit_code) + ": " + error_msg);
        }
        
        // 读取响应
        std::ifstream output_file(temp_output);
        if (!output_file) {
            throw std::runtime_error("Failed to read solver output");
        }
        
        std::string response;
        std::string line;
        while (std::getline(output_file, line)) {
            response += line;
        }
        
        // 清理临时文件
        std::filesystem::remove(temp_input);
        std::filesystem::remove(temp_output);
        
        return response;
        
    } catch (...) {
        // 确保清理临时文件
        std::filesystem::remove(temp_input);
        std::filesystem::remove(temp_output);
        throw;
    }
}

std::string PyMaxflowSolver::findPythonScript() const {
    // 搜索可能的脚本位置
    std::vector<std::string> search_paths = {
        "./pymaxflow_solver.py",
        "../pymaxflow_solver.py",
        "../../pymaxflow_solver.py",
        "./recon/pymaxflow_solver.py",
        "./scripts/pymaxflow_solver.py"
    };
    
    for (const auto& path : search_paths) {
        if (std::filesystem::exists(path)) {
            return std::filesystem::absolute(path);
        }
    }
    
    // 如果找不到，返回默认路径
    return "./pymaxflow_solver.py";
}

} // namespace recon


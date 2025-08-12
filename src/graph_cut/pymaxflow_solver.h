#pragma once

#include <vector>
#include <string>
#include <memory>

namespace recon {

/**
 * PyMaxflow求解器的C++接口
 * 通过调用Python脚本来使用高性能的PyMaxflow库
 */
class PyMaxflowSolver {
public:
    struct Edge {
        int from;
        int to;
        float capacity;
        float rev_capacity;
        
        Edge(int f, int t, float cap, float rev_cap = -1.0f) 
            : from(f), to(t), capacity(cap), rev_capacity(rev_cap >= 0 ? rev_cap : cap) {}
    };
    
    struct TerminalEdge {
        int node;
        float source_capacity;
        float sink_capacity;
        
        TerminalEdge(int n, float src_cap, float sink_cap)
            : node(n), source_capacity(src_cap), sink_capacity(sink_cap) {}
    };
    
    struct SolveResult {
        bool success;
        float flow_value;
        std::vector<int> segments;  // 0 = source side, 1 = sink side
        double solve_time;
        std::string error_message;
        
        SolveResult() : success(false), flow_value(0.0f), solve_time(0.0) {}
    };

private:
    int num_nodes_;
    std::vector<TerminalEdge> terminal_edges_;
    std::vector<Edge> edges_;
    std::string python_script_path_;
    std::vector<bool> source_side_;
    
public:
    /**
     * 构造函数
     * @param num_nodes 节点数量
     * @param python_script_path Python求解器脚本路径（可选）
     */
    explicit PyMaxflowSolver(int num_nodes, const std::string& python_script_path = "");
    
    /**
     * 析构函数
     */
    ~PyMaxflowSolver() = default;
    
    /**
     * 添加源汇边（数据项）
     * @param node 节点ID
     * @param source_cap 到源的容量
     * @param sink_cap 到汇的容量
     */
    void addTerminalEdge(int node, float source_cap, float sink_cap);
    
    /**
     * 添加普通边（平滑项）
     * @param from 起始节点
     * @param to 目标节点
     * @param capacity 容量
     * @param rev_capacity 反向容量（默认等于正向容量）
     */
    void addEdge(int from, int to, float capacity, float rev_capacity = -1.0f);
    
    /**
     * 求解最大流
     * @return 求解结果
     */
    SolveResult solve();
    
    /**
     * 检查节点是否在源侧
     * @param node 节点ID
     * @return true表示在源侧，false表示在汇侧
     */
    bool isSourceSide(int node) const;
    
    /**
     * 获取节点数量
     */
    int getNumNodes() const { return num_nodes_; }
    
    /**
     * 清空所有边
     */
    void clear();
    
    /**
     * 设置Python脚本路径
     */
    void setPythonScriptPath(const std::string& path) { python_script_path_ = path; }
    
    /**
     * 获取统计信息
     */
    struct Statistics {
        int num_nodes;
        int num_terminal_edges;
        int num_edges;
        float flow_value;
        double solve_time;
        int source_nodes;
        int sink_nodes;
    };
    
    Statistics getStatistics(const SolveResult& result) const;

private:
    /**
     * 创建JSON请求数据
     */
    std::string createJsonRequest() const;
    
    /**
     * 解析JSON响应
     */
    SolveResult parseJsonResponse(const std::string& json_response) const;
    
    /**
     * 调用Python求解器
     */
    std::string callPythonSolver(const std::string& json_request) const;
    
    /**
     * 查找Python脚本路径
     */
    std::string findPythonScript() const;
};

} // namespace recon


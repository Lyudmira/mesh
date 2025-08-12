/**
 * Boykov-Kolmogorov最大流求解器
 * 基于gerddie/maxflow库的高性能实现
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#ifndef BOYKOV_KOLMOGOROV_SOLVER_H
#define BOYKOV_KOLMOGOROV_SOLVER_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

// 前向声明避免头文件依赖
namespace maxflow {
    template<typename captype, typename tcaptype, typename flowtype> 
    class Graph;
}

namespace recon {

/**
 * 最大流求解器接口
 */
class MaxFlowSolver {
public:
    virtual ~MaxFlowSolver() = default;
    
    // 基本接口
    virtual int addNode() = 0;
    virtual void addEdge(int node1, int node2, float capacity1to2, float capacity2to1) = 0;
    virtual void setTerminalWeights(int node, float source_weight, float sink_weight) = 0;
    
    // 求解
    virtual float solve() = 0;
    virtual bool getNodeLabel(int node) = 0;  // true = source side, false = sink side
    
    // 信息
    virtual std::string getName() const = 0;
    virtual void reset() = 0;
};

/**
 * Boykov-Kolmogorov算法实现
 * 使用gerddie/maxflow库作为底层实现
 */
class BoykovKolmogorovSolver : public MaxFlowSolver {
public:
    BoykovKolmogorovSolver();
    ~BoykovKolmogorovSolver() override;
    
    // MaxFlowSolver接口实现
    int addNode() override;
    void addEdge(int node1, int node2, float capacity1to2, float capacity2to1) override;
    void setTerminalWeights(int node, float source_weight, float sink_weight) override;
    
    float solve() override;
    bool getNodeLabel(int node) override;
    
    std::string getName() const override { return "Boykov-Kolmogorov"; }
    void reset() override;
    
    // 高级功能
    void enableTreeReuse(bool enable) { tree_reuse_enabled_ = enable; }
    bool isTreeReuseEnabled() const { return tree_reuse_enabled_; }
    
    // 统计信息
    int getNodeCount() const { return node_count_; }
    int getEdgeCount() const { return edge_count_; }
    
private:
    // 使用float类型的maxflow图
    using GraphType = maxflow::Graph<float, float, float>;
    
    std::unique_ptr<GraphType> graph_;
    std::vector<int> node_mapping_;  // 外部节点ID到内部节点ID的映射
    int node_count_;
    int edge_count_;
    bool tree_reuse_enabled_;
    bool solved_;
    
    void ensureGraphExists();
};

/**
 * 求解器工厂
 */
class MaxFlowSolverFactory {
public:
    enum class SolverType {
        BOYKOV_KOLMOGOROV,
        PUSH_RELABEL,
        EDMONDS_KARP
    };
    
    static std::unique_ptr<MaxFlowSolver> createSolver(SolverType type);
    static std::unique_ptr<MaxFlowSolver> createOptimalSolver();
    static std::vector<SolverType> getAvailableSolvers();
    static std::string getSolverName(SolverType type);
};

} // namespace recon

#endif // BOYKOV_KOLMOGOROV_SOLVER_H


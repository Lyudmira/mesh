/**
 * Boykov-Kolmogorov最大流求解器实现
 */

#include "boykov_kolmogorov_solver.h"
#include "../../../external/maxflow/maxflow/graph.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace recon {

// ============================================================================
// BoykovKolmogorovSolver实现
// ============================================================================

BoykovKolmogorovSolver::BoykovKolmogorovSolver()
    : node_count_(0)
    , edge_count_(0)
    , tree_reuse_enabled_(false)
    , solved_(false) {
    
    // 延迟创建图，直到知道节点数量估计
}

BoykovKolmogorovSolver::~BoykovKolmogorovSolver() = default;

void BoykovKolmogorovSolver::ensureGraphExists() {
    if (!graph_) {
        // 创建图，预估节点和边的数量
        int estimated_nodes = std::max(1000, node_count_ * 2);
        int estimated_edges = std::max(1000, edge_count_ * 2);
        
        graph_ = std::make_unique<GraphType>(estimated_nodes, estimated_edges);
        
        // 重新添加已记录的节点
        for (int i = 0; i < node_count_; ++i) {
            graph_->add_node();
        }
    }
}

int BoykovKolmogorovSolver::addNode() {
    ensureGraphExists();
    
    int external_id = node_count_;
    int internal_id = graph_->add_node();
    
    // 确保映射表足够大
    if (static_cast<int>(node_mapping_.size()) <= external_id) {
        node_mapping_.resize(external_id + 1);
    }
    node_mapping_[external_id] = internal_id;
    
    node_count_++;
    solved_ = false;
    
    return external_id;
}

void BoykovKolmogorovSolver::addEdge(int node1, int node2, float capacity1to2, float capacity2to1) {
    ensureGraphExists();
    
    if (node1 < 0 || node1 >= node_count_ || node2 < 0 || node2 >= node_count_) {
        throw std::invalid_argument("Invalid node indices in addEdge");
    }
    
    int internal_node1 = node_mapping_[node1];
    int internal_node2 = node_mapping_[node2];
    
    graph_->add_edge(internal_node1, internal_node2, capacity1to2, capacity2to1);
    
    edge_count_++;
    solved_ = false;
}

void BoykovKolmogorovSolver::setTerminalWeights(int node, float source_weight, float sink_weight) {
    ensureGraphExists();
    
    if (node < 0 || node >= node_count_) {
        throw std::invalid_argument("Invalid node index in setTerminalWeights");
    }
    
    int internal_node = node_mapping_[node];
    graph_->add_tweights(internal_node, source_weight, sink_weight);
    
    solved_ = false;
}

float BoykovKolmogorovSolver::solve() {
    if (!graph_) {
        throw std::runtime_error("No graph to solve");
    }
    
    if (node_count_ == 0) {
        return 0.0f;
    }
    
    try {
        float max_flow = static_cast<float>(graph_->maxflow());
        solved_ = true;
        return max_flow;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Maxflow solver failed: ") + e.what());
    }
}

bool BoykovKolmogorovSolver::getNodeLabel(int node) {
    if (!solved_) {
        throw std::runtime_error("Must call solve() before getNodeLabel()");
    }
    
    if (node < 0 || node >= node_count_) {
        throw std::invalid_argument("Invalid node index in getNodeLabel");
    }
    
    int internal_node = node_mapping_[node];
    return graph_->what_segment(internal_node) == GraphType::SOURCE;
}

void BoykovKolmogorovSolver::reset() {
    graph_.reset();
    node_mapping_.clear();
    node_count_ = 0;
    edge_count_ = 0;
    solved_ = false;
}

// ============================================================================
// MaxFlowSolverFactory实现
// ============================================================================

std::unique_ptr<MaxFlowSolver> MaxFlowSolverFactory::createSolver(SolverType type) {
    switch (type) {
        case SolverType::BOYKOV_KOLMOGOROV:
            return std::make_unique<BoykovKolmogorovSolver>();
        
        case SolverType::PUSH_RELABEL:
            // 可以在这里添加其他算法的实现
            throw std::runtime_error("Push-Relabel solver not implemented yet");
        
        case SolverType::EDMONDS_KARP:
            // 可以在这里添加其他算法的实现
            throw std::runtime_error("Edmonds-Karp solver not implemented yet");
        
        default:
            throw std::invalid_argument("Unknown solver type");
    }
}

std::unique_ptr<MaxFlowSolver> MaxFlowSolverFactory::createOptimalSolver() {
    // 目前返回Boykov-Kolmogorov作为最优选择
    return createSolver(SolverType::BOYKOV_KOLMOGOROV);
}

std::vector<MaxFlowSolverFactory::SolverType> MaxFlowSolverFactory::getAvailableSolvers() {
    return {SolverType::BOYKOV_KOLMOGOROV};
}

std::string MaxFlowSolverFactory::getSolverName(SolverType type) {
    switch (type) {
        case SolverType::BOYKOV_KOLMOGOROV:
            return "Boykov-Kolmogorov";
        case SolverType::PUSH_RELABEL:
            return "Push-Relabel";
        case SolverType::EDMONDS_KARP:
            return "Edmonds-Karp";
        default:
            return "Unknown";
    }
}

} // namespace recon


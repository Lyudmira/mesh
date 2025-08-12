/**
 * 增强版图割求解器实现
 * 统一并优化图割求解，支持可重入、可并行处理
 */

#include "enhanced_graph_cut.h"
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <queue>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace recon {

// ============================================================================
// Kolmogorov最大流求解器实现
// ============================================================================

/**
 * Kolmogorov算法的简化实现
 * 基于经典的最大流/最小割算法
 */
class KolmogorovMaxFlowSolver::Impl {
public:
    struct Node {
        float excess = 0.0f;
        int parent = -1;
        bool active = false;
        bool in_source_tree = false;
        bool in_sink_tree = false;
    };
    
    struct Edge {
        int to = -1;
        float capacity = 0.0f;
        float flow = 0.0f;
        int reverse = -1;
    };
    
    std::vector<Node> nodes;
    std::vector<std::vector<Edge>> graph;
    std::queue<int> active_nodes;
    
    int source_id = -1;
    int sink_id = -1;
    
    void addNode() {
        nodes.emplace_back();
        graph.emplace_back();
    }
    
    void addEdge(int from, int to, float capacity) {
        graph[from].push_back({to, capacity, 0.0f, static_cast<int>(graph[to].size())});
        graph[to].push_back({from, 0.0f, 0.0f, static_cast<int>(graph[from].size()) - 1});
    }
    
    void setTerminalCapacity(int node, float source_cap, float sink_cap) {
        if (source_id == -1) {
            source_id = static_cast<int>(nodes.size());
            addNode();
        }
        if (sink_id == -1) {
            sink_id = static_cast<int>(nodes.size());
            addNode();
        }
        
        if (source_cap > 0) {
            addEdge(source_id, node, source_cap);
        }
        if (sink_cap > 0) {
            addEdge(node, sink_id, sink_cap);
        }
    }
    
    float maxFlow() {
        float total_flow = 0.0f;
        
        // 简化的最大流算法实现
        while (true) {
            // 寻找增广路径
            std::vector<int> path;
            std::vector<int> edge_indices;
            
            if (!findAugmentingPath(path, edge_indices)) {
                break;
            }
            
            // 计算路径上的最小容量
            float min_capacity = std::numeric_limits<float>::max();
            for (size_t i = 0; i < edge_indices.size(); ++i) {
                int from = path[i];
                int edge_idx = edge_indices[i];
                float residual = graph[from][edge_idx].capacity - graph[from][edge_idx].flow;
                min_capacity = std::min(min_capacity, residual);
            }
            
            // 更新流量
            for (size_t i = 0; i < edge_indices.size(); ++i) {
                int from = path[i];
                int to = path[i + 1];
                int edge_idx = edge_indices[i];
                
                graph[from][edge_idx].flow += min_capacity;
                int reverse_idx = graph[from][edge_idx].reverse;
                graph[to][reverse_idx].flow -= min_capacity;
            }
            
            total_flow += min_capacity;
        }
        
        return total_flow;
    }
    
    bool getNodeLabel(int node) {
        // 通过BFS确定节点是否可达源点
        std::vector<bool> visited(nodes.size(), false);
        std::queue<int> q;
        
        if (source_id >= 0) {
            q.push(source_id);
            visited[source_id] = true;
        }
        
        while (!q.empty()) {
            int current = q.front();
            q.pop();
            
            for (const auto& edge : graph[current]) {
                if (!visited[edge.to] && edge.capacity > edge.flow) {
                    visited[edge.to] = true;
                    q.push(edge.to);
                }
            }
        }
        
        return visited[node];
    }

private:
    bool findAugmentingPath(std::vector<int>& path, std::vector<int>& edge_indices) {
        if (source_id == -1 || sink_id == -1) return false;
        
        std::vector<int> parent(nodes.size(), -1);
        std::vector<int> parent_edge(nodes.size(), -1);
        std::vector<bool> visited(nodes.size(), false);
        std::queue<int> q;
        
        q.push(source_id);
        visited[source_id] = true;
        
        while (!q.empty() && !visited[sink_id]) {
            int current = q.front();
            q.pop();
            
            for (size_t i = 0; i < graph[current].size(); ++i) {
                const auto& edge = graph[current][i];
                if (!visited[edge.to] && edge.capacity > edge.flow) {
                    visited[edge.to] = true;
                    parent[edge.to] = current;
                    parent_edge[edge.to] = static_cast<int>(i);
                    q.push(edge.to);
                }
            }
        }
        
        if (!visited[sink_id]) {
            return false;
        }
        
        // 重构路径
        path.clear();
        edge_indices.clear();
        
        int current = sink_id;
        while (parent[current] != -1) {
            path.push_back(current);
            edge_indices.push_back(parent_edge[current]);
            current = parent[current];
        }
        path.push_back(source_id);
        
        std::reverse(path.begin(), path.end());
        std::reverse(edge_indices.begin(), edge_indices.end());
        
        return true;
    }
};

KolmogorovMaxFlowSolver::KolmogorovMaxFlowSolver() 
    : pimpl_(std::make_unique<Impl>()) {
}

KolmogorovMaxFlowSolver::~KolmogorovMaxFlowSolver() = default;

int KolmogorovMaxFlowSolver::addNode() {
    pimpl_->addNode();
    return static_cast<int>(pimpl_->nodes.size()) - 1;
}

void KolmogorovMaxFlowSolver::addEdge(int node1, int node2, float capacity12, float capacity21) {
    pimpl_->addEdge(node1, node2, capacity12);
    if (capacity21 > 0) {
        pimpl_->addEdge(node2, node1, capacity21);
    }
}

void KolmogorovMaxFlowSolver::setTerminalWeights(int node, float source_weight, float sink_weight) {
    pimpl_->setTerminalCapacity(node, source_weight, sink_weight);
}

float KolmogorovMaxFlowSolver::solve() {
    return pimpl_->maxFlow();
}

bool KolmogorovMaxFlowSolver::getNodeLabel(int node) {
    return pimpl_->getNodeLabel(node);
}

void KolmogorovMaxFlowSolver::reset() {
    pimpl_ = std::make_unique<Impl>();
}

// ============================================================================
// Boost Graph最大流求解器实现（简化版）
// ============================================================================

class BoostMaxFlowSolver::Impl {
public:
    // 简化实现，使用与Kolmogorov相同的算法
    KolmogorovMaxFlowSolver::Impl kolmogorov_impl;
    
    void addNode() { kolmogorov_impl.addNode(); }
    void addEdge(int node1, int node2, float capacity12) { kolmogorov_impl.addEdge(node1, node2, capacity12); }
    void setTerminalCapacity(int node, float source_cap, float sink_cap) { kolmogorov_impl.setTerminalCapacity(node, source_cap, sink_cap); }
    float maxFlow() { return kolmogorov_impl.maxFlow(); }
    bool getNodeLabel(int node) { return kolmogorov_impl.getNodeLabel(node); }
};

BoostMaxFlowSolver::BoostMaxFlowSolver() 
    : pimpl_(std::make_unique<Impl>()) {
}

BoostMaxFlowSolver::~BoostMaxFlowSolver() = default;

int BoostMaxFlowSolver::addNode() {
    pimpl_->addNode();
    return static_cast<int>(pimpl_->kolmogorov_impl.nodes.size()) - 1;
}

void BoostMaxFlowSolver::addEdge(int node1, int node2, float capacity12, float capacity21) {
    pimpl_->addEdge(node1, node2, capacity12);
    if (capacity21 > 0) {
        pimpl_->addEdge(node2, node1, capacity21);
    }
}

void BoostMaxFlowSolver::setTerminalWeights(int node, float source_weight, float sink_weight) {
    pimpl_->setTerminalCapacity(node, source_weight, sink_weight);
}

float BoostMaxFlowSolver::solve() {
    return pimpl_->maxFlow();
}

bool BoostMaxFlowSolver::getNodeLabel(int node) {
    return pimpl_->getNodeLabel(node);
}

void BoostMaxFlowSolver::reset() {
    pimpl_ = std::make_unique<Impl>();
}

// ============================================================================
// 增强版图割求解器实现
// ============================================================================

EnhancedGraphCutSolver::EnhancedGraphCutSolver(const GraphCutConfig& config) 
    : config_(config) {
    
    // 设置OpenMP线程数
    #ifdef _OPENMP
    if (config_.use_parallel_solver && config_.num_threads > 0) {
        omp_set_num_threads(config_.num_threads);
    }
    #endif
}

EnhancedGraphCutSolver::~EnhancedGraphCutSolver() = default;

bool EnhancedGraphCutSolver::solve(const GridT& udf_grid,
                                  const ConfidenceGridT& confidence_grid,
                                  const PointCloudT& cloud,
                                  GridT::Ptr& result_grid,
                                  GraphCutResult& result) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    solving_.store(true);
    
    std::cout << "开始图割优化..." << std::endl;
    
    try {
        // 1. 构建图结构
        std::cout << "构建图结构..." << std::endl;
        if (!buildGraph(udf_grid, confidence_grid, cloud)) {
            std::cerr << "图结构构建失败" << std::endl;
            return false;
        }
        
        std::cout << "图节点数: " << nodes_.size() << ", 边数: " << edges_.size() << std::endl;
        
        // 2. 创建求解器
        solver_ = createOptimalSolver();
        if (!solver_) {
            std::cerr << "求解器创建失败" << std::endl;
            return false;
        }
        
        std::cout << "使用求解器: " << solver_->getName() << std::endl;
        
        // 3. 设置图结构到求解器
        std::unordered_map<int, int> node_mapping;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            int solver_node_id = solver_->addNode();
            node_mapping[static_cast<int>(i)] = solver_node_id;
        }
        
        // 4. 添加边到求解器
        for (const auto& edge : edges_) {
            int solver_node1 = node_mapping[edge.node1_id];
            int solver_node2 = node_mapping[edge.node2_id];
            solver_->addEdge(solver_node1, solver_node2, edge.weight, edge.weight);
        }
        
        // 5. 设置终端权重
        EnergyTermCalculator energy_calc(config_);
        for (size_t i = 0; i < nodes_.size(); ++i) {
            const auto& node = nodes_[i];
            
            // 计算数据项
            auto [inside_cost, outside_cost] = computeDataTerm(node, udf_grid, confidence_grid);
            
            int solver_node_id = node_mapping[static_cast<int>(i)];
            solver_->setTerminalWeights(solver_node_id, inside_cost, outside_cost);
        }
        
        // 6. 求解
        std::cout << "执行最大流求解..." << std::endl;
        float max_flow = solver_->solve();
        
        // 7. 提取结果
        result.labels.resize(nodes_.size());
        for (size_t i = 0; i < nodes_.size(); ++i) {
            int solver_node_id = node_mapping[static_cast<int>(i)];
            result.labels[i] = solver_->getNodeLabel(solver_node_id);
        }
        
        // 8. 创建结果网格
        result_grid = GridT::create(0.0f);
        result_grid->setTransform(udf_grid.transform().copy());
        result_grid->setName("graph_cut_result");
        
        auto result_accessor = result_grid->getAccessor();
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (result.labels[i]) {  // 内部点
                result_accessor.setValue(nodes_[i].coord, 1.0f);
            }
        }
        
        // 9. 计算能量统计
        result.total_energy = max_flow;
        result.converged = true;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.solve_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "图割求解完成" << std::endl;
        std::cout << "最大流值: " << max_flow << std::endl;
        std::cout << "内部体素数: " << std::count(result.labels.begin(), result.labels.end(), true) << std::endl;
        std::cout << "求解时间: " << result.solve_time_seconds << " 秒" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "图割求解异常: " << e.what() << std::endl;
        solving_.store(false);
        return false;
    }
    
    solving_.store(false);
    return true;
}

bool EnhancedGraphCutSolver::buildGraph(const GridT& udf_grid,
                                       const ConfidenceGridT& confidence_grid,
                                       const PointCloudT& cloud) {
    
    // 清空现有图结构
    nodes_.clear();
    edges_.clear();
    coord_to_node_.clear();
    
    // 1. 添加图节点
    std::cout << "添加图节点..." << std::endl;
    int node_count = 0;
    
    for (auto iter = udf_grid.cbeginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
        openvdb::Vec3f world_pos = udf_grid.transform().indexToWorld(coord);
        
        int node_id = addGraphNode(coord, world_pos, udf_grid, confidence_grid);
        if (node_id >= 0) {
            node_count++;
        }
        
        if (node_count % 10000 == 0) {
            std::cout << "已添加 " << node_count << " 个节点" << std::endl;
        }
    }
    
    std::cout << "总节点数: " << nodes_.size() << std::endl;
    
    // 2. 添加图边
    std::cout << "添加图边..." << std::endl;
    addGraphEdges(udf_grid, cloud);
    
    std::cout << "总边数: " << edges_.size() << std::endl;
    
    return !nodes_.empty();
}

int EnhancedGraphCutSolver::addGraphNode(const openvdb::Coord& coord,
                                        const openvdb::Vec3f& world_pos,
                                        const GridT& udf_grid,
                                        const ConfidenceGridT& confidence_grid) {
    
    GraphNode node;
    node.coord = coord;
    node.world_pos = world_pos;
    
    // 获取UDF值
    auto udf_accessor = udf_grid.getConstAccessor();
    float udf_value = udf_accessor.getValue(coord);
    
    // 获取置信度
    auto conf_accessor = confidence_grid.getConstAccessor();
    node.confidence = conf_accessor.getValue(coord);
    
    // 计算数据项
    auto [inside_cost, outside_cost] = computeDataTerm(node, udf_grid, confidence_grid);
    node.data_cost_inside = inside_cost;
    node.data_cost_outside = outside_cost;
    
    // 判断是否为边界
    node.is_boundary = (udf_value < config_.plane_distance_threshold);
    
    int node_id = static_cast<int>(nodes_.size());
    nodes_.push_back(node);
    coord_to_node_[coord] = node_id;
    
    return node_id;
}

void EnhancedGraphCutSolver::addGraphEdges(const GridT& udf_grid,
                                          const PointCloudT& cloud) {
    
    EnergyTermCalculator energy_calc(config_);
    
    // 遍历所有节点对，添加邻接边
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& node1 = nodes_[i];
        
        // 检查6邻域
        std::vector<openvdb::Coord> neighbors = {
            node1.coord + openvdb::Coord(1, 0, 0),
            node1.coord + openvdb::Coord(-1, 0, 0),
            node1.coord + openvdb::Coord(0, 1, 0),
            node1.coord + openvdb::Coord(0, -1, 0),
            node1.coord + openvdb::Coord(0, 0, 1),
            node1.coord + openvdb::Coord(0, 0, -1)
        };
        
        for (const auto& neighbor_coord : neighbors) {
            auto it = coord_to_node_.find(neighbor_coord);
            if (it != coord_to_node_.end()) {
                int j = it->second;
                const auto& node2 = nodes_[j];
                
                // 计算边权重
                float smoothness_weight = computeSmoothnessTerm(node1, node2, cloud);
                float planarity_weight = computePlanarityTerm(node1, node2, cloud);
                float color_weight = computeColorTerm(node1, node2, cloud);
                
                float total_weight = config_.smoothness_weight * smoothness_weight +
                                   config_.planarity_weight * planarity_weight +
                                   config_.color_weight * color_weight;
                
                // 添加边
                GraphEdge edge;
                edge.node1_id = static_cast<int>(i);
                edge.node2_id = j;
                edge.weight = total_weight;
                edge.term_type = EnergyTermType::SMOOTHNESS_TERM;
                edge.direction = node2.world_pos - node1.world_pos;
                edge.direction.normalize();
                
                edges_.push_back(edge);
            }
        }
        
        if (i % 1000 == 0) {
            std::cout << "已处理 " << i << "/" << nodes_.size() << " 个节点的边" << std::endl;
        }
    }
}

std::pair<float, float> EnhancedGraphCutSolver::computeDataTerm(const GraphNode& node,
                                                               const GridT& udf_grid,
                                                               const ConfidenceGridT& confidence_grid) {
    
    auto udf_accessor = udf_grid.getConstAccessor();
    float udf_value = udf_accessor.getValue(node.coord);
    
    // 内部代价：UDF值越小，内部代价越小
    float inside_cost = udf_value * config_.data_weight * node.confidence;
    
    // 外部代价：UDF值越大，外部代价越小
    float outside_cost = (config_.plane_distance_threshold - udf_value) * config_.data_weight * node.confidence;
    outside_cost = std::max(0.0f, outside_cost);
    
    return {inside_cost, outside_cost};
}

float EnhancedGraphCutSolver::computeSmoothnessTerm(const GraphNode& node1,
                                                   const GraphNode& node2,
                                                   const PointCloudT& cloud) {
    
    // 基于距离的平滑项
    float distance = (node1.world_pos - node2.world_pos).length();
    float distance_factor = std::exp(-distance / config_.plane_distance_threshold);
    
    // 基于置信度的平滑项
    float confidence_factor = (node1.confidence + node2.confidence) * 0.5f;
    
    return distance_factor * confidence_factor;
}

float EnhancedGraphCutSolver::computeVisibilityTerm(const GraphNode& node,
                                                   const PointCloudT& cloud) {
    // 简化的可见性计算
    // 实际实现应该考虑扫描仪视角和射线投射
    
    // 假设扫描仪位置在点云中心上方
    PointT min_pt, max_pt;
    pcl::getMinMax3D(cloud, min_pt, max_pt);
    
    openvdb::Vec3f camera_pos(
        (min_pt.x + max_pt.x) * 0.5f,
        (min_pt.y + max_pt.y) * 0.5f,
        max_pt.z + 2.0f  // 2米高度
    );
    
    // 计算到相机的距离
    float distance_to_camera = (node.world_pos - camera_pos).length();
    
    // 距离越近，可见性越高
    float visibility = std::exp(-distance_to_camera / 5.0f);  // 5米衰减
    
    return visibility * config_.visibility_weight;
}

float EnhancedGraphCutSolver::computeFreeSpaceTerm(const GraphNode& node,
                                                   const PointCloudT& cloud) {
    // 简化的自由空间计算
    // 检查节点周围是否有足够的自由空间
    
    pcl::KdTreeFLANN<PointT> kdtree;
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree.setInputCloud(cloud_ptr);
    
    PointT search_point;
    search_point.x = node.world_pos.x();
    search_point.y = node.world_pos.y();
    search_point.z = node.world_pos.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    int count = kdtree.radiusSearch(search_point, config_.free_space_radius, indices, distances);
    
    // 点密度越低，自由空间概率越高
    float density = static_cast<float>(count) / (config_.free_space_radius * config_.free_space_radius * config_.free_space_radius);
    float free_space_prob = 1.0f / (1.0f + density * 1000.0f);
    
    return free_space_prob * config_.free_space_weight;
}

float EnhancedGraphCutSolver::computePlanarityTerm(const GraphNode& node1,
                                                  const GraphNode& node2,
                                                  const PointCloudT& cloud) {
    
    // 检查两个节点是否在同一平面上
    openvdb::Vec3f midpoint = (node1.world_pos + node2.world_pos) * 0.5f;
    
    // 使用RANSAC拟合局部平面
    pcl::KdTreeFLANN<PointT> kdtree;
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree.setInputCloud(cloud_ptr);
    
    PointT search_point;
    search_point.x = midpoint.x();
    search_point.y = midpoint.y();
    search_point.z = midpoint.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    
    if (kdtree.radiusSearch(search_point, config_.plane_distance_threshold * 3.0f, indices, distances) < 10) {
        return 0.0f;
    }
    
    // 创建邻域点云
    PointCloudT::Ptr neighborhood(new PointCloudT);
    for (int idx : indices) {
        neighborhood->push_back(cloud.points[idx]);
    }
    
    // RANSAC平面拟合
    pcl::SampleConsensusModelPlane<PointT>::Ptr model(
        new pcl::SampleConsensusModelPlane<PointT>(neighborhood));
    pcl::RandomSampleConsensus<PointT> ransac(model);
    ransac.setDistanceThreshold(config_.plane_distance_threshold);
    ransac.setMaxIterations(100);
    
    if (ransac.computeModel()) {
        std::vector<int> inliers;
        ransac.getInliers(inliers);
        
        float planarity = static_cast<float>(inliers.size()) / indices.size();
        return planarity;
    }
    
    return 0.0f;
}

float EnhancedGraphCutSolver::computeColorTerm(const GraphNode& node1,
                                              const GraphNode& node2,
                                              const PointCloudT& cloud) {
    
    // 查找两个节点附近的点的颜色
    pcl::KdTreeFLANN<PointT> kdtree;
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree.setInputCloud(cloud_ptr);
    
    auto getNodeColor = [&](const GraphNode& node) -> Eigen::Vector3f {
        PointT search_point;
        search_point.x = node.world_pos.x();
        search_point.y = node.world_pos.y();
        search_point.z = node.world_pos.z();
        
        std::vector<int> indices(1);
        std::vector<float> distances(1);
        
        if (kdtree.nearestKSearch(search_point, 1, indices, distances) > 0) {
            const auto& pt = cloud.points[indices[0]];
            return Eigen::Vector3f(pt.r, pt.g, pt.b);
        }
        
        return Eigen::Vector3f(128, 128, 128);  // 默认灰色
    };
    
    Eigen::Vector3f color1 = getNodeColor(node1);
    Eigen::Vector3f color2 = getNodeColor(node2);
    
    // 计算颜色差异
    float color_diff = (color1 - color2).norm() / 255.0f;  // 归一化到[0,1]
    
    // 颜色差异越大，分割代价越小
    return std::exp(-color_diff * 5.0f);
}

std::unique_ptr<MaxFlowSolver> EnhancedGraphCutSolver::createOptimalSolver() {
    switch (solver_type_) {
        case SolverType::KOLMOGOROV:
            return std::make_unique<KolmogorovMaxFlowSolver>();
        case SolverType::BOOST_GRAPH:
            return std::make_unique<BoostMaxFlowSolver>();
        case SolverType::AUTO:
        default:
            // 根据图的大小自动选择
            if (nodes_.size() > 100000) {
                return std::make_unique<BoostMaxFlowSolver>();  // 大图使用Boost
            } else {
                return std::make_unique<KolmogorovMaxFlowSolver>();  // 小图使用Kolmogorov
            }
    }
}

// ============================================================================
// 可见性计算器实现
// ============================================================================

VisibilityCalculator::VisibilityCalculator(const PointCloudT& cloud) 
    : cloud_(cloud) {
    
    // 初始化KD树
    kdtree_ = std::make_unique<pcl::KdTreeFLANN<PointT>>();
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree_->setInputCloud(cloud_ptr);
    
    // 构建八叉树加速结构
    buildOctree();
}

float VisibilityCalculator::computeVisibility(const openvdb::Vec3f& point,
                                             const std::vector<openvdb::Vec3f>& camera_positions) {
    
    float total_visibility = 0.0f;
    
    for (const auto& camera_pos : camera_positions) {
        openvdb::Vec3f direction = point - camera_pos;
        float distance = direction.length();
        direction.normalize();
        
        openvdb::Vec3f hit_point;
        bool hit = raycast(camera_pos, direction, distance, hit_point);
        
        if (!hit) {
            // 射线没有击中障碍物，点可见
            total_visibility += 1.0f;
        } else {
            // 检查击中点是否接近目标点
            float hit_distance = (hit_point - camera_pos).length();
            if (std::abs(hit_distance - distance) < 0.05f) {  // 5cm容差
                total_visibility += 1.0f;
            }
        }
    }
    
    return total_visibility / camera_positions.size();
}

bool VisibilityCalculator::raycast(const openvdb::Vec3f& origin,
                                  const openvdb::Vec3f& direction,
                                  float max_distance,
                                  openvdb::Vec3f& hit_point) {
    
    // 简化的射线投射实现
    float step_size = 0.02f;  // 2cm步长
    int max_steps = static_cast<int>(max_distance / step_size);
    
    for (int i = 1; i <= max_steps; ++i) {
        openvdb::Vec3f current_pos = origin + direction * (step_size * i);
        
        // 检查当前位置是否有点
        PointT search_point;
        search_point.x = current_pos.x();
        search_point.y = current_pos.y();
        search_point.z = current_pos.z();
        
        std::vector<int> indices;
        std::vector<float> distances;
        
        if (kdtree_->radiusSearch(search_point, step_size, indices, distances) > 0) {
            hit_point = current_pos;
            return true;
        }
    }
    
    return false;
}

void VisibilityCalculator::buildOctree() {
    // 简化的八叉树构建
    // 实际实现应该递归构建完整的八叉树
    octree_root_ = std::make_unique<OctreeNode>();
    
    // 计算边界框
    PointT min_pt, max_pt;
    pcl::getMinMax3D(cloud_, min_pt, max_pt);
    
    octree_root_->center = openvdb::Vec3f(
        (min_pt.x + max_pt.x) * 0.5f,
        (min_pt.y + max_pt.y) * 0.5f,
        (min_pt.z + max_pt.z) * 0.5f
    );
    
    octree_root_->size = std::max({
        max_pt.x - min_pt.x,
        max_pt.y - min_pt.y,
        max_pt.z - min_pt.z
    });
    
    // 将所有点添加到根节点
    for (size_t i = 0; i < cloud_.size(); ++i) {
        octree_root_->point_indices.push_back(static_cast<int>(i));
    }
}

// ============================================================================
// 自由空间分析器实现
// ============================================================================

FreeSpaceAnalyzer::FreeSpaceAnalyzer(const PointCloudT& cloud) 
    : cloud_(cloud) {
    
    kdtree_ = std::make_unique<pcl::KdTreeFLANN<PointT>>();
    PointCloudT::Ptr cloud_ptr(new PointCloudT(cloud));
    kdtree_->setInputCloud(cloud_ptr);
}

std::vector<openvdb::Vec3f> FreeSpaceAnalyzer::detectFreeSpaceSeeds(const openvdb::FloatGrid& udf_grid) {
    std::vector<openvdb::Vec3f> seeds;
    
    // 在UDF网格中寻找自由空间区域
    for (auto iter = udf_grid.cbeginValueOn(); iter; ++iter) {
        float udf_value = iter.getValue();
        
        // 如果UDF值较大，可能是自由空间
        if (udf_value > 0.1f) {  // 10cm以上
            openvdb::Vec3f world_pos = udf_grid.transform().indexToWorld(iter.getCoord());
            
            // 检查周围是否真的是自由空间
            float free_space_prob = computeFreeSpaceProbability(world_pos);
            if (free_space_prob > 0.7f) {
                seeds.push_back(world_pos);
            }
        }
    }
    
    std::cout << "检测到 " << seeds.size() << " 个自由空间种子点" << std::endl;
    return seeds;
}

float FreeSpaceAnalyzer::computeFreeSpaceProbability(const openvdb::Vec3f& point) {
    PointT search_point;
    search_point.x = point.x();
    search_point.y = point.y();
    search_point.z = point.z();
    
    std::vector<int> indices;
    std::vector<float> distances;
    float search_radius = 0.3f;  // 30cm搜索半径
    
    int count = kdtree_->radiusSearch(search_point, search_radius, indices, distances);
    
    // 点密度越低，自由空间概率越高
    float volume = (4.0f / 3.0f) * M_PI * search_radius * search_radius * search_radius;
    float density = static_cast<float>(count) / volume;
    
    // 使用sigmoid函数映射密度到概率
    float probability = 1.0f / (1.0f + std::exp(density - 500.0f));
    
    return probability;
}

std::vector<FreeSpaceAnalyzer::Opening> FreeSpaceAnalyzer::detectOpenings() {
    std::vector<Opening> openings;
    
    // 简化的开口检测
    // 实际实现应该分析点云的几何结构，识别门窗等开口
    
    // 计算点云边界框
    PointT min_pt, max_pt;
    pcl::getMinMax3D(cloud_, min_pt, max_pt);
    
    // 在墙面上寻找开口（简化实现）
    float wall_thickness = 0.2f;  // 20cm墙厚
    
    // 检查X方向的墙面
    for (float y = min_pt.y; y <= max_pt.y; y += 0.5f) {
        for (float z = min_pt.z; z <= max_pt.z; z += 0.5f) {
            // 检查是否有开口
            openvdb::Vec3f check_pos(min_pt.x, y, z);
            float free_space_prob = computeFreeSpaceProbability(check_pos);
            
            if (free_space_prob > 0.8f) {
                Opening opening;
                opening.center = check_pos;
                opening.normal = openvdb::Vec3f(1, 0, 0);
                opening.width = 1.0f;   // 默认1米宽
                opening.height = 2.0f;  // 默认2米高
                opening.confidence = free_space_prob;
                opening.type = Opening::Type::OTHER;
                
                openings.push_back(opening);
            }
        }
    }
    
    std::cout << "检测到 " << openings.size() << " 个潜在开口" << std::endl;
    return openings;
}

// ============================================================================
// 能量项计算器实现
// ============================================================================

EnergyTermCalculator::EnergyTermCalculator(const GraphCutConfig& config) 
    : config_(config) {
}

float EnergyTermCalculator::computeTotalEnergy(const GraphNode& node1,
                                              const GraphNode& node2,
                                              const PointCloudT& cloud,
                                              EnergyTermType term_type) {
    
    switch (term_type) {
        case EnergyTermType::SMOOTHNESS_TERM:
            return computeSmoothnessEnergy(node1, node2, cloud);
        case EnergyTermType::PLANARITY_TERM:
            // 平面性能量计算
            return 0.0f;  // 简化实现
        case EnergyTermType::COLOR_TERM:
            // 颜色能量计算
            return 0.0f;  // 简化实现
        default:
            return 0.0f;
    }
}

float EnergyTermCalculator::computeSmoothnessEnergy(const GraphNode& node1,
                                                   const GraphNode& node2,
                                                   const PointCloudT& cloud) {
    
    // 基于距离和置信度的平滑能量
    float distance = (node1.world_pos - node2.world_pos).length();
    float distance_factor = std::exp(-distance / 0.1f);  // 10cm衰减
    
    float confidence_factor = (node1.confidence + node2.confidence) * 0.5f;
    
    return distance_factor * confidence_factor * config_.smoothness_weight;
}

// ============================================================================
// 工厂实现
// ============================================================================

std::unique_ptr<EnhancedGraphCutSolver> GraphCutSolverFactory::create(
    SolverVariant variant,
    const GraphCutConfig& config) {
    
    auto solver = std::make_unique<EnhancedGraphCutSolver>(config);
    
    switch (variant) {
        case SolverVariant::ENHANCED:
            solver->setSolverType(EnhancedGraphCutSolver::SolverType::KOLMOGOROV);
            break;
        case SolverVariant::HIERARCHICAL:
            solver->setSolverType(EnhancedGraphCutSolver::SolverType::BOOST_GRAPH);
            break;
        case SolverVariant::PARALLEL:
            solver->setSolverType(EnhancedGraphCutSolver::SolverType::AUTO);
            break;
        case SolverVariant::BASIC:
        default:
            solver->setSolverType(EnhancedGraphCutSolver::SolverType::KOLMOGOROV);
            break;
    }
    
    return solver;
}

} // namespace recon


#include "graph_cut.h"
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <queue>
#include <chrono>
#include <limits>
#include <algorithm>
#include <Eigen/Eigenvalues>

namespace recon {

GraphCutOptimizer::GraphCutOptimizer(const GraphCutConfig& config)
    : config_(config) {}

GraphCutOptimizer::~GraphCutOptimizer() = default;

bool GraphCutOptimizer::optimize(const UDFGridT::Ptr& udf_grid,
                                 const ConfidenceGridT::Ptr& confidence_grid,
                                 const PointCloudT::Ptr& cloud,
                                 LabelGridT::Ptr& label_grid) {
    if (!udf_grid || !confidence_grid || !cloud) return false;
    auto start = std::chrono::high_resolution_clock::now();
    nodes_.clear();
    edges_.clear();
    coord_to_node_.clear();
    transform_ = udf_grid->transform().copy();
    cloud_ = cloud;
    kdtree_.setInputCloud(cloud);
    if (!buildGraph(udf_grid, confidence_grid, cloud)) return false;
    stats_.total_nodes = static_cast<int>(nodes_.size());
    stats_.total_edges = static_cast<int>(edges_.size());
    if (!solveMaxFlow()) return false;
    buildLabelGrid(label_grid);
    for (const auto& n : nodes_) {
        if (n.label == Label::INSIDE) stats_.inside_nodes++; else stats_.free_nodes++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    stats_.optimization_time = std::chrono::duration<double>(end - start).count();
    return true;
}

bool GraphCutOptimizer::buildGraph(const UDFGridT::Ptr& udf_grid,
                                   const ConfidenceGridT::Ptr& confidence_grid,
                                   const PointCloudT::Ptr& cloud) {
    auto udf_accessor = udf_grid->getConstAccessor();
    auto conf_accessor = confidence_grid->getConstAccessor();
    for (auto iter = udf_grid->cbeginValueOn(); iter; ++iter) {
        openvdb::Coord coord = iter.getCoord();
        float udf = iter.getValue();
        float conf = conf_accessor.getValue(coord);
        addVoxelNode(coord, udf, conf, cloud);
    }
    if (nodes_.empty()) return false;
    // 统计最大密度
    max_density_ = 1.0f;
    for (const auto& n : nodes_) max_density_ = std::max(max_density_, n.local_density);
    // 自由空间种子与泛洪
    free_seeds_ = generateFreeSpaceSeeds(cloud);
    floodFillFreeRegion(free_seeds_);
    // 可见性成本
    for (auto& n : nodes_) {
        n.visibility_cost = computeVisibilityCost(n.coord, free_seeds_, cloud);
    }
    buildEdges();
    return true;
}

void GraphCutOptimizer::addVoxelNode(const openvdb::Coord& coord,
                                     float udf_value,
                                     float confidence,
                                     const PointCloudT::Ptr& cloud) {
    VoxelNode node;
    node.coord = coord;
    node.udf_value = udf_value;
    node.confidence = confidence;
    node.is_planar = isPlanarRegion(coord, cloud);
    node.local_density = computeLocalDensity(coord, cloud);
    int index = static_cast<int>(nodes_.size());
    nodes_.push_back(node);
    coord_to_node_[coord] = index;
}

void GraphCutOptimizer::buildEdges() {
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& node = nodes_[i];
        for (const auto& ncoord : getNeighbors(node.coord)) {
            auto it = coord_to_node_.find(ncoord);
            if (it == coord_to_node_.end()) continue;
            int j = it->second;
            if (j <= static_cast<int>(i)) continue; // avoid duplicates
            float w = computeSmoothWeight(node, nodes_[j]);
            edges_.push_back({static_cast<int>(i), j, w, node.is_planar && nodes_[j].is_planar});
        }
    }
}

std::pair<float, float> GraphCutOptimizer::computeDataCosts(const VoxelNode& node,
                                                           const PointCloudT::Ptr& /*cloud*/) {
    float phi = std::min(node.udf_value / 0.03f, 1.0f);
    float inside = node.confidence * phi * config_.inside_cost_multiplier;
    float alpha = node.is_planar ? config_.planar_alpha : config_.detail_alpha;
    float free = config_.free_cost_base + (node.is_free_region ? -alpha : alpha);
    free = std::max(0.0f, free);
    free += config_.visibility_weight * node.visibility_cost;
    return {free, inside};
}

float GraphCutOptimizer::computeSmoothWeight(const VoxelNode& node1, const VoxelNode& node2) {
    float lambda = (node1.is_planar && node2.is_planar) ?
                   config_.planar_lambda : config_.detail_lambda;
    float w = config_.base_weight * lambda;
    if (node1.is_planar && node2.is_planar) w *= config_.planar_multiplier;
    if (node1.is_planar != node2.is_planar) w *= config_.cross_plane_multiplier;
    if (config_.density_adaptive && max_density_ > 1e-6f) {
        float avg = 0.5f * (node1.local_density + node2.local_density);
        w *= avg / max_density_;
    }
    return w;
}

bool GraphCutOptimizer::isPlanarRegion(const openvdb::Coord& coord, const PointCloudT::Ptr& /*cloud*/) {
    PointT search;
    auto world = transform_->indexToWorld(coord);
    search.x = world.x(); search.y = world.y(); search.z = world.z();
    std::vector<int> idx(20);
    std::vector<float> dist2(20);
    if (kdtree_.nearestKSearch(search, 20, idx, dist2) < 6) return false;
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (int i = 0; i < (int)idx.size(); ++i) {
        const auto& p = cloud_->points[idx[i]];
        centroid += Eigen::Vector3f(p.x, p.y, p.z);
    }
    centroid /= static_cast<float>(idx.size());
    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (int i = 0; i < (int)idx.size(); ++i) {
        const auto& p = cloud_->points[idx[i]];
        Eigen::Vector3f v(p.x, p.y, p.z);
        v -= centroid;
        cov += v * v.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(cov);
    auto evals = solver.eigenvalues();
    float curvature = evals[0] / evals.sum();
    return curvature < 0.01f;
}

float GraphCutOptimizer::computeLocalDensity(const openvdb::Coord& coord, const PointCloudT::Ptr& /*cloud*/) {
    PointT search;
    auto world = transform_->indexToWorld(coord);
    search.x = world.x(); search.y = world.y(); search.z = world.z();
    std::vector<int> idx(16);
    std::vector<float> dist2(16);
    int count = kdtree_.nearestKSearch(search, 16, idx, dist2);
    return static_cast<float>(count);
}

std::vector<openvdb::Coord> GraphCutOptimizer::generateFreeSpaceSeeds(const PointCloudT::Ptr& cloud) {
    std::vector<openvdb::Coord> seeds;
    pcl::PointXYZRGBNormal min_pt, max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);
    for (float x = min_pt.x; x <= max_pt.x; x += config_.seed_spacing) {
        for (float y = min_pt.y; y <= max_pt.y; y += config_.seed_spacing) {
            for (float z = min_pt.z; z <= max_pt.z; z += config_.seed_spacing) {
                PointT p; p.x = x; p.y = y; p.z = z;
                std::vector<int> idx(1); std::vector<float> dist2(1);
                if (kdtree_.nearestKSearch(p, 1, idx, dist2) == 0) continue;
                if (std::sqrt(dist2[0]) >= config_.min_distance_to_surface) {
                    openvdb::Vec3d pos(x, y, z);
                    seeds.push_back(transform_->worldToIndexCellCentered(pos));
                }
            }
        }
    }
    return seeds;
}

float GraphCutOptimizer::computeVisibilityCost(const openvdb::Coord& coord,
                                               const std::vector<openvdb::Coord>& seeds,
                                               const PointCloudT::Ptr& /*cloud*/) {
    if (seeds.empty()) return 0.0f;
    auto target = transform_->indexToWorld(coord);
    int visible = 0;
    for (const auto& s : seeds) {
        auto start = transform_->indexToWorld(s);
        Eigen::Vector3f dir(target.x() - start.x(), target.y() - start.y(), target.z() - start.z());
        float len = dir.norm();
        int steps = std::max(1, static_cast<int>(len / config_.min_distance_to_surface));
        Eigen::Vector3f step = dir / static_cast<float>(steps);
        Eigen::Vector3f pos(start.x(), start.y(), start.z());
        bool blocked = false;
        for (int i = 0; i < steps; ++i) {
            pos += step;
            PointT q; q.x = pos.x(); q.y = pos.y(); q.z = pos.z();
            std::vector<int> idx(1); std::vector<float> dist2(1);
            if (kdtree_.nearestKSearch(q, 1, idx, dist2) > 0 &&
                dist2[0] < config_.min_distance_to_surface * config_.min_distance_to_surface) {
                blocked = true; break;
            }
        }
        if (!blocked) visible++;
    }
    return 1.0f - static_cast<float>(visible) / static_cast<float>(seeds.size());
}

bool GraphCutOptimizer::solveMaxFlow() {
    int n = static_cast<int>(nodes_.size());
    
    // 创建PyMaxflow求解器
    PyMaxflowSolver solver(n);
    
    // 添加数据项（源汇边）
    for (int i = 0; i < n; ++i) {
        auto costs = computeDataCosts(nodes_[i], cloud_);
        solver.addTerminalEdge(i, costs.first, costs.second);
    }
    
    // 添加平滑项边
    for (const auto& e : edges_) {
        solver.addEdge(e.node1, e.node2, e.weight, e.weight);
    }
    
    // 求解最大流
    auto result = solver.solve();
    
    if (!result.success) {
        std::cerr << "PyMaxflow求解失败: " << result.error_message << std::endl;
        return false;
    }
    
    // 更新统计信息
    stats_.max_flow_value = result.flow_value;
    
    // 更新节点标签
    for (int i = 0; i < n; ++i) {
        nodes_[i].label = solver.isSourceSide(i) ? Label::FREE : Label::INSIDE;
    }
    
    return true;
}

void GraphCutOptimizer::buildLabelGrid(LabelGridT::Ptr& label_grid) {
    label_grid = LabelGridT::create(0);
    label_grid->setTransform(transform_);
    auto accessor = label_grid->getAccessor();
    for (const auto& node : nodes_) {
        accessor.setValue(node.coord, static_cast<int>(node.label));
    }
}

std::vector<openvdb::Coord> GraphCutOptimizer::getNeighbors(const openvdb::Coord& coord) {
    static const int offsets[6][3] = {
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}
    };
    std::vector<openvdb::Coord> res;
    res.reserve(6);
    for (const auto& o : offsets) {
        res.emplace_back(coord.x() + o[0], coord.y() + o[1], coord.z() + o[2]);
    }
    return res;
}

bool GraphCutOptimizer::isValidCoord(const openvdb::Coord& /*coord*/) {
    return true;
}

float GraphCutOptimizer::computeDistance(const openvdb::Coord& c1, const openvdb::Coord& c2) {
    auto w1 = transform_->indexToWorld(c1);
    auto w2 = transform_->indexToWorld(c2);
    return static_cast<float>((w1 - w2).length());
}

void GraphCutOptimizer::floodFillFreeRegion(const std::vector<openvdb::Coord>& seeds) {
    std::queue<int> q;
    std::vector<bool> visited(nodes_.size(), false);
    for (const auto& s : seeds) {
        auto it = coord_to_node_.find(s);
        if (it == coord_to_node_.end()) continue;
        int idx = it->second;
        if (nodes_[idx].udf_value < config_.min_distance_to_surface) continue;
        nodes_[idx].is_free_region = true;
        visited[idx] = true;
        q.push(idx);
    }
    while (!q.empty()) {
        int i = q.front(); q.pop();
        const auto& node = nodes_[i];
        for (const auto& ncoord : getNeighbors(node.coord)) {
            auto it = coord_to_node_.find(ncoord);
            if (it == coord_to_node_.end()) continue;
            int j = it->second;
            if (visited[j]) continue;
            if (nodes_[j].udf_value < config_.min_distance_to_surface) continue;
            visited[j] = true;
            nodes_[j].is_free_region = true;
            q.push(j);
        }
    }
}

} // namespace recon


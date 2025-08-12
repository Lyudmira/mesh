/**
 * 最小化图割实现
 * 提供基本的图割功能，确保编译通过
 */

#include "minimal_graph_cut.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <pcl/common/common.h>
#include <pcl/conversions.h>

namespace recon {

// ============================================================================
// MinimalGraphCutSolver实现
// ============================================================================

MinimalGraphCutSolver::MinimalGraphCutSolver(const MinimalGraphCutConfig& config)
    : config_(config) {
    std::cout << "最小化图割求解器初始化完成" << std::endl;
}

MinimalGraphCutSolver::~MinimalGraphCutSolver() = default;

bool MinimalGraphCutSolver::solve(
    const PointCloudT::Ptr& cloud,
    const GridT::Ptr& grid,
    pcl::PolygonMesh& result_mesh) {
    
    std::cout << "开始最小化图割求解..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        if (!cloud || cloud->empty()) {
            std::cerr << "输入点云为空" << std::endl;
            return false;
        }
        
        if (!grid) {
            std::cerr << "输入网格为空" << std::endl;
            return false;
        }
        
        // 1. 构建简化图
        if (!buildSimpleGraph(cloud, grid)) {
            std::cerr << "图构建失败" << std::endl;
            return false;
        }
        
        // 2. 执行简化最大流
        if (!executeSimpleMaxFlow()) {
            std::cerr << "最大流求解失败" << std::endl;
            return false;
        }
        
        // 3. 提取简化网格
        if (!extractSimpleMesh(grid, result_mesh)) {
            std::cerr << "网格提取失败" << std::endl;
            return false;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double solve_time = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "最小化图割求解完成，时间: " << solve_time << " 秒" << std::endl;
        std::cout << "结果网格面数: " << result_mesh.polygons.size() << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "图割求解异常: " << e.what() << std::endl;
        return false;
    }
}

bool MinimalGraphCutSolver::buildSimpleGraph(
    const PointCloudT::Ptr& cloud,
    const GridT::Ptr& grid) {
    
    std::cout << "构建简化图..." << std::endl;
    
    try {
        // 获取活跃体素的边界框
        current_bbox_ = grid->evalActiveVoxelBoundingBox();
        
        if (current_bbox_.empty()) {
            std::cerr << "网格中没有活跃体素" << std::endl;
            return false;
        }
        
        int width = current_bbox_.max().x() - current_bbox_.min().x() + 1;
        int height = current_bbox_.max().y() - current_bbox_.min().y() + 1;
        int depth = current_bbox_.max().z() - current_bbox_.min().z() + 1;
        
        int total_nodes = width * height * depth;
        
        std::cout << "网格维度: " << width << "x" << height << "x" << depth 
                  << ", 总节点数: " << total_nodes << std::endl;
        
        // 初始化图数据结构
        graph_edges_.clear();
        graph_edges_.resize(total_nodes);
        node_labels_.clear();
        node_labels_.resize(total_nodes, 0.5f); // 初始化为中性标签
        
        // 构建图的边
        int edge_count = 0;
        for (int x = current_bbox_.min().x(); x <= current_bbox_.max().x(); ++x) {
            for (int y = current_bbox_.min().y(); y <= current_bbox_.max().y(); ++y) {
                for (int z = current_bbox_.min().z(); z <= current_bbox_.max().z(); ++z) {
                    openvdb::Coord coord(x, y, z);
                    
                    if (grid->tree().isValueOn(coord)) {
                        int node_id = getNodeId(coord, current_bbox_);
                        
                        // 计算数据代价
                        float data_cost = computeSimpleDataCost(coord, grid);
                        node_labels_[node_id] = data_cost;
                        
                        // 添加邻居边
                        std::vector<openvdb::Coord> neighbors = {
                            openvdb::Coord(x+1, y, z),
                            openvdb::Coord(x, y+1, z),
                            openvdb::Coord(x, y, z+1)
                        };
                        
                        for (const auto& neighbor : neighbors) {
                            if (neighbor.x() <= current_bbox_.max().x() &&
                                neighbor.y() <= current_bbox_.max().y() &&
                                neighbor.z() <= current_bbox_.max().z() &&
                                grid->tree().isValueOn(neighbor)) {
                                
                                int neighbor_id = getNodeId(neighbor, current_bbox_);
                                graph_edges_[node_id].push_back(neighbor_id);
                                edge_count++;
                            }
                        }
                    }
                }
            }
        }
        
        std::cout << "简化图构建完成，边数: " << edge_count << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "图构建异常: " << e.what() << std::endl;
        return false;
    }
}

bool MinimalGraphCutSolver::executeSimpleMaxFlow() {
    std::cout << "执行简化最大流..." << std::endl;
    
    try {
        // 简化的最大流：基于阈值的二值化
        for (size_t i = 0; i < node_labels_.size(); ++i) {
            // 简单的阈值分割
            if (node_labels_[i] > 0.5f) {
                node_labels_[i] = 1.0f; // 前景
            } else {
                node_labels_[i] = 0.0f; // 背景
            }
        }
        
        // 应用简单的平滑
        std::vector<float> smoothed_labels = node_labels_;
        for (size_t i = 0; i < node_labels_.size(); ++i) {
            if (!graph_edges_[i].empty()) {
                float sum = node_labels_[i];
                int count = 1;
                
                for (int neighbor_id : graph_edges_[i]) {
                    if (neighbor_id >= 0 && neighbor_id < static_cast<int>(node_labels_.size())) {
                        sum += node_labels_[neighbor_id];
                        count++;
                    }
                }
                
                smoothed_labels[i] = sum / count;
            }
        }
        
        // 再次二值化
        for (size_t i = 0; i < smoothed_labels.size(); ++i) {
            node_labels_[i] = smoothed_labels[i] > 0.5f ? 1.0f : 0.0f;
        }
        
        std::cout << "简化最大流完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "最大流求解异常: " << e.what() << std::endl;
        return false;
    }
}

bool MinimalGraphCutSolver::extractSimpleMesh(
    const GridT::Ptr& grid,
    pcl::PolygonMesh& result_mesh) {
    
    std::cout << "提取简化网格..." << std::endl;
    
    try {
        pcl::PointCloud<pcl::PointXYZ> vertices;
        std::vector<pcl::Vertices> polygons;
        
        // 遍历所有标记为前景的体素
        for (int x = current_bbox_.min().x(); x <= current_bbox_.max().x(); ++x) {
            for (int y = current_bbox_.min().y(); y <= current_bbox_.max().y(); ++y) {
                for (int z = current_bbox_.min().z(); z <= current_bbox_.max().z(); ++z) {
                    openvdb::Coord coord(x, y, z);
                    
                    if (grid->tree().isValueOn(coord)) {
                        int node_id = getNodeId(coord, current_bbox_);
                        
                        // 如果节点被标记为前景，添加体素
                        if (node_id < static_cast<int>(node_labels_.size()) && 
                            node_labels_[node_id] > 0.5f) {
                            addVoxelCube(coord, vertices, polygons);
                        }
                    }
                }
            }
        }
        
        // 构建结果网格
        pcl::PCLPointCloud2 cloud2;
        pcl::toPCLPointCloud2(vertices, cloud2);
        result_mesh.cloud = cloud2;
        result_mesh.polygons = polygons;
        
        std::cout << "简化网格提取完成，顶点数: " << vertices.size() 
                  << ", 面数: " << polygons.size() << std::endl;
        
        return !polygons.empty();
        
    } catch (const std::exception& e) {
        std::cerr << "网格提取异常: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// 辅助方法
// ============================================================================

float MinimalGraphCutSolver::computeSimpleDataCost(
    const openvdb::Coord& coord,
    const GridT::Ptr& grid) {
    
    try {
        float udf_value = grid->tree().getValue(coord);
        
        // 简化的数据代价：基于UDF值
        if (udf_value < -0.01f) {
            return 0.9f; // 内部，高概率为前景
        } else if (udf_value > 0.01f) {
            return 0.1f; // 外部，低概率为前景
        } else {
            return 0.5f; // 边界，中等概率
        }
        
    } catch (const std::exception& e) {
        return 0.5f; // 默认值
    }
}

float MinimalGraphCutSolver::computeSimpleSmoothnessCost(
    const openvdb::Coord& coord1,
    const openvdb::Coord& coord2) {
    
    // 简化的平滑代价：固定值
    return config_.smoothness_weight;
}

int MinimalGraphCutSolver::getNodeId(
    const openvdb::Coord& coord,
    const openvdb::CoordBBox& bbox) {
    
    int x = coord.x() - bbox.min().x();
    int y = coord.y() - bbox.min().y();
    int z = coord.z() - bbox.min().z();
    
    int width = bbox.max().x() - bbox.min().x() + 1;
    int height = bbox.max().y() - bbox.min().y() + 1;
    
    return z * width * height + y * width + x;
}

void MinimalGraphCutSolver::addVoxelCube(
    const openvdb::Coord& coord,
    pcl::PointCloud<pcl::PointXYZ>& vertices,
    std::vector<pcl::Vertices>& polygons) {
    
    float voxel_size = 0.01f; // 假设体素大小
    int base_index = static_cast<int>(vertices.size());
    
    // 添加立方体的8个顶点
    for (int i = 0; i < 8; ++i) {
        pcl::PointXYZ vertex;
        vertex.x = coord.x() * voxel_size + ((i & 1) ? voxel_size : 0);
        vertex.y = coord.y() * voxel_size + ((i & 2) ? voxel_size : 0);
        vertex.z = coord.z() * voxel_size + ((i & 4) ? voxel_size : 0);
        vertices.push_back(vertex);
    }
    
    // 添加立方体的12个三角形
    std::vector<std::array<int, 3>> cube_triangles = {
        {0, 1, 2}, {1, 3, 2}, // 底面
        {4, 6, 5}, {5, 6, 7}, // 顶面
        {0, 2, 4}, {2, 6, 4}, // 前面
        {1, 5, 3}, {3, 5, 7}, // 后面
        {0, 4, 1}, {1, 4, 5}, // 左面
        {2, 3, 6}, {3, 7, 6}  // 右面
    };
    
    for (const auto& triangle : cube_triangles) {
        pcl::Vertices polygon;
        polygon.vertices = {
            static_cast<uint32_t>(base_index + triangle[0]),
            static_cast<uint32_t>(base_index + triangle[1]),
            static_cast<uint32_t>(base_index + triangle[2])
        };
        polygons.push_back(polygon);
    }
}

// ============================================================================
// MinimalGraphCutFactory实现
// ============================================================================

std::unique_ptr<MinimalGraphCutSolver> MinimalGraphCutFactory::createStandard() {
    MinimalGraphCutConfig config;
    config.data_weight = 1.0f;
    config.smoothness_weight = 0.5f;
    return std::make_unique<MinimalGraphCutSolver>(config);
}

std::unique_ptr<MinimalGraphCutSolver> MinimalGraphCutFactory::createFast() {
    MinimalGraphCutConfig config;
    config.data_weight = 1.0f;
    config.smoothness_weight = 0.3f;
    return std::make_unique<MinimalGraphCutSolver>(config);
}

std::unique_ptr<MinimalGraphCutSolver> MinimalGraphCutFactory::createHighQuality() {
    MinimalGraphCutConfig config;
    config.data_weight = 1.0f;
    config.smoothness_weight = 0.8f;
    return std::make_unique<MinimalGraphCutSolver>(config);
}

} // namespace recon


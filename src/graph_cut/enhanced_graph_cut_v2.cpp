/**
 * 增强版图割求解器 v2.0 - 集成UDF映射
 * 使用UDF与图割集成器进行自适应权重调整
 */

#include "enhanced_graph_cut.h"
#include "../integration/udf_graphcut_integrator.h"
#include <iostream>
#include <chrono>

namespace recon {

/**
 * 集成版图割求解方法
 * 使用UDF集成器进行自适应参数调整
 */
bool EnhancedGraphCutSolver::solveWithUDFIntegration(
    const GridT& udf_grid,
    const ConfidenceGridT& confidence_grid,
    const RefinementGridT& refinement_grid,
    const PointCloudT& cloud,
    GridT::Ptr& result_grid,
    GraphCutResult& result) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    solving_.store(true);
    
    std::cout << "开始集成版图割优化..." << std::endl;
    
    try {
        // 1. 创建UDF-图割集成器
        std::cout << "创建UDF-图割集成器..." << std::endl;
        auto integrator = IntegratorFactory::createStandardIntegrator();
        
        // 2. 建立能量参数映射
        std::cout << "建立能量参数映射..." << std::endl;
        EnergyParameterMapping mapping;
        if (!integrator->integrateUDFWithGraphCut(
                udf_grid, confidence_grid, refinement_grid, cloud, mapping)) {
            std::cerr << "能量参数映射失败" << std::endl;
            return false;
        }
        
        // 3. 构建自适应图结构
        std::cout << "构建自适应图结构..." << std::endl;
        if (!buildAdaptiveGraph(udf_grid, confidence_grid, mapping, cloud)) {
            std::cerr << "自适应图结构构建失败" << std::endl;
            return false;
        }
        
        std::cout << "图节点数: " << nodes_.size() << ", 边数: " << edges_.size() << std::endl;
        
        // 4. 创建求解器
        solver_ = createOptimalSolver();
        if (!solver_) {
            std::cerr << "求解器创建失败" << std::endl;
            return false;
        }
        
        std::cout << "使用求解器: " << solver_->getName() << std::endl;
        
        // 5. 设置图结构到求解器
        std::unordered_map<int, int> node_mapping;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            int solver_node_id = solver_->addNode();
            node_mapping[static_cast<int>(i)] = solver_node_id;
        }
        
        // 6. 添加自适应权重的边
        for (const auto& edge : edges_) {
            int solver_node1 = node_mapping[edge.node1_id];
            int solver_node2 = node_mapping[edge.node2_id];
            
            // 使用自适应权重
            float adaptive_weight = computeAdaptiveEdgeWeight(edge, mapping, refinement_grid);
            solver_->addEdge(solver_node1, solver_node2, adaptive_weight, adaptive_weight);
        }
        
        // 7. 设置自适应终端权重
        for (size_t i = 0; i < nodes_.size(); ++i) {
            const auto& node = nodes_[i];
            
            // 从映射中获取自适应权重
            auto it = mapping.voxel_attributes.find(node.coord);
            if (it != mapping.voxel_attributes.end()) {
                const auto& attributes = it->second;
                
                // 使用自适应α权重计算数据项
                float inside_cost = attributes.alpha_weight * computeInsideCost(node, udf_grid);
                float outside_cost = attributes.alpha_weight * computeOutsideCost(node, udf_grid);
                
                // 添加可见性惩罚
                inside_cost += attributes.visibility_penalty;
                
                int solver_node_id = node_mapping[static_cast<int>(i)];
                solver_->setTerminalWeights(solver_node_id, inside_cost, outside_cost);
            } else {
                // 使用默认权重
                auto [inside_cost, outside_cost] = computeDataTerm(node, udf_grid, confidence_grid);
                int solver_node_id = node_mapping[static_cast<int>(i)];
                solver_->setTerminalWeights(solver_node_id, inside_cost, outside_cost);
            }
        }
        
        // 8. 求解
        std::cout << "执行自适应最大流求解..." << std::endl;
        float max_flow = solver_->solve();
        
        // 9. 提取结果
        result.labels.resize(nodes_.size());
        for (size_t i = 0; i < nodes_.size(); ++i) {
            int solver_node_id = node_mapping[static_cast<int>(i)];
            result.labels[i] = solver_->getNodeLabel(solver_node_id);
        }
        
        // 10. 创建结果网格
        result_grid = GridT::create(0.0f);
        result_grid->setTransform(udf_grid.transform().copy());
        result_grid->setName("adaptive_graph_cut_result");
        
        auto result_accessor = result_grid->getAccessor();
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (result.labels[i]) {  // 内部点
                result_accessor.setValue(nodes_[i].coord, 1.0f);
            }
        }
        
        // 11. 计算能量统计
        result.total_energy = max_flow;
        result.converged = true;
        
        // 12. 计算详细能量分解
        computeEnergyBreakdown(result, mapping);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.solve_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        
        std::cout << "集成版图割求解完成" << std::endl;
        std::cout << "最大流值: " << max_flow << std::endl;
        std::cout << "内部体素数: " << std::count(result.labels.begin(), result.labels.end(), true) << std::endl;
        std::cout << "数据能量: " << result.data_energy << std::endl;
        std::cout << "平滑能量: " << result.smoothness_energy << std::endl;
        std::cout << "求解时间: " << result.solve_time_seconds << " 秒" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "集成版图割求解异常: " << e.what() << std::endl;
        solving_.store(false);
        return false;
    }
    
    solving_.store(false);
    return true;
}

bool EnhancedGraphCutSolver::buildAdaptiveGraph(
    const GridT& udf_grid,
    const ConfidenceGridT& confidence_grid,
    const EnergyParameterMapping& mapping,
    const PointCloudT& cloud) {
    
    // 清空现有图结构
    nodes_.clear();
    edges_.clear();
    coord_to_node_.clear();
    
    // 1. 从映射中添加图节点
    std::cout << "从映射添加图节点..." << std::endl;
    int node_count = 0;
    
    for (const auto& [coord, attributes] : mapping.voxel_attributes) {
        openvdb::Vec3f world_pos = udf_grid.transform().indexToWorld(coord);
        
        GraphNode node;
        node.coord = coord;
        node.world_pos = world_pos;
        node.confidence = attributes.confidence;
        node.is_boundary = (attributes.plane_distance < 0.03f);
        
        // 使用映射中的自适应数据项
        node.data_cost_inside = attributes.alpha_weight * computeInsideCost(node, udf_grid);
        node.data_cost_outside = attributes.alpha_weight * computeOutsideCost(node, udf_grid);
        
        int node_id = static_cast<int>(nodes_.size());
        nodes_.push_back(node);
        coord_to_node_[coord] = node_id;
        
        node_count++;
        if (node_count % 10000 == 0) {
            std::cout << "已添加 " << node_count << " 个节点" << std::endl;
        }
    }
    
    std::cout << "总节点数: " << nodes_.size() << std::endl;
    
    // 2. 添加自适应图边
    std::cout << "添加自适应图边..." << std::endl;
    addAdaptiveGraphEdges(udf_grid, mapping, cloud);
    
    std::cout << "总边数: " << edges_.size() << std::endl;
    
    return !nodes_.empty();
}

void EnhancedGraphCutSolver::addAdaptiveGraphEdges(
    const GridT& udf_grid,
    const EnergyParameterMapping& mapping,
    const PointCloudT& cloud) {
    
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
                
                // 从映射中获取自适应λ权重
                auto attr1_it = mapping.voxel_attributes.find(node1.coord);
                auto attr2_it = mapping.voxel_attributes.find(node2.coord);
                
                float adaptive_lambda = config_.smoothness_weight; // 默认值
                
                if (attr1_it != mapping.voxel_attributes.end() && 
                    attr2_it != mapping.voxel_attributes.end()) {
                    
                    const auto& attr1 = attr1_it->second;
                    const auto& attr2 = attr2_it->second;
                    
                    // 使用平均λ权重
                    adaptive_lambda = (attr1.lambda_weight + attr2.lambda_weight) * 0.5f;
                    
                    // 基于区域类型进一步调整
                    if (attr1.is_planar && attr2.is_planar) {
                        adaptive_lambda *= 1.5f; // 平面-平面边增强平滑
                    } else if (attr1.is_edge || attr2.is_edge) {
                        adaptive_lambda *= 0.3f; // 边缘区域减弱平滑
                    }
                }
                
                // 添加边
                GraphEdge edge;
                edge.node1_id = static_cast<int>(i);
                edge.node2_id = j;
                edge.weight = adaptive_lambda;
                edge.term_type = EnergyTermType::SMOOTHNESS_TERM;
                edge.direction = node2.world_pos - node1.world_pos;
                
                edges_.push_back(edge);
            }
        }
    }
}

float EnhancedGraphCutSolver::computeAdaptiveEdgeWeight(
    const GraphEdge& edge,
    const EnergyParameterMapping& mapping,
    const RefinementGridT& refinement_grid) {
    
    const auto& node1 = nodes_[edge.node1_id];
    const auto& node2 = nodes_[edge.node2_id];
    
    // 基础权重
    float weight = edge.weight;
    
    // 基于细化网格调整
    auto ref_accessor = refinement_grid.getConstAccessor();
    int level1 = ref_accessor.getValue(node1.coord);
    int level2 = ref_accessor.getValue(node2.coord);
    
    // 细化级别差异调整
    int level_diff = std::abs(level1 - level2);
    if (level_diff > 0) {
        weight *= (1.0f / (1.0f + level_diff * 0.3f));
    }
    
    // 基于属性映射调整
    auto attr1_it = mapping.voxel_attributes.find(node1.coord);
    auto attr2_it = mapping.voxel_attributes.find(node2.coord);
    
    if (attr1_it != mapping.voxel_attributes.end() && 
        attr2_it != mapping.voxel_attributes.end()) {
        
        const auto& attr1 = attr1_it->second;
        const auto& attr2 = attr2_it->second;
        
        // 颜色一致性调整
        float color_diff = std::abs(attr1.color_gradient - attr2.color_gradient) / 255.0f;
        weight *= std::exp(-color_diff * 2.0f);
        
        // 曲率一致性调整
        float curvature_diff = std::abs(attr1.curvature - attr2.curvature);
        weight *= std::exp(-curvature_diff * 3.0f);
    }
    
    return std::max(0.01f, weight);
}

float EnhancedGraphCutSolver::computeInsideCost(const GraphNode& node, const GridT& udf_grid) {
    auto accessor = udf_grid.getConstAccessor();
    float udf_value = accessor.getValue(node.coord);
    
    // 内部代价：UDF值越小（越接近表面），内部代价越小
    return std::max(0.0f, -udf_value + 0.1f);
}

float EnhancedGraphCutSolver::computeOutsideCost(const GraphNode& node, const GridT& udf_grid) {
    auto accessor = udf_grid.getConstAccessor();
    float udf_value = accessor.getValue(node.coord);
    
    // 外部代价：UDF值越大（越远离表面），外部代价越小
    return std::max(0.0f, udf_value + 0.1f);
}

void EnhancedGraphCutSolver::computeEnergyBreakdown(
    GraphCutResult& result,
    const EnergyParameterMapping& mapping) {
    
    result.data_energy = 0.0f;
    result.smoothness_energy = 0.0f;
    
    // 计算数据能量
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& node = nodes_[i];
        bool label = result.labels[i];
        
        auto attr_it = mapping.voxel_attributes.find(node.coord);
        if (attr_it != mapping.voxel_attributes.end()) {
            const auto& attr = attr_it->second;
            
            if (label) {
                result.data_energy += attr.alpha_weight * node.data_cost_inside;
            } else {
                result.data_energy += attr.alpha_weight * node.data_cost_outside;
            }
            
            // 添加可见性惩罚
            if (label) {
                result.data_energy += attr.visibility_penalty;
            }
        }
    }
    
    // 计算平滑能量
    for (const auto& edge : edges_) {
        bool label1 = result.labels[edge.node1_id];
        bool label2 = result.labels[edge.node2_id];
        
        if (label1 != label2) {
            result.smoothness_energy += edge.weight;
        }
    }
}

} // namespace recon


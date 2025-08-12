/**
 * 主重建器实现
 * 统一调度所有重建模块的核心逻辑
 */

#include "main_reconstructor.h"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/statistical_outlier_removal.h>

namespace recon {

// ============================================================================
// MainReconstructor实现
// ============================================================================

MainReconstructor::MainReconstructor(const MainReconstructorConfig& config)
    : config_(config), current_stage_(ReconstructionStage::INITIALIZATION), 
      is_running_(false), should_stop_(false) {
    
    if (!initializeComponents()) {
        throw std::runtime_error("主重建器初始化失败");
    }
}

MainReconstructor::~MainReconstructor() {
    cleanup();
}

bool MainReconstructor::initializeComponents() {
    try {
        std::cout << "初始化主重建器组件..." << std::endl;
        
        // 创建输出目录
        std::filesystem::create_directories(config_.output_directory);
        if (config_.enable_debug_mode) {
            std::filesystem::create_directories(config_.debug_output_dir);
        }
        
        // 初始化UDF构建器
        if (config_.enable_udf_building) {
            udf_builder_ = std::make_unique<EnhancedUDFBuilder>(config_.udf_config);
            std::cout << "UDF构建器初始化完成" << std::endl;
        }
        
        // 初始化图割集成器
        if (config_.enable_graph_cut) {
            graph_cut_integrator_ = std::make_unique<UDFGraphCutIntegrator>(config_.integration_config);
            std::cout << "图割集成器初始化完成" << std::endl;
        }
        
        // 初始化细节重建集成器
        if (config_.enable_detail_reconstruction) {
            detail_integrator_ = std::make_unique<DetailReconstructionIntegrator>(config_.detail_config);
            std::cout << "细节重建集成器初始化完成" << std::endl;
        }
        
        // 初始化融合合法化集成器
        if (config_.enable_fusion_legalization) {
            fusion_integrator_ = std::make_unique<FusionLegalizationIntegrator>(config_.fusion_config);
            std::cout << "融合合法化集成器初始化完成" << std::endl;
        }
        
        std::cout << "所有组件初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "组件初始化异常: " << e.what() << std::endl;
        return false;
    }
}

bool MainReconstructor::performReconstruction(
    const PointCloudT::Ptr& input_cloud,
    MainReconstructorResult& result) {
    
    if (is_running_) {
        std::cerr << "重建器正在运行中" << std::endl;
        return false;
    }
    
    is_running_ = true;
    should_stop_ = false;
    start_time_ = std::chrono::high_resolution_clock::now();
    
    std::cout << "开始完整3D重建..." << std::endl;
    std::cout << "输入点云: " << input_cloud->size() << " 个点" << std::endl;
    
    try {
        // 重置结果
        result = MainReconstructorResult{};
        
        // 阶段1: 初始化
        updateStage(ReconstructionStage::INITIALIZATION);
        if (!validateInputData(*input_cloud)) {
            result.success = false;
            result.error_message = "输入数据验证失败";
            result.failed_stage = current_stage_;
            is_running_ = false;
            return false;
        }
        
        // 预处理点云
        PointCloudT::Ptr processed_cloud;
        if (!preprocessPointCloud(input_cloud, processed_cloud)) {
            result.success = false;
            result.error_message = "点云预处理失败";
            result.failed_stage = current_stage_;
            is_running_ = false;
            return false;
        }
        
        result.stats.input_points = static_cast<int>(processed_cloud->size());
        
        // 阶段2: UDF构建
        if (config_.enable_udf_building) {
            updateStage(ReconstructionStage::UDF_BUILDING);
            if (!performUDFBuilding(processed_cloud, result.final_grid)) {
                result.success = false;
                result.error_message = "UDF构建失败";
                result.failed_stage = current_stage_;
                is_running_ = false;
                return false;
            }
        }
        
        // 检查是否需要停止
        if (should_stop_) {
            result.success = false;
            result.error_message = "用户请求停止";
            is_running_ = false;
            return false;
        }
        
        // 阶段3: 图割优化
        if (config_.enable_graph_cut) {
            updateStage(ReconstructionStage::GRAPH_CUT);
            if (!performGraphCut(processed_cloud, result.final_grid, result.shell_mesh)) {
                result.success = false;
                result.error_message = "图割优化失败";
                result.failed_stage = current_stage_;
                is_running_ = false;
                return false;
            }
        }
        
        // 检查是否需要停止
        if (should_stop_) {
            result.success = false;
            result.error_message = "用户请求停止";
            is_running_ = false;
            return false;
        }
        
        // 阶段4: 细节重建
        if (config_.enable_detail_reconstruction) {
            updateStage(ReconstructionStage::DETAIL_RECONSTRUCTION);
            if (!performDetailReconstruction(processed_cloud, result.final_grid, result.detail_result)) {
                result.success = false;
                result.error_message = "细节重建失败";
                result.failed_stage = current_stage_;
                is_running_ = false;
                return false;
            }
        }
        
        // 检查是否需要停止
        if (should_stop_) {
            result.success = false;
            result.error_message = "用户请求停止";
            is_running_ = false;
            return false;
        }
        
        // 阶段5: 融合合法化
        if (config_.enable_fusion_legalization) {
            updateStage(ReconstructionStage::FUSION_LEGALIZATION);
            if (!performFusionLegalization(result.shell_mesh, result.detail_result, result.fusion_result)) {
                result.success = false;
                result.error_message = "融合合法化失败";
                result.failed_stage = current_stage_;
                is_running_ = false;
                return false;
            }
            
            // 使用融合后的网格作为最终结果
            result.final_mesh = result.fusion_result.fused_mesh;
        } else {
            // 如果没有启用融合，直接使用外壳网格
            result.final_mesh = result.shell_mesh;
        }
        
        // 阶段6: 最终化
        updateStage(ReconstructionStage::FINALIZATION);
        
        // 计算最终质量分数
        result.overall_quality_score = evaluateReconstructionQuality(result, *processed_cloud);
        
        // 更新统计信息
        updateStatistics(result);
        
        // 保存中间结果
        if (config_.save_intermediate_results) {
            saveIntermediateResults(result);
        }
        
        updateStage(ReconstructionStage::COMPLETED);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.stats.total_time = std::chrono::duration<double>(end_time - start_time_).count();
        
        result.success = true;
        last_result_ = result;
        is_running_ = false;
        
        std::cout << "完整3D重建完成" << std::endl;
        std::cout << "总时间: " << result.stats.total_time << " 秒" << std::endl;
        std::cout << "最终质量分数: " << result.overall_quality_score << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "重建过程异常: " << e.what() << std::endl;
        result.success = false;
        result.error_message = std::string("异常: ") + e.what();
        result.failed_stage = current_stage_;
        is_running_ = false;
        return false;
    }
}

// ============================================================================
// 各阶段实现
// ============================================================================

bool MainReconstructor::performUDFBuilding(
    const PointCloudT::Ptr& input_cloud,
    GridT::Ptr& udf_grid) {
    
    std::cout << "执行UDF构建..." << std::endl;
    auto stage_start = std::chrono::high_resolution_clock::now();
    
    try {
        if (!udf_builder_) {
            std::cerr << "UDF构建器未初始化" << std::endl;
            return false;
        }
        
        // 执行UDF构建
        EnhancedUDFResult udf_result;
        if (!udf_builder_->buildUDF(input_cloud, udf_result)) {
            std::cerr << "UDF构建失败" << std::endl;
            return false;
        }
        
        udf_grid = udf_result.udf_grid;
        stats_.udf_voxels = udf_result.stats.total_voxels;
        
        auto stage_end = std::chrono::high_resolution_clock::now();
        stats_.udf_building_time = std::chrono::duration<double>(stage_end - stage_start).count();
        
        std::cout << "UDF构建完成，体素数: " << stats_.udf_voxels 
                  << ", 时间: " << stats_.udf_building_time << " 秒" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "UDF构建异常: " << e.what() << std::endl;
        return false;
    }
}

bool MainReconstructor::performGraphCut(
    const PointCloudT::Ptr& input_cloud,
    const GridT::Ptr& udf_grid,
    pcl::PolygonMesh& shell_mesh) {
    
    std::cout << "执行图割优化..." << std::endl;
    auto stage_start = std::chrono::high_resolution_clock::now();
    
    try {
        if (!graph_cut_integrator_) {
            std::cerr << "图割集成器未初始化" << std::endl;
            return false;
        }
        
        // 执行集成的图割优化
        UDFGraphCutResult graph_cut_result;
        if (!graph_cut_integrator_->performIntegratedGraphCut(
                input_cloud, udf_grid, graph_cut_result)) {
            std::cerr << "图割优化失败" << std::endl;
            return false;
        }
        
        shell_mesh = graph_cut_result.optimized_mesh;
        stats_.shell_vertices = static_cast<int>(shell_mesh.cloud.width * shell_mesh.cloud.height);
        stats_.shell_faces = static_cast<int>(shell_mesh.polygons.size());
        stats_.graph_cut_nodes = graph_cut_result.stats.total_nodes;
        
        auto stage_end = std::chrono::high_resolution_clock::now();
        stats_.graph_cut_time = std::chrono::duration<double>(stage_end - stage_start).count();
        
        std::cout << "图割优化完成，外壳面数: " << stats_.shell_faces 
                  << ", 时间: " << stats_.graph_cut_time << " 秒" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "图割优化异常: " << e.what() << std::endl;
        return false;
    }
}

bool MainReconstructor::performDetailReconstruction(
    const PointCloudT::Ptr& input_cloud,
    const GridT::Ptr& udf_grid,
    HybridReconstructionResult& detail_result) {
    
    std::cout << "执行细节重建..." << std::endl;
    auto stage_start = std::chrono::high_resolution_clock::now();
    
    try {
        if (!detail_integrator_) {
            std::cerr << "细节重建集成器未初始化" << std::endl;
            return false;
        }
        
        // 执行细节重建
        if (!detail_integrator_->performDetailReconstruction(
                input_cloud, udf_grid, detail_result)) {
            std::cerr << "细节重建失败" << std::endl;
            return false;
        }
        
        stats_.detail_clusters = static_cast<int>(detail_result.cluster_results.size());
        
        // 统计细节顶点和面数
        int total_detail_vertices = 0;
        int total_detail_faces = 0;
        for (const auto& cluster_result : detail_result.cluster_results) {
            pcl::PointCloud<pcl::PointXYZ> vertices;
            pcl::fromPCLPointCloud2(cluster_result.reconstructed_mesh.cloud, vertices);
            total_detail_vertices += static_cast<int>(vertices.size());
            total_detail_faces += static_cast<int>(cluster_result.reconstructed_mesh.polygons.size());
        }
        
        stats_.detail_vertices = total_detail_vertices;
        stats_.detail_faces = total_detail_faces;
        
        auto stage_end = std::chrono::high_resolution_clock::now();
        stats_.detail_reconstruction_time = std::chrono::duration<double>(stage_end - stage_start).count();
        
        std::cout << "细节重建完成，簇数: " << stats_.detail_clusters 
                  << ", 细节面数: " << stats_.detail_faces
                  << ", 时间: " << stats_.detail_reconstruction_time << " 秒" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "细节重建异常: " << e.what() << std::endl;
        return false;
    }
}

bool MainReconstructor::performFusionLegalization(
    const pcl::PolygonMesh& shell_mesh,
    const HybridReconstructionResult& detail_result,
    FusionLegalizationResult& fusion_result) {
    
    std::cout << "执行融合合法化..." << std::endl;
    auto stage_start = std::chrono::high_resolution_clock::now();
    
    try {
        if (!fusion_integrator_) {
            std::cerr << "融合合法化集成器未初始化" << std::endl;
            return false;
        }
        
        // 执行融合合法化
        if (!fusion_integrator_->performFusionAndLegalization(
                shell_mesh, detail_result, fusion_result)) {
            std::cerr << "融合合法化失败: " << fusion_result.error_message << std::endl;
            return false;
        }
        
        stats_.fusion_operations = 1;
        stats_.legalization_fixes = fusion_result.stats.fixed_violations;
        
        auto stage_end = std::chrono::high_resolution_clock::now();
        stats_.fusion_legalization_time = std::chrono::duration<double>(stage_end - stage_start).count();
        
        std::cout << "融合合法化完成，修复违反: " << stats_.legalization_fixes 
                  << ", 时间: " << stats_.fusion_legalization_time << " 秒" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "融合合法化异常: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// 渐进式重建
// ============================================================================

bool MainReconstructor::performProgressiveReconstruction(
    const PointCloudT::Ptr& input_cloud,
    MainReconstructorResult& result) {
    
    std::cout << "开始渐进式重建..." << std::endl;
    
    try {
        // 第一次完整重建
        if (!performReconstruction(input_cloud, result)) {
            return false;
        }
        
        // 迭代改进
        MainReconstructorResult previous_result = result;
        
        for (int iteration = 1; iteration < config_.max_global_iterations; ++iteration) {
            std::cout << "渐进式重建迭代 " << iteration + 1 << "..." << std::endl;
            
            // 检查收敛性
            if (checkConvergence(result, previous_result)) {
                stats_.converged = true;
                std::cout << "渐进式重建收敛，迭代次数: " << iteration << std::endl;
                break;
            }
            
            // 基于当前结果调整参数
            adjustParametersBasedOnQuality(result);
            
            // 重新执行关键阶段
            previous_result = result;
            if (!reprocessCriticalStages(input_cloud, result)) {
                std::cout << "迭代 " << iteration + 1 << " 失败，使用上一次结果" << std::endl;
                result = previous_result;
                break;
            }
            
            stats_.global_iterations = iteration + 1;
        }
        
        std::cout << "渐进式重建完成，总迭代次数: " << stats_.global_iterations << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "渐进式重建异常: " << e.what() << std::endl;
        return false;
    }
}

bool MainReconstructor::performIterativeImprovement(
    const PointCloudT::Ptr& input_cloud,
    MainReconstructorResult& result,
    double target_quality) {
    
    std::cout << "开始迭代质量改进，目标质量: " << target_quality << std::endl;
    
    try {
        for (int iteration = 0; iteration < config_.max_global_iterations; ++iteration) {
            double current_quality = evaluateReconstructionQuality(result, *input_cloud);
            
            std::cout << "迭代 " << iteration + 1 << ", 当前质量: " << current_quality << std::endl;
            
            if (current_quality >= target_quality) {
                stats_.converged = true;
                std::cout << "达到目标质量，迭代改进完成" << std::endl;
                break;
            }
            
            // 基于质量分析调整参数
            adjustParametersBasedOnQuality(result);
            
            // 重新执行关键阶段
            MainReconstructorResult improved_result;
            if (reprocessCriticalStages(input_cloud, improved_result)) {
                double improved_quality = evaluateReconstructionQuality(improved_result, *input_cloud);
                
                if (improved_quality > current_quality) {
                    result = improved_result;
                    std::cout << "质量改进: " << current_quality << " -> " << improved_quality << std::endl;
                } else {
                    std::cout << "质量未改进，保持当前结果" << std::endl;
                }
            }
            
            stats_.global_iterations = iteration + 1;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "迭代质量改进异常: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// 质量评估
// ============================================================================

double MainReconstructor::evaluateReconstructionQuality(
    const MainReconstructorResult& result,
    const PointCloudT& original_cloud) {
    
    try {
        if (!result.success || result.final_mesh.polygons.empty()) {
            return 0.0;
        }
        
        double quality_score = 0.0;
        
        // 1. 完整性评估 (40%)
        double completeness = evaluateCompleteness(result.final_mesh, original_cloud);
        quality_score += 0.4 * completeness;
        
        // 2. 准确性评估 (30%)
        double accuracy = evaluateAccuracy(result.final_mesh, original_cloud);
        quality_score += 0.3 * accuracy;
        
        // 3. 规整性评估 (20%)
        double regularity = evaluateRegularity(result.final_mesh);
        quality_score += 0.2 * regularity;
        
        // 4. 拓扑质量评估 (10%)
        double topology = evaluateTopology(result.final_mesh);
        quality_score += 0.1 * topology;
        
        return std::min(quality_score, 1.0);
        
    } catch (const std::exception& e) {
        std::cerr << "质量评估异常: " << e.what() << std::endl;
        return 0.0;
    }
}

double MainReconstructor::evaluateCompleteness(
    const pcl::PolygonMesh& mesh,
    const PointCloudT& original_cloud) {
    
    // 简化的完整性评估
    if (mesh.polygons.empty() || original_cloud.empty()) {
        return 0.0;
    }
    
    // 基于面数的简单评估
    double face_ratio = std::min(1.0, static_cast<double>(mesh.polygons.size()) / 1000.0);
    return face_ratio;
}

double MainReconstructor::evaluateAccuracy(
    const pcl::PolygonMesh& mesh,
    const PointCloudT& original_cloud) {
    
    // 简化的准确性评估
    if (mesh.polygons.empty() || original_cloud.empty()) {
        return 0.0;
    }
    
    // 简化假设：如果有合理数量的面，认为准确性较好
    return mesh.polygons.size() > 100 ? 0.8 : 0.5;
}

double MainReconstructor::evaluateRegularity(const pcl::PolygonMesh& mesh) {
    // 简化的规整性评估
    if (mesh.polygons.empty()) {
        return 0.0;
    }
    
    // 检查三角形面片的比例
    int triangle_count = 0;
    for (const auto& polygon : mesh.polygons) {
        if (polygon.vertices.size() == 3) {
            triangle_count++;
        }
    }
    
    double triangle_ratio = static_cast<double>(triangle_count) / mesh.polygons.size();
    return triangle_ratio; // 三角形比例越高，规整性越好
}

double MainReconstructor::evaluateTopology(const pcl::PolygonMesh& mesh) {
    // 简化的拓扑评估
    if (mesh.polygons.empty()) {
        return 0.0;
    }
    
    // 基本的拓扑检查：是否有有效的面片
    return 0.8; // 简化假设
}

// ============================================================================
// 辅助方法
// ============================================================================

bool MainReconstructor::validateInputData(const PointCloudT& input_cloud) {
    if (input_cloud.empty()) {
        std::cerr << "输入点云为空" << std::endl;
        return false;
    }
    
    if (input_cloud.size() < 100) {
        std::cerr << "输入点云点数太少: " << input_cloud.size() << std::endl;
        return false;
    }
    
    // 检查点的有效性
    int valid_points = 0;
    for (const auto& point : input_cloud) {
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
            valid_points++;
        }
    }
    
    if (valid_points < static_cast<int>(input_cloud.size()) * 0.8) {
        std::cerr << "有效点数太少: " << valid_points << "/" << input_cloud.size() << std::endl;
        return false;
    }
    
    std::cout << "输入数据验证通过，有效点数: " << valid_points << std::endl;
    return true;
}

bool MainReconstructor::preprocessPointCloud(
    const PointCloudT::Ptr& input_cloud,
    PointCloudT::Ptr& processed_cloud) {
    
    std::cout << "预处理点云..." << std::endl;
    
    try {
        processed_cloud = std::make_shared<PointCloudT>();
        
        // 简化的预处理：统计离群点移除
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(input_cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh(1.0);
        sor.filter(*processed_cloud);
        
        std::cout << "预处理完成，点数: " << input_cloud->size() 
                  << " -> " << processed_cloud->size() << std::endl;
        
        return !processed_cloud->empty();
        
    } catch (const std::exception& e) {
        std::cerr << "点云预处理异常: " << e.what() << std::endl;
        return false;
    }
}

bool MainReconstructor::checkConvergence(
    const MainReconstructorResult& current_result,
    const MainReconstructorResult& previous_result) {
    
    if (!current_result.success || !previous_result.success) {
        return false;
    }
    
    // 简化的收敛检查：质量分数变化
    double quality_change = std::abs(current_result.overall_quality_score - 
                                   previous_result.overall_quality_score);
    
    return quality_change < config_.convergence_threshold;
}

void MainReconstructor::adjustParametersBasedOnQuality(const MainReconstructorResult& result) {
    // 简化的参数调整
    if (result.overall_quality_score < 0.5) {
        // 质量较低，增加处理强度
        if (udf_builder_) {
            // 可以调整UDF参数
        }
        if (graph_cut_integrator_) {
            // 可以调整图割参数
        }
    }
}

bool MainReconstructor::reprocessCriticalStages(
    const PointCloudT::Ptr& input_cloud,
    MainReconstructorResult& result) {
    
    std::cout << "重新处理关键阶段..." << std::endl;
    
    // 简化实现：重新执行完整重建
    return performReconstruction(input_cloud, result);
}

void MainReconstructor::updateStage(ReconstructionStage new_stage) {
    current_stage_ = new_stage;
    
    // 记录阶段时间
    auto current_time = std::chrono::high_resolution_clock::now();
    if (stage_start_time_.time_since_epoch().count() > 0) {
        auto stage_duration = std::chrono::duration<double>(current_time - stage_start_time_).count();
        recordStageTime(current_stage_, stage_duration);
    }
    stage_start_time_ = current_time;
    
    std::cout << "进入阶段: " << static_cast<int>(new_stage) << std::endl;
}

void MainReconstructor::recordStageTime(ReconstructionStage stage, double time) {
    switch (stage) {
        case ReconstructionStage::INITIALIZATION:
            stats_.initialization_time = time;
            break;
        case ReconstructionStage::UDF_BUILDING:
            stats_.udf_building_time = time;
            break;
        case ReconstructionStage::GRAPH_CUT:
            stats_.graph_cut_time = time;
            break;
        case ReconstructionStage::DETAIL_RECONSTRUCTION:
            stats_.detail_reconstruction_time = time;
            break;
        case ReconstructionStage::FUSION_LEGALIZATION:
            stats_.fusion_legalization_time = time;
            break;
        case ReconstructionStage::FINALIZATION:
            stats_.finalization_time = time;
            break;
        default:
            break;
    }
}

void MainReconstructor::updateStatistics(MainReconstructorResult& result) {
    // 更新最终统计信息
    pcl::PointCloud<pcl::PointXYZ> final_vertices;
    pcl::fromPCLPointCloud2(result.final_mesh.cloud, final_vertices);
    
    result.stats.final_vertices = static_cast<int>(final_vertices.size());
    result.stats.final_faces = static_cast<int>(result.final_mesh.polygons.size());
    
    // 计算质量改进
    if (stats_.initial_quality > 0) {
        result.stats.overall_improvement = result.overall_quality_score - stats_.initial_quality;
    }
    
    // 复制统计信息
    result.stats = stats_;
}

void MainReconstructor::saveIntermediateResults(const MainReconstructorResult& result) {
    if (!config_.save_intermediate_results) {
        return;
    }
    
    try {
        std::string output_dir = config_.output_directory + "/intermediate";
        std::filesystem::create_directories(output_dir);
        
        // 保存外壳网格
        if (!result.shell_mesh.polygons.empty()) {
            std::string shell_path = output_dir + "/shell_mesh.ply";
            pcl::io::savePLYFile(shell_path, result.shell_mesh);
            std::cout << "外壳网格已保存: " << shell_path << std::endl;
        }
        
        // 保存最终网格
        if (!result.final_mesh.polygons.empty()) {
            std::string final_path = output_dir + "/final_mesh.ply";
            pcl::io::savePLYFile(final_path, result.final_mesh);
            std::cout << "最终网格已保存: " << final_path << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "中间结果保存异常: " << e.what() << std::endl;
    }
}

bool MainReconstructor::generateQualityReport(
    const MainReconstructorResult& result,
    const std::string& report_path) {
    
    try {
        std::ofstream report_file(report_path);
        
        if (!report_file.is_open()) {
            std::cerr << "无法创建质量报告文件: " << report_path << std::endl;
            return false;
        }
        
        report_file << "3D重建质量报告\n";
        report_file << "===============\n\n";
        
        report_file << "基本信息:\n";
        report_file << "  重建状态: " << (result.success ? "成功" : "失败") << "\n";
        report_file << "  总处理时间: " << result.stats.total_time << " 秒\n";
        report_file << "  整体质量分数: " << result.overall_quality_score << "\n\n";
        
        report_file << "数据统计:\n";
        report_file << "  输入点数: " << result.stats.input_points << "\n";
        report_file << "  外壳顶点数: " << result.stats.shell_vertices << "\n";
        report_file << "  外壳面数: " << result.stats.shell_faces << "\n";
        report_file << "  细节顶点数: " << result.stats.detail_vertices << "\n";
        report_file << "  细节面数: " << result.stats.detail_faces << "\n";
        report_file << "  最终顶点数: " << result.stats.final_vertices << "\n";
        report_file << "  最终面数: " << result.stats.final_faces << "\n\n";
        
        report_file << "处理统计:\n";
        report_file << "  UDF体素数: " << result.stats.udf_voxels << "\n";
        report_file << "  图割节点数: " << result.stats.graph_cut_nodes << "\n";
        report_file << "  细节簇数: " << result.stats.detail_clusters << "\n";
        report_file << "  融合操作数: " << result.stats.fusion_operations << "\n";
        report_file << "  合法化修复数: " << result.stats.legalization_fixes << "\n\n";
        
        report_file << "时间分解:\n";
        report_file << "  初始化时间: " << result.stats.initialization_time << " 秒\n";
        report_file << "  UDF构建时间: " << result.stats.udf_building_time << " 秒\n";
        report_file << "  图割时间: " << result.stats.graph_cut_time << " 秒\n";
        report_file << "  细节重建时间: " << result.stats.detail_reconstruction_time << " 秒\n";
        report_file << "  融合合法化时间: " << result.stats.fusion_legalization_time << " 秒\n";
        report_file << "  最终化时间: " << result.stats.finalization_time << " 秒\n\n";
        
        report_file << "迭代信息:\n";
        report_file << "  全局迭代次数: " << result.stats.global_iterations << "\n";
        report_file << "  是否收敛: " << (result.stats.converged ? "是" : "否") << "\n";
        report_file << "  质量改进: " << result.stats.overall_improvement << "\n\n";
        
        if (!result.warnings.empty()) {
            report_file << "警告信息:\n";
            for (const auto& warning : result.warnings) {
                report_file << "  - " << warning << "\n";
            }
            report_file << "\n";
        }
        
        if (!result.quality_issues.empty()) {
            report_file << "质量问题:\n";
            for (const auto& issue : result.quality_issues) {
                report_file << "  - " << issue << "\n";
            }
            report_file << "\n";
        }
        
        if (!result.recommendations.empty()) {
            report_file << "改进建议:\n";
            for (const auto& recommendation : result.recommendations) {
                report_file << "  - " << recommendation << "\n";
            }
            report_file << "\n";
        }
        
        report_file.close();
        
        std::cout << "质量报告已生成: " << report_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "质量报告生成异常: " << e.what() << std::endl;
        return false;
    }
}

bool MainReconstructor::saveReconstructionResult(
    const MainReconstructorResult& result,
    const std::string& output_path) {
    
    try {
        if (!result.success || result.final_mesh.polygons.empty()) {
            std::cerr << "无有效结果可保存" << std::endl;
            return false;
        }
        
        // 保存最终网格
        if (pcl::io::savePLYFile(output_path, result.final_mesh) == 0) {
            std::cout << "重建结果已保存: " << output_path << std::endl;
            return true;
        } else {
            std::cerr << "保存重建结果失败: " << output_path << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "保存重建结果异常: " << e.what() << std::endl;
        return false;
    }
}

double MainReconstructor::getProgressPercentage() const {
    switch (current_stage_) {
        case ReconstructionStage::INITIALIZATION:
            return 0.0;
        case ReconstructionStage::UDF_BUILDING:
            return 20.0;
        case ReconstructionStage::GRAPH_CUT:
            return 40.0;
        case ReconstructionStage::DETAIL_RECONSTRUCTION:
            return 60.0;
        case ReconstructionStage::FUSION_LEGALIZATION:
            return 80.0;
        case ReconstructionStage::FINALIZATION:
            return 95.0;
        case ReconstructionStage::COMPLETED:
            return 100.0;
        default:
            return 0.0;
    }
}

void MainReconstructor::reset() {
    current_stage_ = ReconstructionStage::INITIALIZATION;
    is_running_ = false;
    should_stop_ = false;
    stats_ = MainReconstructorStats{};
    last_result_ = MainReconstructorResult{};
}

void MainReconstructor::cleanup() {
    // 清理资源
    udf_builder_.reset();
    graph_cut_integrator_.reset();
    detail_integrator_.reset();
    fusion_integrator_.reset();
}

// ============================================================================
// MainReconstructorFactory实现
// ============================================================================

std::unique_ptr<MainReconstructor> 
MainReconstructorFactory::createStandardReconstructor() {
    MainReconstructorConfig config;
    
    // 标准配置：平衡质量和性能
    config.enable_udf_building = true;
    config.enable_graph_cut = true;
    config.enable_detail_reconstruction = true;
    config.enable_fusion_legalization = true;
    config.min_overall_quality = 0.6;
    config.max_global_iterations = 3;
    
    return std::make_unique<MainReconstructor>(config);
}

std::unique_ptr<MainReconstructor> 
MainReconstructorFactory::createHighQualityReconstructor() {
    MainReconstructorConfig config;
    
    // 高质量配置
    config.enable_udf_building = true;
    config.enable_graph_cut = true;
    config.enable_detail_reconstruction = true;
    config.enable_fusion_legalization = true;
    config.min_overall_quality = 0.8;
    config.max_global_iterations = 5;
    config.enable_progressive_reconstruction = true;
    
    return std::make_unique<MainReconstructor>(config);
}

std::unique_ptr<MainReconstructor> 
MainReconstructorFactory::createFastReconstructor() {
    MainReconstructorConfig config;
    
    // 快速配置
    config.enable_udf_building = true;
    config.enable_graph_cut = true;
    config.enable_detail_reconstruction = false; // 跳过细节重建
    config.enable_fusion_legalization = false;   // 跳过融合合法化
    config.min_overall_quality = 0.3;
    config.max_global_iterations = 1;
    config.enable_progressive_reconstruction = false;
    
    return std::make_unique<MainReconstructor>(config);
}

std::unique_ptr<MainReconstructor> 
MainReconstructorFactory::createMemoryOptimizedReconstructor() {
    MainReconstructorConfig config;
    
    // 内存优化配置
    config.enable_memory_optimization = true;
    config.save_intermediate_results = false;
    config.enable_detailed_logging = false;
    
    return std::make_unique<MainReconstructor>(config);
}

std::unique_ptr<MainReconstructor> 
MainReconstructorFactory::createArchitecturalReconstructor() {
    MainReconstructorConfig config;
    
    // 建筑专用配置
    config.enable_fusion_legalization = true;
    config.fusion_config.enable_plane_alignment = true;
    config.fusion_config.enable_line_projection = true;
    config.fusion_config.legalization_config.enforce_manhattan_world = true;
    config.fusion_config.legalization_config.snap_to_grid = true;
    
    return std::make_unique<MainReconstructor>(config);
}

std::unique_ptr<MainReconstructor> 
MainReconstructorFactory::createDebugReconstructor() {
    MainReconstructorConfig config;
    
    // 调试配置
    config.enable_debug_mode = true;
    config.enable_detailed_logging = true;
    config.save_intermediate_results = true;
    config.enable_performance_profiling = true;
    
    return std::make_unique<MainReconstructor>(config);
}

} // namespace recon


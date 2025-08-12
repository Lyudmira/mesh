/**
 * 增强版细节重建模块实现文件
 * 
 * 版本: 2.0 - 增强版本
 * 日期: 2025-08-12
 */

#include "enhanced_detail_reconstruction.h"
#include <pcl/filters/fast_bilateral.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/geometry.h>
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/extract_clusters.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace recon {

// ============================================================================
// EnhancedDetailReconstructor 实现
// ============================================================================

EnhancedDetailReconstructor::EnhancedDetailReconstructor(
    const EnhancedDetailReconstructionConfig& config) 
    : config_(config) {
    
    // 初始化KD树
    kdtree_.reset(new pcl::KdTreeFLANN<PointT>);
    
    // 设置线程数
    if (config_.performance.num_threads > 0) {
        #ifdef _OPENMP
        omp_set_num_threads(config_.performance.num_threads);
        #endif
    }
    
    if (config_.enable_debug_output) {
        std::cout << "增强版细节重建器初始化完成" << std::endl;
        std::cout << "  主要方法: ";
        switch (config_.primary_method) {
            case EnhancedDetailMethod::GP3_ENHANCED:
                std::cout << "增强GP3" << std::endl;
                break;
            case EnhancedDetailMethod::RIMLS_FULL:
                std::cout << "完整RIMLS" << std::endl;
                break;
            case EnhancedDetailMethod::POISSON_ADAPTIVE:
                std::cout << "自适应泊松" << std::endl;
                break;
            case EnhancedDetailMethod::HYBRID:
                std::cout << "混合方法" << std::endl;
                break;
        }
        std::cout << "  去噪方法: ";
        switch (config_.denoising_method) {
            case DenoisingMethod::BILATERAL:
                std::cout << "双边滤波" << std::endl;
                break;
            case DenoisingMethod::RIMLS_DENOISING:
                std::cout << "RIMLS去噪" << std::endl;
                break;
            case DenoisingMethod::WLOP:
                std::cout << "WLOP" << std::endl;
                break;
            case DenoisingMethod::ADAPTIVE:
                std::cout << "自适应去噪" << std::endl;
                break;
        }
        #ifdef _OPENMP
        std::cout << "  并行处理: 启用 (" << omp_get_max_threads() << " 线程)" << std::endl;
        #else
        std::cout << "  并行处理: 禁用" << std::endl;
        #endif
    }
}

EnhancedDetailReconstructor::~EnhancedDetailReconstructor() = default;

bool EnhancedDetailReconstructor::reconstructDetails(
    const pcl::PolygonMesh& shell_mesh,
    const PointCloudT::Ptr& original_cloud,
    pcl::PolygonMesh& detail_mesh) {
    
    // 重置统计信息
    stats_ = Statistics{};
    auto total_start = std::chrono::high_resolution_clock::now();
    
    if (!original_cloud || original_cloud->empty()) {
        std::cerr << "原始点云为空" << std::endl;
        return false;
    }
    
    stats_.original_points = static_cast<int>(original_cloud->size());
    
    if (config_.enable_debug_output) {
        std::cout


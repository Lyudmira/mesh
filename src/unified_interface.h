/**
 * 统一接口
 * 为所有重建模块提供清晰的统一接口
 * 避免多版本混淆，简化使用
 */

#ifndef UNIFIED_INTERFACE_H
#define UNIFIED_INTERFACE_H

// ============================================================================
// 推荐使用的核心模块（最新版本）
// ============================================================================

// UDF构建 - 使用增强版本
#include "udf_builder/enhanced_udf_builder.h"
using RecommendedUDFBuilder = EnhancedUDFBuilder;

// 图割优化 - 使用增强版本
#include "graph_cut/enhanced_graph_cut.h"
using RecommendedGraphCut = EnhancedGraphCut;

// 细节重建 - 使用新的细节层架构
#include "detail_layer/adaptive_detail_selector.h"
#include "detail_layer/hybrid_reconstructor.h"
#include "detail_layer/detail_reconstruction_integrator.h"
using RecommendedDetailSelector = AdaptiveDetailSelector;
using RecommendedDetailReconstructor = HybridReconstructor;
using RecommendedDetailIntegrator = DetailReconstructionIntegrator;

// 融合处理 - 使用新的融合合法化架构
#include "fusion/alpha_wrapping_fusion.h"
#include "fusion/legalization_processor.h"
#include "fusion/fusion_legalization_integrator.h"
using RecommendedFusion = AlphaWrappingFusion;
using RecommendedLegalizer = LegalizationProcessor;
using RecommendedFusionIntegrator = FusionLegalizationIntegrator;

// 集成器
#include "integration/udf_graphcut_integrator.h"
using RecommendedIntegrator = UDFGraphCutIntegrator;

// 主重建器
#include "main_reconstructor.h"
using RecommendedMainReconstructor = MainReconstructor;

// ============================================================================
// 已弃用的模块（保留兼容性，不推荐使用）
// ============================================================================

// 旧版UDF构建器
#include "udf_builder/udf_builder.h"
#include "udf_builder/optimized_udf_builder.h"
// 使用 RecommendedUDFBuilder 替代

// 旧版图割
#include "graph_cut/graph_cut.h"
#include "graph_cut/simple_graph_cut.h"
// 使用 RecommendedGraphCut 替代

// 旧版细节重建
#include "detail_reconstruction.h"
#include "enhanced_detail_reconstruction.h"
// 使用 RecommendedDetailIntegrator 替代

// 旧版融合
#include "fusion/mesh_fusion.h"
#include "fusion/enhanced_mesh_fusion.h"
// 使用 RecommendedFusionIntegrator 替代

// ============================================================================
// 简化的工厂接口
// ============================================================================

namespace recon {

/**
 * 统一重建器工厂
 * 提供简化的创建接口
 */
class UnifiedReconstructorFactory {
public:
    /**
     * 创建标准重建器（推荐）
     */
    static std::unique_ptr<RecommendedMainReconstructor> createStandard() {
        return MainReconstructorFactory::createStandardReconstructor();
    }
    
    /**
     * 创建高质量重建器
     */
    static std::unique_ptr<RecommendedMainReconstructor> createHighQuality() {
        return MainReconstructorFactory::createHighQualityReconstructor();
    }
    
    /**
     * 创建快速重建器
     */
    static std::unique_ptr<RecommendedMainReconstructor> createFast() {
        return MainReconstructorFactory::createFastReconstructor();
    }
    
    /**
     * 创建建筑专用重建器
     */
    static std::unique_ptr<RecommendedMainReconstructor> createArchitectural() {
        return MainReconstructorFactory::createArchitecturalReconstructor();
    }
};

/**
 * 简化的重建接口
 * 提供最简单的使用方式
 */
class SimpleReconstructor {
public:
    /**
     * 一键重建
     * @param input_cloud 输入点云
     * @param output_mesh 输出网格
     * @param quality_mode 质量模式：0=快速，1=标准，2=高质量
     * @return 是否成功
     */
    static bool reconstruct(
        const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& input_cloud,
        pcl::PolygonMesh& output_mesh,
        int quality_mode = 1
    );
    
    /**
     * 带质量报告的重建
     */
    static bool reconstructWithReport(
        const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& input_cloud,
        pcl::PolygonMesh& output_mesh,
        double& quality_score,
        std::string& report,
        int quality_mode = 1
    );
};

} // namespace recon

// ============================================================================
// 使用指南
// ============================================================================

/*
推荐使用方式：

1. 简单使用：
   #include "unified_interface.h"
   
   auto cloud = loadPointCloud("input.pcd");
   pcl::PolygonMesh mesh;
   bool success = SimpleReconstructor::reconstruct(cloud, mesh);

2. 标准使用：
   #include "unified_interface.h"
   
   auto reconstructor = UnifiedReconstructorFactory::createStandard();
   MainReconstructorResult result;
   bool success = reconstructor->performReconstruction(cloud, result);

3. 高级使用：
   #include "main_reconstructor.h"
   
   MainReconstructorConfig config;
   // 自定义配置...
   auto reconstructor = std::make_unique<MainReconstructor>(config);
   // 执行重建...

已弃用的模块：
- UDFBuilder, OptimizedUDFBuilder -> 使用 EnhancedUDFBuilder
- GraphCut, SimpleGraphCut -> 使用 EnhancedGraphCut  
- DetailReconstruction, EnhancedDetailReconstruction -> 使用 DetailReconstructionIntegrator
- MeshFusion, EnhancedMeshFusion -> 使用 FusionLegalizationIntegrator
*/

#endif // UNIFIED_INTERFACE_H


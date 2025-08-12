/**
 * 统一接口实现
 * 提供简化的重建接口
 */

#include "unified_interface.h"
#include <iostream>
#include <sstream>

namespace recon {

// ============================================================================
// SimpleReconstructor实现
// ============================================================================

bool SimpleReconstructor::reconstruct(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& input_cloud,
    pcl::PolygonMesh& output_mesh,
    int quality_mode) {
    
    try {
        std::cout << "开始简化重建，质量模式: " << quality_mode << std::endl;
        
        // 根据质量模式选择重建器
        std::unique_ptr<RecommendedMainReconstructor> reconstructor;
        
        switch (quality_mode) {
            case 0: // 快速模式
                reconstructor = UnifiedReconstructorFactory::createFast();
                break;
            case 2: // 高质量模式
                reconstructor = UnifiedReconstructorFactory::createHighQuality();
                break;
            default: // 标准模式
                reconstructor = UnifiedReconstructorFactory::createStandard();
                break;
        }
        
        if (!reconstructor) {
            std::cerr << "重建器创建失败" << std::endl;
            return false;
        }
        
        // 执行重建
        MainReconstructorResult result;
        bool success = reconstructor->performReconstruction(input_cloud, result);
        
        if (success) {
            output_mesh = result.final_mesh;
            std::cout << "简化重建成功，质量分数: " << result.overall_quality_score << std::endl;
        } else {
            std::cerr << "简化重建失败: " << result.error_message << std::endl;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "简化重建异常: " << e.what() << std::endl;
        return false;
    }
}

bool SimpleReconstructor::reconstructWithReport(
    const pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& input_cloud,
    pcl::PolygonMesh& output_mesh,
    double& quality_score,
    std::string& report,
    int quality_mode) {
    
    try {
        std::cout << "开始带报告的重建，质量模式: " << quality_mode << std::endl;
        
        // 根据质量模式选择重建器
        std::unique_ptr<RecommendedMainReconstructor> reconstructor;
        
        switch (quality_mode) {
            case 0: // 快速模式
                reconstructor = UnifiedReconstructorFactory::createFast();
                break;
            case 2: // 高质量模式
                reconstructor = UnifiedReconstructorFactory::createHighQuality();
                break;
            default: // 标准模式
                reconstructor = UnifiedReconstructorFactory::createStandard();
                break;
        }
        
        if (!reconstructor) {
            std::cerr << "重建器创建失败" << std::endl;
            return false;
        }
        
        // 执行重建
        MainReconstructorResult result;
        bool success = reconstructor->performReconstruction(input_cloud, result);
        
        if (success) {
            output_mesh = result.final_mesh;
            quality_score = result.overall_quality_score;
            
            // 生成简化报告
            std::ostringstream report_stream;
            report_stream << "重建报告\n";
            report_stream << "========\n\n";
            report_stream << "基本信息:\n";
            report_stream << "  状态: 成功\n";
            report_stream << "  质量分数: " << quality_score << "\n";
            report_stream << "  处理时间: " << result.stats.total_time << " 秒\n\n";
            
            report_stream << "数据统计:\n";
            report_stream << "  输入点数: " << result.stats.input_points << "\n";
            report_stream << "  最终顶点数: " << result.stats.final_vertices << "\n";
            report_stream << "  最终面数: " << result.stats.final_faces << "\n\n";
            
            if (result.stats.global_iterations > 0) {
                report_stream << "迭代信息:\n";
                report_stream << "  迭代次数: " << result.stats.global_iterations << "\n";
                report_stream << "  是否收敛: " << (result.stats.converged ? "是" : "否") << "\n\n";
            }
            
            if (!result.warnings.empty()) {
                report_stream << "警告:\n";
                for (const auto& warning : result.warnings) {
                    report_stream << "  - " << warning << "\n";
                }
                report_stream << "\n";
            }
            
            report = report_stream.str();
            
            std::cout << "带报告的重建成功" << std::endl;
        } else {
            quality_score = 0.0;
            
            std::ostringstream report_stream;
            report_stream << "重建报告\n";
            report_stream << "========\n\n";
            report_stream << "状态: 失败\n";
            report_stream << "错误: " << result.error_message << "\n";
            report_stream << "失败阶段: " << static_cast<int>(result.failed_stage) << "\n";
            
            report = report_stream.str();
            
            std::cerr << "带报告的重建失败: " << result.error_message << std::endl;
        }
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "带报告的重建异常: " << e.what() << std::endl;
        quality_score = 0.0;
        report = std::string("异常: ") + e.what();
        return false;
    }
}

} // namespace recon


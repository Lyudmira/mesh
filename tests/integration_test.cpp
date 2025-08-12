/**
 * 集成测试
 * 测试完整的3D重建系统功能
 * 
 * 版本: 1.0
 * 日期: 2025-08-12
 */

#include "../src/main_reconstructor.h"
#include <gtest/gtest.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <filesystem>
#include <iostream>

namespace recon {
namespace test {

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建测试输出目录
        test_output_dir_ = "./test_output";
        std::filesystem::create_directories(test_output_dir_);
        
        // 初始化OpenVDB
        openvdb::initialize();
        
        // 创建测试点云
        createTestPointCloud();
        
        std::cout << "集成测试环境初始化完成" << std::endl;
    }
    
    void TearDown() override {
        // 清理测试文件
        if (std::filesystem::exists(test_output_dir_)) {
            std::filesystem::remove_all(test_output_dir_);
        }
        
        std::cout << "集成测试环境清理完成" << std::endl;
    }
    
    void createTestPointCloud() {
        test_cloud_ = std::make_shared<MainReconstructor::PointCloudT>();
        
        // 创建一个简单的立方体点云用于测试
        const int points_per_side = 50;
        const float cube_size = 2.0f;
        const float step = cube_size / points_per_side;
        
        // 生成立方体表面的点
        for (int i = 0; i < points_per_side; ++i) {
            for (int j = 0; j < points_per_side; ++j) {
                float x = i * step - cube_size / 2;
                float y = j * step - cube_size / 2;
                
                // 底面和顶面
                addTestPoint(x, y, -cube_size / 2, 0, 0, -1, 128, 128, 128);
                addTestPoint(x, y, cube_size / 2, 0, 0, 1, 128, 128, 128);
                
                // 前面和后面
                addTestPoint(x, -cube_size / 2, y, 0, -1, 0, 128, 128, 128);
                addTestPoint(x, cube_size / 2, y, 0, 1, 0, 128, 128, 128);
                
                // 左面和右面
                addTestPoint(-cube_size / 2, x, y, -1, 0, 0, 128, 128, 128);
                addTestPoint(cube_size / 2, x, y, 1, 0, 0, 128, 128, 128);
            }
        }
        
        std::cout << "创建测试点云，点数: " << test_cloud_->size() << std::endl;
    }
    
    void addTestPoint(float x, float y, float z, float nx, float ny, float nz,
                     uint8_t r, uint8_t g, uint8_t b) {
        MainReconstructor::PointT point;
        point.x = x;
        point.y = y;
        point.z = z;
        point.normal_x = nx;
        point.normal_y = ny;
        point.normal_z = nz;
        point.r = r;
        point.g = g;
        point.b = b;
        point.a = 255;
        
        test_cloud_->push_back(point);
    }
    
    bool saveTestResult(const MainReconstructorResult& result, const std::string& test_name) {
        try {
            std::string mesh_path = test_output_dir_ + "/" + test_name + "_result.ply";
            
            if (pcl::io::savePLYFile(mesh_path, result.final_mesh) == 0) {
                std::cout << "测试结果已保存: " << mesh_path << std::endl;
                return true;
            } else {
                std::cerr << "保存测试结果失败: " << mesh_path << std::endl;
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "保存测试结果异常: " << e.what() << std::endl;
            return false;
        }
    }
    
    std::string test_output_dir_;
    MainReconstructor::PointCloudT::Ptr test_cloud_;
};

// ============================================================================
// 基础功能测试
// ============================================================================

TEST_F(IntegrationTest, TestMainReconstructorCreation) {
    std::cout << "测试主重建器创建..." << std::endl;
    
    // 测试默认构造
    EXPECT_NO_THROW({
        auto reconstructor = std::make_unique<MainReconstructor>();
        EXPECT_TRUE(reconstructor != nullptr);
    });
    
    // 测试配置构造
    EXPECT_NO_THROW({
        MainReconstructorConfig config;
        config.enable_debug_mode = true;
        config.enable_detailed_logging = true;
        
        auto reconstructor = std::make_unique<MainReconstructor>(config);
        EXPECT_TRUE(reconstructor != nullptr);
        EXPECT_TRUE(reconstructor->getConfig().enable_debug_mode);
    });
    
    std::cout << "主重建器创建测试通过" << std::endl;
}

TEST_F(IntegrationTest, TestFactoryMethods) {
    std::cout << "测试工厂方法..." << std::endl;
    
    // 测试标准重建器
    EXPECT_NO_THROW({
        auto reconstructor = MainReconstructorFactory::createStandardReconstructor();
        EXPECT_TRUE(reconstructor != nullptr);
    });
    
    // 测试高质量重建器
    EXPECT_NO_THROW({
        auto reconstructor = MainReconstructorFactory::createHighQualityReconstructor();
        EXPECT_TRUE(reconstructor != nullptr);
    });
    
    // 测试快速重建器
    EXPECT_NO_THROW({
        auto reconstructor = MainReconstructorFactory::createFastReconstructor();
        EXPECT_TRUE(reconstructor != nullptr);
    });
    
    // 测试调试重建器
    EXPECT_NO_THROW({
        auto reconstructor = MainReconstructorFactory::createDebugReconstructor();
        EXPECT_TRUE(reconstructor != nullptr);
    });
    
    std::cout << "工厂方法测试通过" << std::endl;
}

// ============================================================================
// 完整重建测试
// ============================================================================

TEST_F(IntegrationTest, TestCompleteReconstruction) {
    std::cout << "测试完整重建流程..." << std::endl;
    
    // 创建标准重建器
    auto reconstructor = MainReconstructorFactory::createStandardReconstructor();
    ASSERT_TRUE(reconstructor != nullptr);
    
    // 执行重建
    MainReconstructorResult result;
    bool success = reconstructor->performReconstruction(test_cloud_, result);
    
    // 验证结果
    EXPECT_TRUE(success);
    EXPECT_TRUE(result.success);
    EXPECT_GT(result.final_mesh.polygons.size(), 0);
    EXPECT_GT(result.stats.final_vertices, 0);
    EXPECT_GT(result.stats.final_faces, 0);
    EXPECT_GE(result.overall_quality_score, 0.0);
    EXPECT_LE(result.overall_quality_score, 1.0);
    
    // 保存测试结果
    EXPECT_TRUE(saveTestResult(result, "complete_reconstruction"));
    
    // 验证统计信息
    const auto& stats = result.stats;
    EXPECT_GT(stats.total_time, 0.0);
    EXPECT_EQ(stats.input_points, static_cast<int>(test_cloud_->size()));
    EXPECT_TRUE(stats.converged || stats.global_iterations > 0);
    
    std::cout << "完整重建测试通过" << std::endl;
    std::cout << "  输入点数: " << stats.input_points << std::endl;
    std::cout << "  最终顶点数: " << stats.final_vertices << std::endl;
    std::cout << "  最终面数: " << stats.final_faces << std::endl;
    std::cout << "  总时间: " << stats.total_time << " 秒" << std::endl;
    std::cout << "  质量分数: " << result.overall_quality_score << std::endl;
}

TEST_F(IntegrationTest, TestHighQualityReconstruction) {
    std::cout << "测试高质量重建..." << std::endl;
    
    // 创建高质量重建器
    auto reconstructor = MainReconstructorFactory::createHighQualityReconstructor();
    ASSERT_TRUE(reconstructor != nullptr);
    
    // 执行重建
    MainReconstructorResult result;
    bool success = reconstructor->performReconstruction(test_cloud_, result);
    
    // 验证结果
    EXPECT_TRUE(success);
    EXPECT_TRUE(result.success);
    EXPECT_GT(result.final_mesh.polygons.size(), 0);
    
    // 高质量重建应该有更好的质量分数
    EXPECT_GE(result.overall_quality_score, 0.3);
    
    // 保存测试结果
    EXPECT_TRUE(saveTestResult(result, "high_quality_reconstruction"));
    
    std::cout << "高质量重建测试通过" << std::endl;
    std::cout << "  质量分数: " << result.overall_quality_score << std::endl;
}

TEST_F(IntegrationTest, TestFastReconstruction) {
    std::cout << "测试快速重建..." << std::endl;
    
    // 创建快速重建器
    auto reconstructor = MainReconstructorFactory::createFastReconstructor();
    ASSERT_TRUE(reconstructor != nullptr);
    
    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 执行重建
    MainReconstructorResult result;
    bool success = reconstructor->performReconstruction(test_cloud_, result);
    
    // 记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
    
    // 验证结果
    EXPECT_TRUE(success);
    EXPECT_TRUE(result.success);
    EXPECT_GT(result.final_mesh.polygons.size(), 0);
    
    // 快速重建应该更快
    EXPECT_LT(elapsed_time, 60.0); // 应该在60秒内完成
    
    // 保存测试结果
    EXPECT_TRUE(saveTestResult(result, "fast_reconstruction"));
    
    std::cout << "快速重建测试通过" << std::endl;
    std::cout << "  重建时间: " << elapsed_time << " 秒" << std::endl;
}

// ============================================================================
// 渐进式重建测试
// ============================================================================

TEST_F(IntegrationTest, TestProgressiveReconstruction) {
    std::cout << "测试渐进式重建..." << std::endl;
    
    // 创建标准重建器
    auto reconstructor = MainReconstructorFactory::createStandardReconstructor();
    ASSERT_TRUE(reconstructor != nullptr);
    
    // 执行渐进式重建
    MainReconstructorResult result;
    bool success = reconstructor->performProgressiveReconstruction(test_cloud_, result);
    
    // 验证结果
    EXPECT_TRUE(success);
    EXPECT_TRUE(result.success);
    EXPECT_GT(result.final_mesh.polygons.size(), 0);
    
    // 渐进式重建应该有迭代信息
    EXPECT_GT(result.stats.global_iterations, 0);
    
    // 保存测试结果
    EXPECT_TRUE(saveTestResult(result, "progressive_reconstruction"));
    
    std::cout << "渐进式重建测试通过" << std::endl;
    std::cout << "  全局迭代次数: " << result.stats.global_iterations << std::endl;
    std::cout << "  是否收敛: " << (result.stats.converged ? "是" : "否") << std::endl;
}

// ============================================================================
// 质量评估测试
// ============================================================================

TEST_F(IntegrationTest, TestQualityEvaluation) {
    std::cout << "测试质量评估..." << std::endl;
    
    // 创建重建器
    auto reconstructor = MainReconstructorFactory::createStandardReconstructor();
    ASSERT_TRUE(reconstructor != nullptr);
    
    // 执行重建
    MainReconstructorResult result;
    bool success = reconstructor->performReconstruction(test_cloud_, result);
    ASSERT_TRUE(success);
    
    // 测试质量评估
    double quality_score = reconstructor->evaluateReconstructionQuality(result, *test_cloud_);
    
    EXPECT_GE(quality_score, 0.0);
    EXPECT_LE(quality_score, 1.0);
    EXPECT_NEAR(quality_score, result.overall_quality_score, 0.1);
    
    std::cout << "质量评估测试通过" << std::endl;
    std::cout << "  评估质量分数: " << quality_score << std::endl;
}

TEST_F(IntegrationTest, TestQualityReport) {
    std::cout << "测试质量报告生成..." << std::endl;
    
    // 创建重建器
    auto reconstructor = MainReconstructorFactory::createStandardReconstructor();
    ASSERT_TRUE(reconstructor != nullptr);
    
    // 执行重建
    MainReconstructorResult result;
    bool success = reconstructor->performReconstruction(test_cloud_, result);
    ASSERT_TRUE(success);
    
    // 生成质量报告
    std::string report_path = test_output_dir_ + "/quality_report.txt";
    bool report_success = reconstructor->generateQualityReport(result, report_path);
    
    EXPECT_TRUE(report_success);
    EXPECT_TRUE(std::filesystem::exists(report_path));
    
    // 检查报告文件大小
    auto file_size = std::filesystem::file_size(report_path);
    EXPECT_GT(file_size, 0);
    
    std::cout << "质量报告生成测试通过" << std::endl;
    std::cout << "  报告文件: " << report_path << std::endl;
    std::cout << "  文件大小: " << file_size << " 字节" << std::endl;
}

// ============================================================================
// 错误处理测试
// ============================================================================

TEST_F(IntegrationTest, TestEmptyPointCloudHandling) {
    std::cout << "测试空点云处理..." << std::endl;
    
    // 创建空点云
    auto empty_cloud = std::make_shared<MainReconstructor::PointCloudT>();
    
    // 创建重建器
    auto reconstructor = MainReconstructorFactory::createStandardReconstructor();
    ASSERT_TRUE(reconstructor != nullptr);
    
    // 尝试重建空点云
    MainReconstructorResult result;
    bool success = reconstructor->performReconstruction(empty_cloud, result);
    
    // 应该失败但不崩溃
    EXPECT_FALSE(success);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
    
    std::cout << "空点云处理测试通过" << std::endl;
    std::cout << "  错误信息: " << result.error_message << std::endl;
}

TEST_F(IntegrationTest, TestInvalidPointCloudHandling) {
    std::cout << "测试无效点云处理..." << std::endl;
    
    // 创建包含无效点的点云
    auto invalid_cloud = std::make_shared<MainReconstructor::PointCloudT>();
    
    MainReconstructor::PointT invalid_point;
    invalid_point.x = std::numeric_limits<float>::quiet_NaN();
    invalid_point.y = std::numeric_limits<float>::quiet_NaN();
    invalid_point.z = std::numeric_limits<float>::quiet_NaN();
    invalid_point.normal_x = 0;
    invalid_point.normal_y = 0;
    invalid_point.normal_z = 0;
    
    for (int i = 0; i < 100; ++i) {
        invalid_cloud->push_back(invalid_point);
    }
    
    // 创建重建器
    auto reconstructor = MainReconstructorFactory::createStandardReconstructor();
    ASSERT_TRUE(reconstructor != nullptr);
    
    // 尝试重建无效点云
    MainReconstructorResult result;
    bool success = reconstructor->performReconstruction(invalid_cloud, result);
    
    // 应该失败但不崩溃
    EXPECT_FALSE(success);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error_message.empty());
    
    std::cout << "无效点云处理测试通过" << std::endl;
    std::cout << "  错误信息: " << result.error_message << std::endl;
}

// ============================================================================
// 性能测试
// ============================================================================

TEST_F(IntegrationTest, TestPerformanceBenchmark) {
    std::cout << "测试性能基准..." << std::endl;
    
    // 创建不同大小的点云进行性能测试
    std::vector<int> point_counts = {1000, 5000, 10000};
    
    for (int point_count : point_counts) {
        std::cout << "测试 " << point_count << " 个点的重建性能..." << std::endl;
        
        // 创建指定大小的点云
        auto perf_cloud = std::make_shared<MainReconstructor::PointCloudT>();
        
        // 简单的球形点云
        for (int i = 0; i < point_count; ++i) {
            float theta = 2.0f * M_PI * i / point_count;
            float phi = M_PI * (i % 100) / 100.0f;
            
            MainReconstructor::PointT point;
            point.x = std::sin(phi) * std::cos(theta);
            point.y = std::sin(phi) * std::sin(theta);
            point.z = std::cos(phi);
            point.normal_x = point.x;
            point.normal_y = point.y;
            point.normal_z = point.z;
            point.r = 128;
            point.g = 128;
            point.b = 128;
            point.a = 255;
            
            perf_cloud->push_back(point);
        }
        
        // 创建快速重建器进行性能测试
        auto reconstructor = MainReconstructorFactory::createFastReconstructor();
        ASSERT_TRUE(reconstructor != nullptr);
        
        // 记录时间
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 执行重建
        MainReconstructorResult result;
        bool success = reconstructor->performReconstruction(perf_cloud, result);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_time = std::chrono::duration<double>(end_time - start_time).count();
        
        // 验证结果
        EXPECT_TRUE(success);
        
        std::cout << "  点数: " << point_count 
                  << ", 时间: " << elapsed_time << " 秒"
                  << ", 速度: " << point_count / elapsed_time << " 点/秒" << std::endl;
        
        // 性能要求：应该能在合理时间内完成
        EXPECT_LT(elapsed_time, 120.0); // 2分钟内完成
    }
    
    std::cout << "性能基准测试通过" << std::endl;
}

// ============================================================================
// 重建器管理器测试
// ============================================================================

TEST_F(IntegrationTest, TestReconstructorManager) {
    std::cout << "测试重建器管理器..." << std::endl;
    
    ReconstructorManager manager;
    
    // 添加多个重建任务
    MainReconstructorConfig config;
    config.enable_debug_mode = false;
    
    int task1 = manager.addReconstructionTask(test_cloud_, config, "task1");
    int task2 = manager.addReconstructionTask(test_cloud_, config, "task2");
    
    EXPECT_GE(task1, 0);
    EXPECT_GE(task2, 0);
    EXPECT_NE(task1, task2);
    EXPECT_EQ(manager.getActiveTaskCount(), 2);
    
    // 执行所有任务
    bool success = manager.executeAllTasks();
    EXPECT_TRUE(success);
    
    // 检查任务状态
    EXPECT_EQ(manager.getTaskStatus(task1), ReconstructionStage::COMPLETED);
    EXPECT_EQ(manager.getTaskStatus(task2), ReconstructionStage::COMPLETED);
    
    // 获取任务结果
    MainReconstructorResult result1, result2;
    EXPECT_TRUE(manager.getTaskResult(task1, result1));
    EXPECT_TRUE(manager.getTaskResult(task2, result2));
    
    EXPECT_TRUE(result1.success);
    EXPECT_TRUE(result2.success);
    
    // 清理任务
    manager.cleanupCompletedTasks();
    EXPECT_EQ(manager.getActiveTaskCount(), 0);
    
    std::cout << "重建器管理器测试通过" << std::endl;
}

} // namespace test
} // namespace recon

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "开始集成测试..." << std::endl;
    
    ::testing::InitGoogleTest(&argc, argv);
    
    // 设置测试环境
    ::testing::Environment* const env = ::testing::AddGlobalTestEnvironment(
        new ::testing::Environment());
    
    int result = RUN_ALL_TESTS();
    
    std::cout << "集成测试完成，结果: " << (result == 0 ? "通过" : "失败") << std::endl;
    
    return result;
}


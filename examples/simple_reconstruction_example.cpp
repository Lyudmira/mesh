/**
 * 简单重建示例
 * 演示如何使用主重建器进行3D重建
 */

#include "recon/src/main_reconstructor.h"
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <iostream>
#include <memory>

using namespace recon;

// 创建测试点云
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr createTestCube() {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
    
    // 创建一个简单的立方体点云
    const int points_per_side = 30;
    const float cube_size = 1.0f;
    const float step = cube_size / points_per_side;
    
    std::cout << "创建测试立方体点云..." << std::endl;
    
    // 生成立方体表面的点
    for (int i = 0; i < points_per_side; ++i) {
        for (int j = 0; j < points_per_side; ++j) {
            float x = i * step - cube_size / 2;
            float y = j * step - cube_size / 2;
            
            // 添加立方体的6个面
            pcl::PointXYZRGBNormal point;
            
            // 底面 (z = -0.5)
            point.x = x; point.y = y; point.z = -cube_size / 2;
            point.normal_x = 0; point.normal_y = 0; point.normal_z = -1;
            point.r = 128; point.g = 128; point.b = 128; point.a = 255;
            cloud->push_back(point);
            
            // 顶面 (z = 0.5)
            point.x = x; point.y = y; point.z = cube_size / 2;
            point.normal_x = 0; point.normal_y = 0; point.normal_z = 1;
            cloud->push_back(point);
            
            // 前面 (y = -0.5)
            point.x = x; point.y = -cube_size / 2; point.z = y;
            point.normal_x = 0; point.normal_y = -1; point.normal_z = 0;
            cloud->push_back(point);
            
            // 后面 (y = 0.5)
            point.x = x; point.y = cube_size / 2; point.z = y;
            point.normal_x = 0; point.normal_y = 1; point.normal_z = 0;
            cloud->push_back(point);
            
            // 左面 (x = -0.5)
            point.x = -cube_size / 2; point.y = x; point.z = y;
            point.normal_x = -1; point.normal_y = 0; point.normal_z = 0;
            cloud->push_back(point);
            
            // 右面 (x = 0.5)
            point.x = cube_size / 2; point.y = x; point.z = y;
            point.normal_x = 1; point.normal_y = 0; point.normal_z = 0;
            cloud->push_back(point);
        }
    }
    
    std::cout << "测试点云创建完成，点数: " << cloud->size() << std::endl;
    return cloud;
}

int main(int argc, char** argv) {
    std::cout << "=== 简单3D重建示例 ===" << std::endl;
    
    try {
        // 初始化OpenVDB
        openvdb::initialize();
        
        // 创建或加载点云
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr input_cloud;
        
        if (argc > 1) {
            // 从文件加载点云
            std::string input_file = argv[1];
            std::cout << "从文件加载点云: " << input_file << std::endl;
            
            input_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
            
            if (input_file.ends_with(".pcd")) {
                if (pcl::io::loadPCDFile(input_file, *input_cloud) == -1) {
                    std::cerr << "无法加载PCD文件: " << input_file << std::endl;
                    return -1;
                }
            } else if (input_file.ends_with(".ply")) {
                if (pcl::io::loadPLYFile(input_file, *input_cloud) == -1) {
                    std::cerr << "无法加载PLY文件: " << input_file << std::endl;
                    return -1;
                }
            } else {
                std::cerr << "不支持的文件格式: " << input_file << std::endl;
                return -1;
            }
            
            std::cout << "点云加载完成，点数: " << input_cloud->size() << std::endl;
        } else {
            // 创建测试点云
            input_cloud = createTestCube();
        }
        
        if (input_cloud->empty()) {
            std::cerr << "输入点云为空" << std::endl;
            return -1;
        }
        
        // 创建重建器
        std::cout << "创建标准重建器..." << std::endl;
        auto reconstructor = MainReconstructorFactory::createStandardReconstructor();
        
        if (!reconstructor) {
            std::cerr << "重建器创建失败" << std::endl;
            return -1;
        }
        
        // 执行重建
        std::cout << "开始重建..." << std::endl;
        MainReconstructorResult result;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        bool success = reconstructor->performReconstruction(input_cloud, result);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        // 输出结果
        if (success) {
            std::cout << "\n=== 重建成功 ===" << std::endl;
            std::cout << "总时间: " << total_time << " 秒" << std::endl;
            std::cout << "质量分数: " << result.overall_quality_score << std::endl;
            std::cout << "最终顶点数: " << result.stats.final_vertices << std::endl;
            std::cout << "最终面数: " << result.stats.final_faces << std::endl;
            
            // 保存结果
            std::string output_file = argc > 2 ? argv[2] : "reconstruction_result.ply";
            if (reconstructor->saveReconstructionResult(result, output_file)) {
                std::cout << "结果已保存到: " << output_file << std::endl;
            }
            
            // 生成质量报告
            std::string report_file = "quality_report.txt";
            if (reconstructor->generateQualityReport(result, report_file)) {
                std::cout << "质量报告已保存到: " << report_file << std::endl;
            }
            
        } else {
            std::cout << "\n=== 重建失败 ===" << std::endl;
            std::cout << "错误信息: " << result.error_message << std::endl;
            std::cout << "失败阶段: " << static_cast<int>(result.failed_stage) << std::endl;
            
            if (!result.warnings.empty()) {
                std::cout << "警告信息:" << std::endl;
                for (const auto& warning : result.warnings) {
                    std::cout << "  - " << warning << std::endl;
                }
            }
        }
        
        return success ? 0 : -1;
        
    } catch (const std::exception& e) {
        std::cerr << "程序异常: " << e.what() << std::endl;
        return -1;
    }
}


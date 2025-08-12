/**
 * 最小化测试程序
 * 验证基本编译和运行
 */

#include <iostream>
#include <memory>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/ply_io.h>
#include <openvdb/openvdb.h>

// 使用最小化的图割实现
#include "recon/src/graph_cut/minimal_graph_cut.h"

using namespace recon;

// 创建简单的测试点云
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr createSimpleTestCloud() {
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
    
    std::cout << "创建简单测试点云..." << std::endl;
    
    // 创建一个简单的平面点云
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            pcl::PointXYZRGBNormal point;
            point.x = i * 0.1f;
            point.y = j * 0.1f;
            point.z = 0.0f;
            point.normal_x = 0.0f;
            point.normal_y = 0.0f;
            point.normal_z = 1.0f;
            point.r = 128;
            point.g = 128;
            point.b = 128;
            point.a = 255;
            cloud->push_back(point);
        }
    }
    
    std::cout << "测试点云创建完成，点数: " << cloud->size() << std::endl;
    return cloud;
}

// 创建简单的测试网格
openvdb::FloatGrid::Ptr createSimpleTestGrid() {
    std::cout << "创建简单测试网格..." << std::endl;
    
    auto grid = openvdb::FloatGrid::create();
    auto accessor = grid->getAccessor();
    
    // 创建一个简单的立方体区域
    for (int x = 0; x < 10; ++x) {
        for (int y = 0; y < 10; ++y) {
            for (int z = 0; z < 10; ++z) {
                openvdb::Coord coord(x, y, z);
                
                // 计算到中心的距离
                float center_x = 5.0f;
                float center_y = 5.0f;
                float center_z = 5.0f;
                
                float distance = std::sqrt(
                    (x - center_x) * (x - center_x) +
                    (y - center_y) * (y - center_y) +
                    (z - center_z) * (z - center_z)
                );
                
                // 设置UDF值：内部为负，外部为正
                float udf_value = distance - 3.0f; // 半径为3的球
                accessor.setValue(coord, udf_value);
            }
        }
    }
    
    std::cout << "测试网格创建完成" << std::endl;
    return grid;
}

int main() {
    std::cout << "=== 最小化测试程序 ===" << std::endl;
    
    try {
        // 初始化OpenVDB
        openvdb::initialize();
        std::cout << "OpenVDB初始化完成" << std::endl;
        
        // 创建测试数据
        auto test_cloud = createSimpleTestCloud();
        auto test_grid = createSimpleTestGrid();
        
        if (!test_cloud || test_cloud->empty()) {
            std::cerr << "测试点云创建失败" << std::endl;
            return -1;
        }
        
        if (!test_grid) {
            std::cerr << "测试网格创建失败" << std::endl;
            return -1;
        }
        
        // 创建最小化图割求解器
        std::cout << "创建最小化图割求解器..." << std::endl;
        auto solver = MinimalGraphCutFactory::createStandard();
        
        if (!solver) {
            std::cerr << "图割求解器创建失败" << std::endl;
            return -1;
        }
        
        // 执行图割求解
        std::cout << "执行图割求解..." << std::endl;
        pcl::PolygonMesh result_mesh;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        bool success = solver->solve(test_cloud, test_grid, result_mesh);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        double total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        // 输出结果
        if (success) {
            std::cout << "\n=== 测试成功 ===" << std::endl;
            std::cout << "处理时间: " << total_time << " 秒" << std::endl;
            std::cout << "结果顶点数: " << result_mesh.cloud.width * result_mesh.cloud.height << std::endl;
            std::cout << "结果面数: " << result_mesh.polygons.size() << std::endl;
            
            // 保存结果
            std::string output_file = "minimal_test_result.ply";
            if (pcl::io::savePLYFile(output_file, result_mesh) == 0) {
                std::cout << "结果已保存到: " << output_file << std::endl;
            } else {
                std::cout << "结果保存失败" << std::endl;
            }
            
        } else {
            std::cout << "\n=== 测试失败 ===" << std::endl;
        }
        
        return success ? 0 : -1;
        
    } catch (const std::exception& e) {
        std::cerr << "程序异常: " << e.what() << std::endl;
        return -1;
    }
}


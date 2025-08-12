#include <iostream>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

int main() {
    std::cout << "=== 简单测试程序 ===" << std::endl;
    
    try {
        // 初始化OpenVDB
        openvdb::initialize();
        std::cout << "OpenVDB初始化成功" << std::endl;
        
        // 创建简单点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        cloud->push_back(pcl::PointXYZ(0, 0, 0));
        cloud->push_back(pcl::PointXYZ(1, 0, 0));
        cloud->push_back(pcl::PointXYZ(0, 1, 0));
        std::cout << "点云创建成功，点数: " << cloud->size() << std::endl;
        
        // 创建简单网格
        auto grid = openvdb::FloatGrid::create();
        auto accessor = grid->getAccessor();
        accessor.setValue(openvdb::Coord(0, 0, 0), 1.0f);
        std::cout << "网格创建成功" << std::endl;
        
        std::cout << "=== 测试成功 ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "测试失败: " << e.what() << std::endl;
        return -1;
    }
}

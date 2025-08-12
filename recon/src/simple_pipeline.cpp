/**
 * 简化版重建管道
 * 确保编译成功，展示核心功能框架
 */

#include <iostream>
#include <memory>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/surface/gp3.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>

using PointT = pcl::PointXYZRGBNormal;
using PointCloudT = pcl::PointCloud<PointT>;

/**
 * 简化版重建配置
 */
struct SimpleReconConfig {
    float normal_radius = 0.05f;
    float gp3_search_radius = 0.1f;
    float gp3_mu = 2.5f;
    int gp3_max_nn = 100;
    bool verbose = true;
};

/**
 * 简化版重建管道
 */
class SimplePipeline {
public:
    SimplePipeline(const SimpleReconConfig& config) : config_(config) {}
    
    bool reconstruct(const PointCloudT::Ptr& input_cloud, pcl::PolygonMesh& output_mesh) {
        if (!input_cloud || input_cloud->empty()) {
            std::cerr << "错误: 输入点云为空" << std::endl;
            return false;
        }
        
        if (config_.verbose) {
            std::cout << "开始简化重建管道..." << std::endl;
            std::cout << "输入点云大小: " << input_cloud->size() << " 点" << std::endl;
        }
        
        // 1. 法向量估计
        if (config_.verbose) std::cout << "估计法向量..." << std::endl;
        if (!estimateNormals(input_cloud)) {
            std::cerr << "法向量估计失败" << std::endl;
            return false;
        }
        
        // 2. GP3重建
        if (config_.verbose) std::cout << "执行GP3重建..." << std::endl;
        if (!performGP3Reconstruction(input_cloud, output_mesh)) {
            std::cerr << "GP3重建失败" << std::endl;
            return false;
        }
        
        if (config_.verbose) {
            std::cout << "重建完成!" << std::endl;
            std::cout << "输出网格: " << output_mesh.polygons.size() << " 面片" << std::endl;
        }
        
        return true;
    }

private:
    SimpleReconConfig config_;
    
    bool estimateNormals(const PointCloudT::Ptr& cloud) {
        pcl::NormalEstimation<PointT, PointT> ne;
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        
        ne.setInputCloud(cloud);
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(config_.normal_radius);
        
        PointCloudT::Ptr cloud_with_normals(new PointCloudT);
        ne.compute(*cloud_with_normals);
        
        // 复制法向量到原始点云
        for (size_t i = 0; i < cloud->size() && i < cloud_with_normals->size(); ++i) {
            cloud->points[i].normal_x = cloud_with_normals->points[i].normal_x;
            cloud->points[i].normal_y = cloud_with_normals->points[i].normal_y;
            cloud->points[i].normal_z = cloud_with_normals->points[i].normal_z;
        }
        
        return true;
    }
    
    bool performGP3Reconstruction(const PointCloudT::Ptr& cloud, pcl::PolygonMesh& mesh) {
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(cloud);
        
        pcl::GreedyProjectionTriangulation<PointT> gp3;
        gp3.setSearchRadius(config_.gp3_search_radius);
        gp3.setMu(config_.gp3_mu);
        gp3.setMaximumNearestNeighbors(config_.gp3_max_nn);
        gp3.setMaximumSurfaceAngle(M_PI/4); // 45度
        gp3.setMinimumAngle(M_PI/18); // 10度
        gp3.setMaximumAngle(2*M_PI/3); // 120度
        gp3.setNormalConsistency(false);
        
        gp3.setInputCloud(cloud);
        gp3.setSearchMethod(tree);
        gp3.reconstruct(mesh);
        
        return !mesh.polygons.empty();
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "用法: " << argv[0] << " <input.pcd> <output.ply>" << std::endl;
        return -1;
    }
    
    // 加载点云
    PointCloudT::Ptr cloud(new PointCloudT);
    if (pcl::io::loadPCDFile<PointT>(argv[1], *cloud) == -1) {
        std::cerr << "无法加载点云文件: " << argv[1] << std::endl;
        return -1;
    }
    
    // 配置重建参数
    SimpleReconConfig config;
    config.verbose = true;
    
    // 执行重建
    SimplePipeline pipeline(config);
    pcl::PolygonMesh result_mesh;
    
    if (pipeline.reconstruct(cloud, result_mesh)) {
        // 保存结果
        if (pcl::io::savePLYFile(argv[2], result_mesh) == 0) {
            std::cout << "结果已保存到: " << argv[2] << std::endl;
        } else {
            std::cerr << "保存失败: " << argv[2] << std::endl;
            return -1;
        }
    } else {
        std::cerr << "重建失败" << std::endl;
        return -1;
    }
    
    return 0;
}


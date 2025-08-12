/**
 * 简化版图割求解器
 * 避免复杂依赖，专注核心功能
 */

#ifndef SIMPLE_GRAPH_CUT_H
#define SIMPLE_GRAPH_CUT_H

#include <vector>
#include <memory>
#include <openvdb/openvdb.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Dense>

namespace recon {

/**
 * 简化版图割求解器
 */
class SimpleGraphCut {
public:
    using PointT = pcl::PointXYZRGBNormal;
    using PointCloudT = pcl::PointCloud<PointT>;
    using UDFGridT = openvdb::FloatGrid;

    /**
     * 构造函数
     */
    SimpleGraphCut();
    
    /**
     * 析构函数
     */
    ~SimpleGraphCut();

    /**
     * 执行图割
     */
    bool performGraphCut(const UDFGridT::Ptr& udf_grid,
                        const PointCloudT::Ptr& cloud,
                        std::vector<bool>& labels);

    /**
     * 设置参数
     */
    void setDataWeight(float weight) { data_weight_ = weight; }
    void setSmoothWeight(float weight) { smooth_weight_ = weight; }

private:
    float data_weight_ = 1.0f;
    float smooth_weight_ = 0.5f;
    
    /**
     * 简化的图割实现
     */
    bool simpleGraphCut(const std::vector<float>& data_costs,
                       const std::vector<std::vector<float>>& smooth_costs,
                       std::vector<bool>& labels);
};

} // namespace recon

#endif // SIMPLE_GRAPH_CUT_H


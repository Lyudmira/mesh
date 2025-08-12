/**
 * 简化版图割求解器实现
 */

#include "simple_graph_cut.h"
#include <algorithm>
#include <iostream>
#include <queue>

namespace recon {

SimpleGraphCut::SimpleGraphCut() = default;

SimpleGraphCut::~SimpleGraphCut() = default;

bool SimpleGraphCut::performGraphCut(const UDFGridT::Ptr& udf_grid,
                                     const PointCloudT::Ptr& cloud,
                                     std::vector<bool>& labels) {
    
    if (!udf_grid || !cloud || cloud->empty()) {
        return false;
    }
    
    std::cout << "执行简化图割..." << std::endl;
    
    // 简化实现：基于UDF值的阈值分割
    labels.resize(cloud->size());
    
    auto accessor = udf_grid->getConstAccessor();
    
    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& pt = cloud->points[i];
        openvdb::Vec3f world_pos(pt.x, pt.y, pt.z);
        openvdb::Coord coord = udf_grid->transform().worldToIndexCellCentered(world_pos);
        
        float udf_value = accessor.getValue(coord);
        
        // 简单的二分类：UDF值小于阈值的为内部
        labels[i] = udf_value < 0.05f;  // 5cm阈值
    }
    
    int inside_count = std::count(labels.begin(), labels.end(), true);
    std::cout << "图割完成: " << inside_count << "/" << cloud->size() << " 点被标记为内部" << std::endl;
    
    return true;
}

bool SimpleGraphCut::simpleGraphCut(const std::vector<float>& data_costs,
                                   const std::vector<std::vector<float>>& smooth_costs,
                                   std::vector<bool>& labels) {
    
    // 极简的图割实现：基于数据项的贪心分割
    labels.resize(data_costs.size());
    
    for (size_t i = 0; i < data_costs.size(); ++i) {
        labels[i] = data_costs[i] < 0.5f;
    }
    
    return true;
}

} // namespace recon


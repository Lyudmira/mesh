/**
 * @file types.h
 * @brief 统一的类型定义和基础数据结构
 * 
 * 本文件定义了整个重建系统中使用的核心数据类型，
 * 解决了各模块间类型不一致和缺失的问题。
 * 
 * @version 2.0
 * @date 2025-08-12
 */

#ifndef RECON_BASE_TYPES_H
#define RECON_BASE_TYPES_H

#include <vector>
#include <memory>
#include <string>
#include <map>
#include <chrono>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <sstream>

namespace recon {
namespace core {

// ============================================================================
// 基础几何类型
// ============================================================================

/**
 * @brief 3D点结构
 */
struct Point3D {
    float x, y, z;
    
    Point3D() : x(0), y(0), z(0) {}
    Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    Point3D operator+(const Point3D& other) const {
        return Point3D(x + other.x, y + other.y, z + other.z);
    }
    
    Point3D operator-(const Point3D& other) const {
        return Point3D(x - other.x, y - other.y, z - other.z);
    }
    
    Point3D operator*(float scale) const {
        return Point3D(x * scale, y * scale, z * scale);
    }
    
    float dot(const Point3D& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    Point3D normalized() const {
        float len = length();
        if (len > 1e-6f) {
            return Point3D(x / len, y / len, z / len);
        }
        return Point3D(0, 0, 1);
    }
};

/**
 * @brief 带颜色和法向量的点
 */
struct ColoredPoint {
    Point3D position;
    Point3D normal;
    float r, g, b;  // 颜色值 [0, 1]
    float curvature;
    
    ColoredPoint() : r(1), g(1), b(1), curvature(0) {}
    ColoredPoint(const Point3D& pos, float r_ = 1, float g_ = 1, float b_ = 1) 
        : position(pos), r(r_), g(g_), b(b_), curvature(0) {}
};

/**
 * @brief 三角形面
 */
struct Triangle {
    int v1, v2, v3;  // 顶点索引
    
    Triangle() : v1(0), v2(0), v3(0) {}
    Triangle(int a, int b, int c) : v1(a), v2(b), v3(c) {}
    
    bool isValid() const {
        return v1 != v2 && v2 != v3 && v1 != v3;
    }
};

/**
 * @brief 边结构
 */
struct Edge {
    int from, to;
    float capacity, rev_capacity;
    
    Edge() : from(0), to(0), capacity(0), rev_capacity(0) {}
    Edge(int f, int t, float cap, float rev_cap = 0) 
        : from(f), to(t), capacity(cap), rev_capacity(rev_cap) {}
};

// ============================================================================
// 复合数据结构
// ============================================================================

/**
 * @brief 点云数据结构
 */
class PointCloud {
public:
    std::vector<ColoredPoint> points;
    
    PointCloud() = default;
    explicit PointCloud(size_t size) : points(size) {}
    
    size_t size() const { return points.size(); }
    bool empty() const { return points.empty(); }
    void clear() { points.clear(); }
    void reserve(size_t size) { points.reserve(size); }
    
    ColoredPoint& operator[](size_t index) { return points[index]; }
    const ColoredPoint& operator[](size_t index) const { return points[index]; }
    
    void addPoint(const ColoredPoint& point) { points.push_back(point); }
    void addPoint(const Point3D& pos, float r = 1, float g = 1, float b = 1) {
        points.emplace_back(pos, r, g, b);
    }
    
    // 计算边界框
    std::pair<Point3D, Point3D> getBoundingBox() const {
        if (empty()) return {Point3D(), Point3D()};
        
        Point3D min_pt = points[0].position;
        Point3D max_pt = points[0].position;
        
        for (const auto& point : points) {
            const auto& pos = point.position;
            min_pt.x = std::min(min_pt.x, pos.x);
            min_pt.y = std::min(min_pt.y, pos.y);
            min_pt.z = std::min(min_pt.z, pos.z);
            max_pt.x = std::max(max_pt.x, pos.x);
            max_pt.y = std::max(max_pt.y, pos.y);
            max_pt.z = std::max(max_pt.z, pos.z);
        }
        
        return {min_pt, max_pt};
    }
};

/**
 * @brief 网格数据结构
 */
class Mesh {
public:
    std::vector<ColoredPoint> vertices;
    std::vector<Triangle> faces;
    
    Mesh() = default;
    
    size_t vertexCount() const { return vertices.size(); }
    size_t faceCount() const { return faces.size(); }
    bool empty() const { return vertices.empty() || faces.empty(); }
    
    void clear() {
        vertices.clear();
        faces.clear();
    }
    
    void addVertex(const ColoredPoint& vertex) {
        vertices.push_back(vertex);
    }
    
    void addTriangle(const Triangle& triangle) {
        faces.push_back(triangle);
    }
    
    void addFace(const Triangle& face) {
        if (face.isValid() && 
            face.v1 < static_cast<int>(vertices.size()) &&
            face.v2 < static_cast<int>(vertices.size()) &&
            face.v3 < static_cast<int>(vertices.size())) {
            faces.push_back(face);
        }
    }
    
    // 保存为OBJ格式
    bool saveToOBJ(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file) return false;
        
        file << "# OBJ file generated by Recon System\n";
        file << "# Vertices: " << vertices.size() << "\n";
        file << "# Faces: " << faces.size() << "\n\n";
        
        // 写入顶点
        for (const auto& vertex : vertices) {
            const auto& pos = vertex.position;
            file << "v " << pos.x << " " << pos.y << " " << pos.z << "\n";
        }
        
        // 写入法向量
        for (const auto& vertex : vertices) {
            const auto& normal = vertex.normal;
            file << "vn " << normal.x << " " << normal.y << " " << normal.z << "\n";
        }
        
        // 写入面
        for (const auto& face : faces) {
            file << "f " << (face.v1 + 1) << "//" << (face.v1 + 1) << " "
                 << (face.v2 + 1) << "//" << (face.v2 + 1) << " "
                 << (face.v3 + 1) << "//" << (face.v3 + 1) << "\n";
        }
        
        return true;
    }
};

// ============================================================================
// 体素和网格类型（简化版本，避免OpenVDB依赖）
// ============================================================================

/**
 * @brief 体素数据
 */
struct Voxel {
    float distance;      // UDF距离值
    float confidence;    // 置信度
    bool is_refined;     // 是否已细化
    
    Voxel() : distance(0), confidence(0), is_refined(false) {}
    Voxel(float dist, float conf = 1.0f) : distance(dist), confidence(conf), is_refined(false) {}
};

/**
 * @brief 简化的体素网格（替代OpenVDB）
 */
class VoxelGrid {
public:
    struct GridIndex {
        int x, y, z;
        GridIndex(int x_ = 0, int y_ = 0, int z_ = 0) : x(x_), y(y_), z(z_) {}
        
        bool operator<(const GridIndex& other) const {
            if (x != other.x) return x < other.x;
            if (y != other.y) return y < other.y;
            return z < other.z;
        }
    };
    
private:
    std::map<GridIndex, Voxel> voxels_;
    float voxel_size_;
    Point3D origin_;
    
public:
    explicit VoxelGrid(float voxel_size = 0.01f, const Point3D& origin = Point3D()) 
        : voxel_size_(voxel_size), origin_(origin) {}
    
    float getVoxelSize() const { return voxel_size_; }
    const Point3D& getOrigin() const { return origin_; }
    
    GridIndex worldToGrid(const Point3D& world_pos) const {
        Point3D relative = world_pos - origin_;
        return GridIndex(
            static_cast<int>(std::floor(relative.x / voxel_size_)),
            static_cast<int>(std::floor(relative.y / voxel_size_)),
            static_cast<int>(std::floor(relative.z / voxel_size_))
        );
    }
    
    Point3D gridToWorld(const GridIndex& grid_pos) const {
        return Point3D(
            origin_.x + grid_pos.x * voxel_size_,
            origin_.y + grid_pos.y * voxel_size_,
            origin_.z + grid_pos.z * voxel_size_
        );
    }
    
    void setValue(const GridIndex& index, const Voxel& voxel) {
        voxels_[index] = voxel;
    }
    
    bool getValue(const GridIndex& index, Voxel& voxel) const {
        auto it = voxels_.find(index);
        if (it != voxels_.end()) {
            voxel = it->second;
            return true;
        }
        return false;
    }
    
    size_t getActiveVoxelCount() const { return voxels_.size(); }
    
    std::vector<GridIndex> getActiveIndices() const {
        std::vector<GridIndex> indices;
        indices.reserve(voxels_.size());
        for (const auto& pair : voxels_) {
            indices.push_back(pair.first);
        }
        return indices;
    }
    
    void clear() { voxels_.clear(); }
};

// ============================================================================
// 性能和质量指标
// ============================================================================

/**
 * @brief 性能统计结构
 */
struct PerformanceStats {
    double total_time = 0.0;
    double preprocessing_time = 0.0;
    double algorithm_time = 0.0;
    double postprocessing_time = 0.0;
    
    size_t input_size = 0;
    size_t output_size = 0;
    size_t memory_usage_mb = 0;
    
    double throughput = 0.0;  // 处理速度（单位/秒）
    
    std::map<std::string, double> custom_metrics;
    
    void addMetric(const std::string& name, double value) {
        custom_metrics[name] = value;
    }
    
    double getMetric(const std::string& name) const {
        auto it = custom_metrics.find(name);
        return (it != custom_metrics.end()) ? it->second : 0.0;
    }
};

/**
 * @brief 质量评估结果
 */
struct QualityMetrics {
    double planarity_rms = 0.0;
    double hausdorff_distance = 0.0;
    double mesh_quality_score = 0.0;
    double feature_preservation_score = 0.0;
    double color_consistency_score = 0.0;
    
    size_t triangle_count = 0;
    size_t vertex_count = 0;
    size_t degenerate_triangles = 0;
    
    bool topology_valid = true;
    bool manifold = true;
    
    std::map<std::string, double> custom_quality_metrics;
    
    void addQualityMetric(const std::string& name, double value) {
        custom_quality_metrics[name] = value;
    }
    
    double getOverallScore() const {
        // 综合质量评分计算
        double score = 0.0;
        score += mesh_quality_score * 0.3;
        score += feature_preservation_score * 0.25;
        score += color_consistency_score * 0.2;
        score += (topology_valid ? 1.0 : 0.0) * 0.15;
        score += (manifold ? 1.0 : 0.0) * 0.1;
        return score;
    }
};

// ============================================================================
// 统一的结果类型模板
// ============================================================================

/**
 * @brief 统一的操作结果类型
 */
template<typename T>
struct Result {
    bool success = false;
    T data;
    std::string error_message;
    PerformanceStats performance;
    QualityMetrics quality;
    
    Result() = default;
    explicit Result(const T& data_) : success(true), data(data_) {}
    explicit Result(const std::string& error) : success(false), error_message(error) {}
    
    explicit operator bool() const { return success; }
    const T& operator*() const { return data; }
    T& operator*() { return data; }
    const T* operator->() const { return &data; }
    T* operator->() { return &data; }
    
    static Result<T> makeSuccess(const T& data) {
        Result<T> result;
        result.success = true;
        result.data = data;
        return result;
    }
    
    static Result<T> makeError(const std::string& error) {
        Result<T> result;
        result.success = false;
        result.error_message = error;
        return result;
    }
};

// ============================================================================
// 常用类型别名
// ============================================================================

// 点云相关
using PointCloudPtr = std::shared_ptr<PointCloud>;
using ConstPointCloudPtr = std::shared_ptr<const PointCloud>;

// 网格相关
using MeshPtr = std::shared_ptr<Mesh>;
using ConstMeshPtr = std::shared_ptr<const Mesh>;

// 体素网格相关
using VoxelGridPtr = std::shared_ptr<VoxelGrid>;
using ConstVoxelGridPtr = std::shared_ptr<const VoxelGrid>;

// 结果类型
using PointCloudResult = Result<PointCloudPtr>;
using MeshResult = Result<MeshPtr>;
using VoxelGridResult = Result<VoxelGridPtr>;
using BoolResult = Result<bool>;

// 数值类型
using IndexVector = std::vector<int>;
using FloatVector = std::vector<float>;
using DoubleVector = std::vector<double>;

} // namespace core
} // namespace recon

#endif // RECON_BASE_TYPES_H


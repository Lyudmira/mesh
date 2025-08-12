/**
 * @file data_validator.h
 * @brief 数据验证和质量检查系统
 * 
 * 提供了全面的数据验证功能，包括：
 * - 点云数据质量检查
 * - 网格拓扑验证
 * - 几何精度验证
 * - 性能指标计算
 * - 质量评分系统
 * 
 * @version 2.0
 * @date 2025-08-12
 */

#ifndef RECON_UTILS_DATA_VALIDATOR_H
#define RECON_UTILS_DATA_VALIDATOR_H

#include "../base/types.h"
#include "logger.h"
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <limits>

namespace recon {
namespace utils {

/**
 * @brief 验证结果类型
 */
enum class ValidationResult {
    PASS,           // 验证通过
    WARNING,        // 有警告但可接受
    FAIL,           // 验证失败
    CRITICAL_FAIL   // 严重失败，不可继续
};

/**
 * @brief 验证问题描述
 */
struct ValidationIssue {
    ValidationResult severity;
    std::string category;
    std::string description;
    std::string suggestion;
    std::map<std::string, double> metrics;
    
    ValidationIssue(ValidationResult sev, const std::string& cat, 
                    const std::string& desc, const std::string& sugg = "")
        : severity(sev), category(cat), description(desc), suggestion(sugg) {}
    
    void addMetric(const std::string& name, double value) {
        metrics[name] = value;
    }
};

/**
 * @brief 验证报告
 */
struct ValidationReport {
    std::vector<ValidationIssue> issues;
    core::QualityMetrics quality_metrics;
    core::PerformanceStats performance_stats;
    
    bool passed = true;
    bool has_warnings = false;
    bool has_critical_failures = false;
    
    void addIssue(const ValidationIssue& issue) {
        issues.push_back(issue);
        
        switch (issue.severity) {
            case ValidationResult::WARNING:
                has_warnings = true;
                break;
            case ValidationResult::FAIL:
                passed = false;
                break;
            case ValidationResult::CRITICAL_FAIL:
                passed = false;
                has_critical_failures = true;
                break;
            default:
                break;
        }
    }
    
    size_t getIssueCount(ValidationResult severity) const {
        return std::count_if(issues.begin(), issues.end(),
            [severity](const ValidationIssue& issue) {
                return issue.severity == severity;
            });
    }
    
    double getOverallScore() const {
        if (has_critical_failures) return 0.0;
        if (!passed) return 0.3;
        if (has_warnings) return 0.7;
        return 1.0;
    }
};

/**
 * @brief 点云数据验证器
 */
class PointCloudValidator {
public:
    /**
     * @brief 验证点云数据质量
     */
    static ValidationReport validate(const core::PointCloud& cloud) {
        ValidationReport report;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        LOG_INFO("PointCloudValidator", "开始验证点云数据，点数: " + std::to_string(cloud.size()));
        
        // 基础检查
        validateBasicProperties(cloud, report);
        
        // 几何检查
        validateGeometry(cloud, report);
        
        // 颜色检查
        validateColors(cloud, report);
        
        // 法向量检查
        validateNormals(cloud, report);
        
        // 密度分析
        analyzeDensity(cloud, report);
        
        // 噪声检测
        detectNoise(cloud, report);
        
        // 计算性能统计
        auto end_time = std::chrono::high_resolution_clock::now();
        report.performance_stats.total_time = 
            std::chrono::duration<double>(end_time - start_time).count();
        report.performance_stats.input_size = cloud.size();
        
        // 计算质量指标
        calculateQualityMetrics(cloud, report);
        
        LOG_INFO("PointCloudValidator", "点云验证完成，总体评分: " + 
                std::to_string(report.getOverallScore()));
        
        return report;
    }
    
private:
    static void validateBasicProperties(const core::PointCloud& cloud, ValidationReport& report) {
        // 检查点云是否为空
        if (cloud.empty()) {
            report.addIssue(ValidationIssue(ValidationResult::CRITICAL_FAIL, "Basic",
                "Point cloud is empty", "Provide non-empty point cloud"));
            return;
        }
        
        // 检查点云大小
        if (cloud.size() < 100) {
            report.addIssue(ValidationIssue(ValidationResult::WARNING, "Basic",
                "Point cloud has very few points (" + std::to_string(cloud.size()) + ")",
                "Consider using more points for better reconstruction quality"));
        }
        
        if (cloud.size() > 10000000) {  // 10M points
            report.addIssue(ValidationIssue(ValidationResult::WARNING, "Basic",
                "Point cloud is very large (" + std::to_string(cloud.size()) + " points)",
                "Consider downsampling for better performance"));
        }
        
        LOG_DEBUG("PointCloudValidator", "基础属性检查完成");
    }
    
    static void validateGeometry(const core::PointCloud& cloud, ValidationReport& report) {
        if (cloud.empty()) return;
        
        // 计算边界框
        auto [min_pt, max_pt] = cloud.getBoundingBox();
        core::Point3D size = max_pt - min_pt;
        
        // 检查退化维度
        const float min_dimension = 0.001f;  // 1mm
        int degenerate_dims = 0;
        if (size.x < min_dimension) degenerate_dims++;
        if (size.y < min_dimension) degenerate_dims++;
        if (size.z < min_dimension) degenerate_dims++;
        
        if (degenerate_dims > 0) {
            std::string msg = "Point cloud has " + std::to_string(degenerate_dims) + 
                             " degenerate dimension(s)";
            ValidationResult severity = (degenerate_dims >= 2) ? 
                ValidationResult::CRITICAL_FAIL : ValidationResult::WARNING;
            report.addIssue(ValidationIssue(severity, "Geometry", msg,
                "Ensure point cloud spans all three dimensions"));
        }
        
        // 检查重复点
        std::set<std::tuple<float, float, float>> unique_points;
        int duplicate_count = 0;
        
        for (const auto& point : cloud.points) {
            auto key = std::make_tuple(
                std::round(point.position.x * 1000) / 1000,  // 1mm精度
                std::round(point.position.y * 1000) / 1000,
                std::round(point.position.z * 1000) / 1000
            );
            
            if (unique_points.find(key) != unique_points.end()) {
                duplicate_count++;
            } else {
                unique_points.insert(key);
            }
        }
        
        if (duplicate_count > 0) {
            double duplicate_ratio = static_cast<double>(duplicate_count) / cloud.size();
            ValidationResult severity = (duplicate_ratio > 0.1) ? 
                ValidationResult::FAIL : ValidationResult::WARNING;
            
            ValidationIssue issue(severity, "Geometry",
                "Found " + std::to_string(duplicate_count) + " duplicate points (" +
                std::to_string(duplicate_ratio * 100) + "%)",
                "Remove duplicate points to improve reconstruction quality");
            issue.addMetric("duplicate_count", duplicate_count);
            issue.addMetric("duplicate_ratio", duplicate_ratio);
            report.addIssue(issue);
        }
        
        // 检查异常值
        detectOutliers(cloud, report);
        
        LOG_DEBUG("PointCloudValidator", "几何检查完成");
    }
    
    static void validateColors(const core::PointCloud& cloud, ValidationReport& report) {
        if (cloud.empty()) return;
        
        int invalid_color_count = 0;
        double total_brightness = 0.0;
        
        for (const auto& point : cloud.points) {
            // 检查颜色值范围
            if (point.r < 0 || point.r > 1 || 
                point.g < 0 || point.g > 1 || 
                point.b < 0 || point.b > 1) {
                invalid_color_count++;
            }
            
            // 计算亮度
            double brightness = 0.299 * point.r + 0.587 * point.g + 0.114 * point.b;
            total_brightness += brightness;
        }
        
        if (invalid_color_count > 0) {
            double invalid_ratio = static_cast<double>(invalid_color_count) / cloud.size();
            ValidationResult severity = (invalid_ratio > 0.05) ? 
                ValidationResult::FAIL : ValidationResult::WARNING;
            
            ValidationIssue issue(severity, "Color",
                "Found " + std::to_string(invalid_color_count) + " points with invalid colors",
                "Ensure color values are in [0,1] range");
            issue.addMetric("invalid_color_count", invalid_color_count);
            issue.addMetric("invalid_color_ratio", invalid_ratio);
            report.addIssue(issue);
        }
        
        // 检查颜色分布
        double avg_brightness = total_brightness / cloud.size();
        if (avg_brightness < 0.1) {
            report.addIssue(ValidationIssue(ValidationResult::WARNING, "Color",
                "Point cloud appears very dark (avg brightness: " + 
                std::to_string(avg_brightness) + ")",
                "Check lighting conditions during capture"));
        } else if (avg_brightness > 0.9) {
            report.addIssue(ValidationIssue(ValidationResult::WARNING, "Color",
                "Point cloud appears very bright (avg brightness: " + 
                std::to_string(avg_brightness) + ")",
                "Check for overexposure during capture"));
        }
        
        LOG_DEBUG("PointCloudValidator", "颜色检查完成");
    }
    
    static void validateNormals(const core::PointCloud& cloud, ValidationReport& report) {
        if (cloud.empty()) return;
        
        int invalid_normal_count = 0;
        int zero_normal_count = 0;
        
        for (const auto& point : cloud.points) {
            float length = point.normal.length();
            
            if (length < 1e-6f) {
                zero_normal_count++;
            } else if (std::abs(length - 1.0f) > 0.1f) {
                invalid_normal_count++;
            }
        }
        
        if (zero_normal_count > 0) {
            double zero_ratio = static_cast<double>(zero_normal_count) / cloud.size();
            ValidationResult severity = (zero_ratio > 0.1) ? 
                ValidationResult::FAIL : ValidationResult::WARNING;
            
            ValidationIssue issue(severity, "Normal",
                "Found " + std::to_string(zero_normal_count) + " points with zero normals",
                "Compute or fix normal vectors");
            issue.addMetric("zero_normal_count", zero_normal_count);
            issue.addMetric("zero_normal_ratio", zero_ratio);
            report.addIssue(issue);
        }
        
        if (invalid_normal_count > 0) {
            double invalid_ratio = static_cast<double>(invalid_normal_count) / cloud.size();
            if (invalid_ratio > 0.05) {
                ValidationIssue issue(ValidationResult::WARNING, "Normal",
                    "Found " + std::to_string(invalid_normal_count) + " points with unnormalized normals",
                    "Normalize normal vectors to unit length");
                issue.addMetric("invalid_normal_count", invalid_normal_count);
                issue.addMetric("invalid_normal_ratio", invalid_ratio);
                report.addIssue(issue);
            }
        }
        
        LOG_DEBUG("PointCloudValidator", "法向量检查完成");
    }
    
    static void analyzeDensity(const core::PointCloud& cloud, ValidationReport& report) {
        if (cloud.size() < 2) return;
        
        // 简化的密度分析：计算到最近邻的平均距离
        std::vector<float> nearest_distances;
        nearest_distances.reserve(std::min(cloud.size(), size_t(1000)));  // 采样1000个点
        
        size_t sample_step = std::max(size_t(1), cloud.size() / 1000);
        
        for (size_t i = 0; i < cloud.size(); i += sample_step) {
            const auto& point1 = cloud.points[i];
            float min_dist = std::numeric_limits<float>::max();
            
            for (size_t j = 0; j < cloud.size(); ++j) {
                if (i == j) continue;
                
                const auto& point2 = cloud.points[j];
                core::Point3D diff = point1.position - point2.position;
                float dist = diff.length();
                
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            
            if (min_dist < std::numeric_limits<float>::max()) {
                nearest_distances.push_back(min_dist);
            }
        }
        
        if (!nearest_distances.empty()) {
            // 计算统计信息
            std::sort(nearest_distances.begin(), nearest_distances.end());
            
            float median_dist = nearest_distances[nearest_distances.size() / 2];
            float mean_dist = std::accumulate(nearest_distances.begin(), nearest_distances.end(), 0.0f) / nearest_distances.size();
            
            // 检查密度均匀性
            float q1 = nearest_distances[nearest_distances.size() / 4];
            float q3 = nearest_distances[nearest_distances.size() * 3 / 4];
            float iqr = q3 - q1;
            float density_variation = iqr / median_dist;
            
            if (density_variation > 2.0f) {
                ValidationIssue issue(ValidationResult::WARNING, "Density",
                    "Point cloud has highly variable density (variation: " + 
                    std::to_string(density_variation) + ")",
                    "Consider resampling to achieve more uniform density");
                issue.addMetric("density_variation", density_variation);
                issue.addMetric("median_distance", median_dist);
                issue.addMetric("mean_distance", mean_dist);
                report.addIssue(issue);
            }
        }
        
        LOG_DEBUG("PointCloudValidator", "密度分析完成");
    }
    
    static void detectOutliers(const core::PointCloud& cloud, ValidationReport& report) {
        if (cloud.size() < 10) return;
        
        // 计算中心点和标准差
        core::Point3D center(0, 0, 0);
        for (const auto& point : cloud.points) {
            center = center + point.position;
        }
        center = center * (1.0f / cloud.size());
        
        // 计算到中心的距离
        std::vector<float> distances;
        distances.reserve(cloud.size());
        
        for (const auto& point : cloud.points) {
            core::Point3D diff = point.position - center;
            distances.push_back(diff.length());
        }
        
        // 计算统计信息
        std::sort(distances.begin(), distances.end());
        float median = distances[distances.size() / 2];
        float q3 = distances[distances.size() * 3 / 4];
        float iqr_threshold = q3 + 1.5f * (q3 - distances[distances.size() / 4]);
        
        // 计算异常值
        int outlier_count = 0;
        for (float dist : distances) {
            if (dist > iqr_threshold) {
                outlier_count++;
            }
        }
        
        if (outlier_count > 0) {
            double outlier_ratio = static_cast<double>(outlier_count) / cloud.size();
            if (outlier_ratio > 0.05) {  // 超过5%的异常值
                ValidationResult severity = (outlier_ratio > 0.2) ? 
                    ValidationResult::FAIL : ValidationResult::WARNING;
                
                ValidationIssue issue(severity, "Geometry",
                    "Found " + std::to_string(outlier_count) + " outlier points (" +
                    std::to_string(outlier_ratio * 100) + "%)",
                    "Remove outlier points to improve reconstruction quality");
                issue.addMetric("outlier_count", outlier_count);
                issue.addMetric("outlier_ratio", outlier_ratio);
                issue.addMetric("outlier_threshold", iqr_threshold);
                report.addIssue(issue);
            }
        }
    }
    
    static void detectNoise(const core::PointCloud& cloud, ValidationReport& report) {
        // 简化的噪声检测：基于局部密度
        if (cloud.size() < 50) return;
        
        // 采样检测
        size_t sample_size = std::min(cloud.size(), size_t(500));
        size_t sample_step = cloud.size() / sample_size;
        
        int isolated_points = 0;
        const float isolation_radius = 0.01f;  // 1cm
        const int min_neighbors = 3;
        
        for (size_t i = 0; i < cloud.size(); i += sample_step) {
            const auto& point1 = cloud.points[i];
            int neighbor_count = 0;
            
            for (size_t j = 0; j < cloud.size(); ++j) {
                if (i == j) continue;
                
                const auto& point2 = cloud.points[j];
                core::Point3D diff = point1.position - point2.position;
                
                if (diff.length() < isolation_radius) {
                    neighbor_count++;
                    if (neighbor_count >= min_neighbors) break;
                }
            }
            
            if (neighbor_count < min_neighbors) {
                isolated_points++;
            }
        }
        
        if (isolated_points > 0) {
            double isolation_ratio = static_cast<double>(isolated_points) / sample_size;
            if (isolation_ratio > 0.1) {
                ValidationIssue issue(ValidationResult::WARNING, "Noise",
                    "Detected potential noise: " + std::to_string(isolated_points) + 
                    " isolated points in sample",
                    "Apply noise filtering before reconstruction");
                issue.addMetric("isolated_points", isolated_points);
                issue.addMetric("isolation_ratio", isolation_ratio);
                report.addIssue(issue);
            }
        }
        
        LOG_DEBUG("PointCloudValidator", "噪声检测完成");
    }
    
    static void calculateQualityMetrics(const core::PointCloud& cloud, ValidationReport& report) {
        // 计算基础质量指标
        report.quality_metrics.vertex_count = cloud.size();
        
        if (!cloud.empty()) {
            // 计算边界框体积
            auto [min_pt, max_pt] = cloud.getBoundingBox();
            core::Point3D size = max_pt - min_pt;
            double volume = size.x * size.y * size.z;
            
            // 计算密度
            double density = cloud.size() / volume;
            report.quality_metrics.addQualityMetric("point_density", density);
            report.quality_metrics.addQualityMetric("bounding_box_volume", volume);
            
            // 计算颜色一致性
            double color_variance = calculateColorVariance(cloud);
            report.quality_metrics.color_consistency_score = 1.0 - std::min(1.0, color_variance);
            
            // 计算整体质量评分
            double base_score = 0.8;  // 基础分
            
            // 根据问题调整评分
            for (const auto& issue : report.issues) {
                switch (issue.severity) {
                    case ValidationResult::WARNING:
                        base_score -= 0.05;
                        break;
                    case ValidationResult::FAIL:
                        base_score -= 0.2;
                        break;
                    case ValidationResult::CRITICAL_FAIL:
                        base_score = 0.0;
                        break;
                    default:
                        break;
                }
            }
            
            report.quality_metrics.mesh_quality_score = std::max(0.0, base_score);
        }
    }
    
    static double calculateColorVariance(const core::PointCloud& cloud) {
        if (cloud.empty()) return 0.0;
        
        // 计算颜色均值
        double mean_r = 0, mean_g = 0, mean_b = 0;
        for (const auto& point : cloud.points) {
            mean_r += point.r;
            mean_g += point.g;
            mean_b += point.b;
        }
        mean_r /= cloud.size();
        mean_g /= cloud.size();
        mean_b /= cloud.size();
        
        // 计算方差
        double var_r = 0, var_g = 0, var_b = 0;
        for (const auto& point : cloud.points) {
            var_r += (point.r - mean_r) * (point.r - mean_r);
            var_g += (point.g - mean_g) * (point.g - mean_g);
            var_b += (point.b - mean_b) * (point.b - mean_b);
        }
        var_r /= cloud.size();
        var_g /= cloud.size();
        var_b /= cloud.size();
        
        return (var_r + var_g + var_b) / 3.0;
    }
};

/**
 * @brief 网格数据验证器
 */
class MeshValidator {
public:
    /**
     * @brief 验证网格数据质量
     */
    static ValidationReport validate(const core::Mesh& mesh) {
        ValidationReport report;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        LOG_INFO("MeshValidator", "开始验证网格数据，顶点数: " + std::to_string(mesh.vertexCount()) +
                ", 面数: " + std::to_string(mesh.faceCount()));
        
        // 基础检查
        validateBasicProperties(mesh, report);
        
        if (!mesh.empty()) {
            // 拓扑检查
            validateTopology(mesh, report);
            
            // 几何检查
            validateGeometry(mesh, report);
            
            // 质量检查
            validateQuality(mesh, report);
        }
        
        // 计算性能统计
        auto end_time = std::chrono::high_resolution_clock::now();
        report.performance_stats.total_time = 
            std::chrono::duration<double>(end_time - start_time).count();
        report.performance_stats.input_size = mesh.vertexCount();
        report.performance_stats.output_size = mesh.faceCount();
        
        // 计算质量指标
        calculateQualityMetrics(mesh, report);
        
        LOG_INFO("MeshValidator", "网格验证完成，总体评分: " + 
                std::to_string(report.getOverallScore()));
        
        return report;
    }
    
private:
    static void validateBasicProperties(const core::Mesh& mesh, ValidationReport& report) {
        // 检查是否为空
        if (mesh.empty()) {
            report.addIssue(ValidationIssue(ValidationResult::CRITICAL_FAIL, "Basic",
                "Mesh is empty", "Generate non-empty mesh"));
            return;
        }
        
        // 检查顶点数
        if (mesh.vertexCount() < 3) {
            report.addIssue(ValidationIssue(ValidationResult::CRITICAL_FAIL, "Basic",
                "Mesh has too few vertices (" + std::to_string(mesh.vertexCount()) + ")",
                "Ensure mesh has at least 3 vertices"));
            return;
        }
        
        // 检查面数
        if (mesh.faceCount() == 0) {
            report.addIssue(ValidationIssue(ValidationResult::CRITICAL_FAIL, "Basic",
                "Mesh has no faces", "Generate faces for the mesh"));
            return;
        }
        
        LOG_DEBUG("MeshValidator", "基础属性检查完成");
    }
    
    static void validateTopology(const core::Mesh& mesh, ValidationReport& report) {
        // 检查面的顶点索引
        int invalid_face_count = 0;
        int degenerate_face_count = 0;
        
        for (const auto& face : mesh.faces) {
            // 检查索引范围
            if (face.v1 >= static_cast<int>(mesh.vertexCount()) ||
                face.v2 >= static_cast<int>(mesh.vertexCount()) ||
                face.v3 >= static_cast<int>(mesh.vertexCount()) ||
                face.v1 < 0 || face.v2 < 0 || face.v3 < 0) {
                invalid_face_count++;
                continue;
            }
            
            // 检查退化面
            if (!face.isValid()) {
                degenerate_face_count++;
            }
        }
        
        if (invalid_face_count > 0) {
            report.addIssue(ValidationIssue(ValidationResult::CRITICAL_FAIL, "Topology",
                "Found " + std::to_string(invalid_face_count) + " faces with invalid vertex indices",
                "Fix vertex indices to be within valid range"));
        }
        
        if (degenerate_face_count > 0) {
            double degenerate_ratio = static_cast<double>(degenerate_face_count) / mesh.faceCount();
            ValidationResult severity = (degenerate_ratio > 0.05) ? 
                ValidationResult::FAIL : ValidationResult::WARNING;
            
            ValidationIssue issue(severity, "Topology",
                "Found " + std::to_string(degenerate_face_count) + " degenerate faces",
                "Remove degenerate faces to improve mesh quality");
            issue.addMetric("degenerate_count", degenerate_face_count);
            issue.addMetric("degenerate_ratio", degenerate_ratio);
            report.addIssue(issue);
        }
        
        // 检查流形性（简化版本）
        checkManifoldness(mesh, report);
        
        LOG_DEBUG("MeshValidator", "拓扑检查完成");
    }
    
    static void validateGeometry(const core::Mesh& mesh, ValidationReport& report) {
        // 检查面积
        int zero_area_faces = 0;
        double total_area = 0.0;
        
        for (const auto& face : mesh.faces) {
            if (face.v1 < static_cast<int>(mesh.vertexCount()) &&
                face.v2 < static_cast<int>(mesh.vertexCount()) &&
                face.v3 < static_cast<int>(mesh.vertexCount())) {
                
                const auto& v1 = mesh.vertices[face.v1].position;
                const auto& v2 = mesh.vertices[face.v2].position;
                const auto& v3 = mesh.vertices[face.v3].position;
                
                // 计算面积
                core::Point3D edge1 = v2 - v1;
                core::Point3D edge2 = v3 - v1;
                
                // 叉积的模长的一半就是三角形面积
                core::Point3D cross(
                    edge1.y * edge2.z - edge1.z * edge2.y,
                    edge1.z * edge2.x - edge1.x * edge2.z,
                    edge1.x * edge2.y - edge1.y * edge2.x
                );
                
                double area = cross.length() * 0.5;
                total_area += area;
                
                if (area < 1e-10) {
                    zero_area_faces++;
                }
            }
        }
        
        if (zero_area_faces > 0) {
            double zero_ratio = static_cast<double>(zero_area_faces) / mesh.faceCount();
            ValidationResult severity = (zero_ratio > 0.05) ? 
                ValidationResult::FAIL : ValidationResult::WARNING;
            
            ValidationIssue issue(severity, "Geometry",
                "Found " + std::to_string(zero_area_faces) + " faces with zero area",
                "Remove zero-area faces");
            issue.addMetric("zero_area_count", zero_area_faces);
            issue.addMetric("zero_area_ratio", zero_ratio);
            report.addIssue(issue);
        }
        
        LOG_DEBUG("MeshValidator", "几何检查完成");
    }
    
    static void validateQuality(const core::Mesh& mesh, ValidationReport& report) {
        // 计算面质量（长宽比等）
        std::vector<double> aspect_ratios;
        aspect_ratios.reserve(mesh.faceCount());
        
        for (const auto& face : mesh.faces) {
            if (face.v1 < static_cast<int>(mesh.vertexCount()) &&
                face.v2 < static_cast<int>(mesh.vertexCount()) &&
                face.v3 < static_cast<int>(mesh.vertexCount())) {
                
                const auto& v1 = mesh.vertices[face.v1].position;
                const auto& v2 = mesh.vertices[face.v2].position;
                const auto& v3 = mesh.vertices[face.v3].position;
                
                // 计算边长
                double edge1_len = (v2 - v1).length();
                double edge2_len = (v3 - v2).length();
                double edge3_len = (v1 - v3).length();
                
                if (edge1_len > 0 && edge2_len > 0 && edge3_len > 0) {
                    double max_edge = std::max({edge1_len, edge2_len, edge3_len});
                    double min_edge = std::min({edge1_len, edge2_len, edge3_len});
                    double aspect_ratio = max_edge / min_edge;
                    aspect_ratios.push_back(aspect_ratio);
                }
            }
        }
        
        if (!aspect_ratios.empty()) {
            double mean_aspect_ratio = std::accumulate(aspect_ratios.begin(), aspect_ratios.end(), 0.0) / aspect_ratios.size();
            
            // 计算高长宽比面的数量
            int high_aspect_faces = std::count_if(aspect_ratios.begin(), aspect_ratios.end(),
                [](double ratio) { return ratio > 10.0; });
            
            if (high_aspect_faces > 0) {
                double high_aspect_ratio = static_cast<double>(high_aspect_faces) / aspect_ratios.size();
                if (high_aspect_ratio > 0.1) {
                    ValidationIssue issue(ValidationResult::WARNING, "Quality",
                        "Found " + std::to_string(high_aspect_faces) + " faces with high aspect ratio",
                        "Consider remeshing to improve triangle quality");
                    issue.addMetric("high_aspect_count", high_aspect_faces);
                    issue.addMetric("high_aspect_ratio", high_aspect_ratio);
                    issue.addMetric("mean_aspect_ratio", mean_aspect_ratio);
                    report.addIssue(issue);
                }
            }
        }
        
        LOG_DEBUG("MeshValidator", "质量检查完成");
    }
    
    static void checkManifoldness(const core::Mesh& mesh, ValidationReport& report) {
        // 简化的流形性检查：检查边的共享情况
        std::map<std::pair<int, int>, int> edge_count;
        
        for (const auto& face : mesh.faces) {
            if (face.isValid() &&
                face.v1 < static_cast<int>(mesh.vertexCount()) &&
                face.v2 < static_cast<int>(mesh.vertexCount()) &&
                face.v3 < static_cast<int>(mesh.vertexCount())) {
                
                // 添加三条边（确保边的方向一致）
                auto addEdge = [&](int v1, int v2) {
                    auto edge = (v1 < v2) ? std::make_pair(v1, v2) : std::make_pair(v2, v1);
                    edge_count[edge]++;
                };
                
                addEdge(face.v1, face.v2);
                addEdge(face.v2, face.v3);
                addEdge(face.v3, face.v1);
            }
        }
        
        // 检查非流形边（被超过2个面共享的边）
        int non_manifold_edges = 0;
        int boundary_edges = 0;
        
        for (const auto& [edge, count] : edge_count) {
            if (count > 2) {
                non_manifold_edges++;
            } else if (count == 1) {
                boundary_edges++;
            }
        }
        
        if (non_manifold_edges > 0) {
            report.addIssue(ValidationIssue(ValidationResult::FAIL, "Topology",
                "Found " + std::to_string(non_manifold_edges) + " non-manifold edges",
                "Fix mesh topology to ensure manifoldness"));
            report.quality_metrics.manifold = false;
        } else {
            report.quality_metrics.manifold = true;
        }
        
        if (boundary_edges > 0) {
            ValidationIssue issue(ValidationResult::WARNING, "Topology",
                "Found " + std::to_string(boundary_edges) + " boundary edges",
                "Mesh has open boundaries");
            issue.addMetric("boundary_edge_count", boundary_edges);
            report.addIssue(issue);
        }
    }
    
    static void calculateQualityMetrics(const core::Mesh& mesh, ValidationReport& report) {
        report.quality_metrics.vertex_count = mesh.vertexCount();
        report.quality_metrics.triangle_count = mesh.faceCount();
        
        // 计算退化三角形数量
        report.quality_metrics.degenerate_triangles = 0;
        for (const auto& face : mesh.faces) {
            if (!face.isValid()) {
                report.quality_metrics.degenerate_triangles++;
            }
        }
        
        // 计算拓扑有效性
        report.quality_metrics.topology_valid = (report.quality_metrics.degenerate_triangles == 0);
        
        // 计算整体质量评分
        double base_score = 0.9;  // 基础分
        
        // 根据问题调整评分
        for (const auto& issue : report.issues) {
            switch (issue.severity) {
                case ValidationResult::WARNING:
                    base_score -= 0.05;
                    break;
                case ValidationResult::FAIL:
                    base_score -= 0.2;
                    break;
                case ValidationResult::CRITICAL_FAIL:
                    base_score = 0.0;
                    break;
                default:
                    break;
            }
        }
        
        report.quality_metrics.mesh_quality_score = std::max(0.0, base_score);
    }
};

} // namespace utils
} // namespace recon

#endif // RECON_UTILS_DATA_VALIDATOR_H


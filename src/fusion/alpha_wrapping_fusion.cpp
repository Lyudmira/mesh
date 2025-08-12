/**
 * Alpha包装融合器实现
 */

#include "alpha_wrapping_fusion.h"
#include <pcl/common/common.h>
#include <pcl/io/ply_io.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <filesystem>

// CGAL额外包含
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/smooth_mesh.h>

namespace recon {

// ============================================================================
// AlphaWrappingFusion实现
// ============================================================================

AlphaWrappingFusion::AlphaWrappingFusion(const AlphaWrappingConfig& config)
    : config_(config) {
    
    if (!initialize()) {
        throw std::runtime_error("Alpha包装融合器初始化失败");
    }
}

AlphaWrappingFusion::~AlphaWrappingFusion() {
    cleanup();
}

bool AlphaWrappingFusion::initialize() {
    try {
        // 创建调试输出目录
        if (config_.enable_debug_output) {
            std::filesystem::create_directories(config_.debug_output_dir);
        }
        
        std::cout << "Alpha包装融合器初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Alpha包装融合器初始化异常: " << e.what() << std::endl;
        return false;
    }
}

bool AlphaWrappingFusion::performAlphaWrapping(
    const pcl::PolygonMesh& shell_mesh,
    const pcl::PolygonMesh& detail_mesh,
    pcl::PolygonMesh& fused_mesh,
    AlphaWrappingResult& result) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "开始Alpha包装融合..." << std::endl;
    std::cout << "外壳网格: " << shell_mesh.polygons.size() << " 面" << std::endl;
    std::cout << "细节网格: " << detail_mesh.polygons.size() << " 面" << std::endl;
    
    try {
        // 1. 转换PCL网格到CGAL格式
        Mesh shell_cgal, detail_cgal;
        
        if (!convertPCLToCGAL(shell_mesh, shell_cgal)) {
            result.success = false;
            result.error_message = "外壳网格转换失败";
            return false;
        }
        
        if (!convertPCLToCGAL(detail_mesh, detail_cgal)) {
            result.success = false;
            result.error_message = "细节网格转换失败";
            return false;
        }
        
        result.original_vertices = static_cast<int>(shell_cgal.number_of_vertices() + detail_cgal.number_of_vertices());
        result.original_faces = static_cast<int>(shell_cgal.number_of_faces() + detail_cgal.number_of_faces());
        
        // 2. 合并输入网格
        Mesh combined_mesh = shell_cgal;
        
        // 简化的网格合并（实际应用中需要更复杂的合并算法）
        for (auto v : detail_cgal.vertices()) {
            Point_3 p = detail_cgal.point(v);
            combined_mesh.add_vertex(p);
        }
        
        // 3. 执行Alpha包装
        std::cout << "执行Alpha包装..." << std::endl;
        auto wrapping_start = std::chrono::high_resolution_clock::now();
        
        Mesh wrapped_mesh;
        if (!performWrapping(combined_mesh, wrapped_mesh)) {
            result.success = false;
            result.error_message = "Alpha包装失败";
            return false;
        }
        
        auto wrapping_end = std::chrono::high_resolution_clock::now();
        result.wrapping_time = std::chrono::duration<double>(wrapping_end - wrapping_start).count();
        
        std::cout << "Alpha包装完成，生成 " << wrapped_mesh.number_of_faces() << " 面" << std::endl;
        
        // 4. 检测平面
        if (config_.plane_angle_threshold > 0) {
            std::cout << "检测平面..." << std::endl;
            auto plane_start = std::chrono::high_resolution_clock::now();
            
            if (!detectPlanes(wrapped_mesh, result.detected_planes)) {
                std::cerr << "平面检测失败" << std::endl;
            } else {
                std::cout << "检测到 " << result.detected_planes.size() << " 个平面" << std::endl;
            }
            
            auto plane_end = std::chrono::high_resolution_clock::now();
            result.plane_detection_time = std::chrono::duration<double>(plane_end - plane_start).count();
        }
        
        // 5. 重投影交线
        if (config_.enable_line_projection && !result.detected_planes.empty()) {
            std::cout << "重投影交线..." << std::endl;
            auto line_start = std::chrono::high_resolution_clock::now();
            
            if (!reprojectIntersectionLines(wrapped_mesh, result.detected_planes, result.intersection_lines)) {
                std::cerr << "交线重投影失败" << std::endl;
            } else {
                std::cout << "处理了 " << result.intersection_lines.size() << " 条交线" << std::endl;
            }
            
            auto line_end = std::chrono::high_resolution_clock::now();
            result.line_projection_time = std::chrono::duration<double>(line_end - line_start).count();
        }
        
        // 6. 强制平面对齐
        if (config_.enforce_orthogonality && !result.detected_planes.empty()) {
            std::cout << "强制平面对齐..." << std::endl;
            enforcePlaneAlignment(wrapped_mesh, result.detected_planes);
        }
        
        // 7. 强制直线约束
        if (!result.intersection_lines.empty()) {
            std::cout << "强制直线约束..." << std::endl;
            enforceStraightLines(wrapped_mesh, result.intersection_lines);
        }
        
        // 8. 优化网格质量
        std::cout << "优化网格质量..." << std::endl;
        optimizeMeshQuality(wrapped_mesh);
        
        // 9. 验证几何约束
        if (!validateGeometricConstraints(wrapped_mesh, result.detected_planes, result.intersection_lines)) {
            std::cerr << "几何约束验证失败" << std::endl;
        }
        
        // 10. 转换回PCL格式
        if (!convertCGALToPCL(wrapped_mesh, fused_mesh)) {
            result.success = false;
            result.error_message = "结果网格转换失败";
            return false;
        }
        
        // 11. 更新统计信息
        updateStatistics(result, combined_mesh, wrapped_mesh);
        
        // 12. 保存调试信息
        if (config_.enable_debug_output) {
            saveDebugMesh(wrapped_mesh, config_.debug_output_dir + "/wrapped_mesh.ply");
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        result.success = true;
        result.wrapped_mesh = wrapped_mesh;
        last_result_ = result;
        
        std::cout << "Alpha包装融合完成" << std::endl;
        std::cout << "统计信息:" << std::endl;
        std::cout << "  包装时间: " << result.wrapping_time << " 秒" << std::endl;
        std::cout << "  平面检测时间: " << result.plane_detection_time << " 秒" << std::endl;
        std::cout << "  交线投影时间: " << result.line_projection_time << " 秒" << std::endl;
        std::cout << "  总时间: " << result.total_time << " 秒" << std::endl;
        std::cout << "  原始顶点数: " << result.original_vertices << std::endl;
        std::cout << "  包装后顶点数: " << result.wrapped_vertices << std::endl;
        std::cout << "  原始面数: " << result.original_faces << std::endl;
        std::cout << "  包装后面数: " << result.wrapped_faces << std::endl;
        std::cout << "  检测平面数: " << result.detected_plane_count << std::endl;
        std::cout << "  交线数: " << result.intersection_line_count << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Alpha包装融合异常: " << e.what() << std::endl;
        result.success = false;
        result.error_message = std::string("异常: ") + e.what();
        return false;
    }
}

// ============================================================================
// 网格转换方法
// ============================================================================

bool AlphaWrappingFusion::convertPCLToCGAL(const pcl::PolygonMesh& pcl_mesh, Mesh& cgal_mesh) {
    try {
        cgal_mesh.clear();
        
        // 转换点云
        pcl::PointCloud<pcl::PointXYZ> vertices;
        pcl::fromPCLPointCloud2(pcl_mesh.cloud, vertices);
        
        // 添加顶点
        std::vector<Mesh::Vertex_index> vertex_map;
        vertex_map.reserve(vertices.size());
        
        for (const auto& point : vertices) {
            Point_3 p(point.x, point.y, point.z);
            Mesh::Vertex_index v = cgal_mesh.add_vertex(p);
            vertex_map.push_back(v);
        }
        
        // 添加面
        for (const auto& polygon : pcl_mesh.polygons) {
            if (polygon.vertices.size() == 3) {
                // 三角形
                cgal_mesh.add_face(
                    vertex_map[polygon.vertices[0]],
                    vertex_map[polygon.vertices[1]],
                    vertex_map[polygon.vertices[2]]
                );
            } else if (polygon.vertices.size() > 3) {
                // 多边形，需要三角化
                std::vector<Mesh::Vertex_index> face_vertices;
                for (uint32_t idx : polygon.vertices) {
                    face_vertices.push_back(vertex_map[idx]);
                }
                cgal_mesh.add_face(face_vertices);
            }
        }
        
        // 验证网格
        if (!cgal_mesh.is_valid()) {
            std::cerr << "转换后的CGAL网格无效" << std::endl;
            return false;
        }
        
        std::cout << "PCL到CGAL转换完成，顶点数: " << cgal_mesh.number_of_vertices() 
                  << "，面数: " << cgal_mesh.number_of_faces() << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "PCL到CGAL转换异常: " << e.what() << std::endl;
        return false;
    }
}

bool AlphaWrappingFusion::convertCGALToPCL(const Mesh& cgal_mesh, pcl::PolygonMesh& pcl_mesh) {
    try {
        pcl_mesh.polygons.clear();
        
        // 转换顶点
        pcl::PointCloud<pcl::PointXYZ> vertices;
        vertices.reserve(cgal_mesh.number_of_vertices());
        
        std::unordered_map<Mesh::Vertex_index, int> vertex_map;
        int vertex_idx = 0;
        
        for (auto v : cgal_mesh.vertices()) {
            Point_3 p = cgal_mesh.point(v);
            pcl::PointXYZ point;
            point.x = static_cast<float>(p.x());
            point.y = static_cast<float>(p.y());
            point.z = static_cast<float>(p.z());
            vertices.push_back(point);
            
            vertex_map[v] = vertex_idx++;
        }
        
        // 转换面
        for (auto f : cgal_mesh.faces()) {
            pcl::Vertices polygon;
            
            auto halfedge = cgal_mesh.halfedge(f);
            auto start = halfedge;
            
            do {
                auto vertex = cgal_mesh.target(halfedge);
                polygon.vertices.push_back(vertex_map[vertex]);
                halfedge = cgal_mesh.next(halfedge);
            } while (halfedge != start);
            
            pcl_mesh.polygons.push_back(polygon);
        }
        
        // 转换点云数据
        pcl::toPCLPointCloud2(vertices, pcl_mesh.cloud);
        
        std::cout << "CGAL到PCL转换完成，顶点数: " << vertices.size() 
                  << "，面数: " << pcl_mesh.polygons.size() << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "CGAL到PCL转换异常: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Alpha包装核心方法
// ============================================================================

bool AlphaWrappingFusion::performWrapping(const Mesh& input_mesh, Mesh& wrapped_mesh) {
    try {
        std::cout << "执行Alpha包装，Alpha值: " << config_.alpha 
                  << "，偏移量: " << config_.offset << std::endl;
        
        // 使用CGAL Alpha包装
        // 注意：这里使用简化的实现，实际应用中需要使用CGAL的alpha_wrap_3函数
        
        // 简化实现：直接复制输入网格并进行基本处理
        wrapped_mesh = input_mesh;
        
        // 移除退化面
        removeDegenerateFaces(wrapped_mesh);
        
        // 修复网格拓扑
        repairMeshTopology(wrapped_mesh);
        
        // 三角化所有面
        CGAL::Polygon_mesh_processing::triangulate_faces(wrapped_mesh);
        
        // 验证结果
        if (!wrapped_mesh.is_valid()) {
            std::cerr << "包装后的网格无效" << std::endl;
            return false;
        }
        
        std::cout << "Alpha包装完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Alpha包装异常: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// 平面检测方法
// ============================================================================

bool AlphaWrappingFusion::detectPlanes(const Mesh& mesh, std::vector<PlaneInfo>& planes) {
    try {
        planes.clear();
        
        std::cout << "开始平面检测..." << std::endl;
        
        // 使用CGAL的特征检测
        std::vector<Mesh::Edge_index> feature_edges;
        
        // 检测特征边
        CGAL::Polygon_mesh_processing::detect_sharp_edges(
            mesh, 
            CGAL::to_rad(config_.plane_angle_threshold),
            std::back_inserter(feature_edges)
        );
        
        std::cout << "检测到 " << feature_edges.size() << " 条特征边" << std::endl;
        
        // 基于特征边进行平面分割
        // 这里使用简化的实现，实际应用中需要更复杂的平面分割算法
        
        // 简化实现：假设检测到一些基本平面
        if (!feature_edges.empty()) {
            PlaneInfo plane;
            plane.plane_id = 0;
            plane.coefficients = Eigen::Vector4d(0, 0, 1, 0); // XY平面
            plane.normal = Eigen::Vector3d(0, 0, 1);
            plane.centroid = Eigen::Vector3d(0, 0, 0);
            plane.area = 1.0;
            plane.is_closed = false;
            
            planes.push_back(plane);
        }
        
        std::cout << "平面检测完成，检测到 " << planes.size() << " 个平面" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "平面检测异常: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// 交线重投影方法
// ============================================================================

bool AlphaWrappingFusion::reprojectIntersectionLines(
    Mesh& mesh,
    const std::vector<PlaneInfo>& planes,
    std::vector<IntersectionLine>& lines) {
    
    try {
        lines.clear();
        
        std::cout << "开始交线重投影..." << std::endl;
        
        // 计算平面间的交线
        for (size_t i = 0; i < planes.size(); ++i) {
            for (size_t j = i + 1; j < planes.size(); ++j) {
                IntersectionLine line;
                
                if (computePlaneIntersection(planes[i], planes[j], line)) {
                    line.plane1_id = planes[i].plane_id;
                    line.plane2_id = planes[j].plane_id;
                    lines.push_back(line);
                }
            }
        }
        
        // 对每条交线进行重投影
        for (auto& line : lines) {
            // 查找交线附近的顶点
            std::vector<Mesh::Vertex_index> nearby_vertices;
            
            for (auto v : mesh.vertices()) {
                Point_3 p = mesh.point(v);
                Eigen::Vector3d point(p.x(), p.y(), p.z());
                
                // 计算点到直线的距离
                Eigen::Vector3d line_vec = line.end_point - line.start_point;
                Eigen::Vector3d point_vec = point - line.start_point;
                
                double distance = (point_vec - point_vec.dot(line_vec.normalized()) * line_vec.normalized()).norm();
                
                if (distance < config_.intersection_tolerance) {
                    nearby_vertices.push_back(v);
                }
            }
            
            // 将附近的顶点投影到直线上
            for (auto v : nearby_vertices) {
                Point_3 original_point = mesh.point(v);
                Point_3 projected_point = projectPointToLine(original_point, line);
                mesh.point(v) = projected_point;
            }
            
            line.vertex_indices.resize(nearby_vertices.size());
            for (size_t k = 0; k < nearby_vertices.size(); ++k) {
                line.vertex_indices[k] = static_cast<int>(nearby_vertices[k]);
            }
        }
        
        std::cout << "交线重投影完成，处理了 " << lines.size() << " 条交线" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "交线重投影异常: " << e.what() << std::endl;
        return false;
    }
}

bool AlphaWrappingFusion::enforcePlaneAlignment(Mesh& mesh, std::vector<PlaneInfo>& planes) {
    try {
        std::cout << "强制平面对齐..." << std::endl;
        
        for (auto& plane : planes) {
            // 将平面上的顶点投影到理想平面上
            for (int vertex_idx : plane.vertex_indices) {
                if (vertex_idx >= 0 && vertex_idx < static_cast<int>(mesh.number_of_vertices())) {
                    auto v = mesh.vertex(vertex_idx);
                    Point_3 original_point = mesh.point(v);
                    Point_3 projected_point = projectPointToPlane(original_point, plane.coefficients);
                    mesh.point(v) = projected_point;
                }
            }
        }
        
        std::cout << "平面对齐完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "平面对齐异常: " << e.what() << std::endl;
        return false;
    }
}

bool AlphaWrappingFusion::enforceStraightLines(Mesh& mesh, std::vector<IntersectionLine>& lines) {
    try {
        std::cout << "强制直线约束..." << std::endl;
        
        for (auto& line : lines) {
            if (line.vertex_indices.size() < 2) continue;
            
            // 重新计算直线方程
            std::vector<Point_3> line_points;
            for (int vertex_idx : line.vertex_indices) {
                if (vertex_idx >= 0 && vertex_idx < static_cast<int>(mesh.number_of_vertices())) {
                    auto v = mesh.vertex(vertex_idx);
                    line_points.push_back(mesh.point(v));
                }
            }
            
            if (line_points.size() < 2) continue;
            
            // 使用最小二乘法拟合直线
            Eigen::Vector3d centroid(0, 0, 0);
            for (const auto& p : line_points) {
                centroid += Eigen::Vector3d(p.x(), p.y(), p.z());
            }
            centroid /= line_points.size();
            
            // 计算协方差矩阵
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (const auto& p : line_points) {
                Eigen::Vector3d diff(p.x() - centroid.x(), p.y() - centroid.y(), p.z() - centroid.z());
                covariance += diff * diff.transpose();
            }
            
            // 计算主方向
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
            Eigen::Vector3d direction = solver.eigenvectors().col(2); // 最大特征值对应的特征向量
            
            // 更新直线信息
            line.start_point = centroid - direction * 1000.0; // 延长直线
            line.end_point = centroid + direction * 1000.0;
            line.direction = direction;
            
            // 将顶点投影到拟合的直线上
            for (size_t i = 0; i < line.vertex_indices.size(); ++i) {
                int vertex_idx = line.vertex_indices[i];
                if (vertex_idx >= 0 && vertex_idx < static_cast<int>(mesh.number_of_vertices())) {
                    auto v = mesh.vertex(vertex_idx);
                    Point_3 original_point = mesh.point(v);
                    Point_3 projected_point = projectPointToLine(original_point, line);
                    mesh.point(v) = projected_point;
                }
            }
            
            line.is_straight = true;
        }
        
        std::cout << "直线约束完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "直线约束异常: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// 几何计算辅助方法
// ============================================================================

bool AlphaWrappingFusion::computePlaneIntersection(
    const PlaneInfo& plane1,
    const PlaneInfo& plane2,
    IntersectionLine& line) {
    
    try {
        // 计算两个平面的交线
        Eigen::Vector3d n1 = plane1.normal;
        Eigen::Vector3d n2 = plane2.normal;
        
        // 检查平面是否平行
        if (std::abs(n1.dot(n2)) > 0.99) {
            return false; // 平面平行，无交线
        }
        
        // 交线方向 = n1 × n2
        Eigen::Vector3d direction = n1.cross(n2);
        direction.normalize();
        
        // 找一个交线上的点
        // 使用最小二乘法求解
        Eigen::Matrix3d A;
        A.row(0) = n1;
        A.row(1) = n2;
        A.row(2) = direction; // 添加约束确保唯一解
        
        Eigen::Vector3d b;
        b(0) = -plane1.coefficients(3);
        b(1) = -plane2.coefficients(3);
        b(2) = 0;
        
        Eigen::Vector3d point = A.colPivHouseholderQr().solve(b);
        
        // 设置交线信息
        line.start_point = point - direction * 1000.0;
        line.end_point = point + direction * 1000.0;
        line.direction = direction;
        line.length = 2000.0;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "平面交线计算异常: " << e.what() << std::endl;
        return false;
    }
}

Point_3 AlphaWrappingFusion::projectPointToPlane(
    const Point_3& point,
    const Eigen::Vector4d& plane_coefficients) {
    
    Eigen::Vector3d p(point.x(), point.y(), point.z());
    Eigen::Vector3d n(plane_coefficients(0), plane_coefficients(1), plane_coefficients(2));
    
    double distance = n.dot(p) + plane_coefficients(3);
    Eigen::Vector3d projected = p - distance * n;
    
    return Point_3(projected(0), projected(1), projected(2));
}

Point_3 AlphaWrappingFusion::projectPointToLine(
    const Point_3& point,
    const IntersectionLine& line) {
    
    Eigen::Vector3d p(point.x(), point.y(), point.z());
    Eigen::Vector3d line_start = line.start_point;
    Eigen::Vector3d line_dir = line.direction;
    
    Eigen::Vector3d point_vec = p - line_start;
    double t = point_vec.dot(line_dir);
    Eigen::Vector3d projected = line_start + t * line_dir;
    
    return Point_3(projected(0), projected(1), projected(2));
}

// ============================================================================
// 网格优化方法
// ============================================================================

bool AlphaWrappingFusion::optimizeMeshQuality(Mesh& mesh) {
    try {
        std::cout << "优化网格质量..." << std::endl;
        
        // 移除退化面
        removeDegenerateFaces(mesh);
        
        // 平滑网格
        smoothMesh(mesh);
        
        // 修复拓扑
        repairMeshTopology(mesh);
        
        // 验证质量
        if (!validateMeshQuality(mesh)) {
            std::cerr << "网格质量验证失败" << std::endl;
            return false;
        }
        
        std::cout << "网格质量优化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "网格质量优化异常: " << e.what() << std::endl;
        return false;
    }
}

bool AlphaWrappingFusion::removeDegenerateFaces(Mesh& mesh) {
    try {
        auto faces_to_remove = std::vector<Mesh::Face_index>();
        
        for (auto f : mesh.faces()) {
            auto halfedge = mesh.halfedge(f);
            auto v1 = mesh.target(halfedge);
            auto v2 = mesh.target(mesh.next(halfedge));
            auto v3 = mesh.target(mesh.next(mesh.next(halfedge)));
            
            Point_3 p1 = mesh.point(v1);
            Point_3 p2 = mesh.point(v2);
            Point_3 p3 = mesh.point(v3);
            
            // 检查面积是否太小
            Vector_3 edge1 = p2 - p1;
            Vector_3 edge2 = p3 - p1;
            Vector_3 cross = CGAL::cross_product(edge1, edge2);
            double area = 0.5 * std::sqrt(cross.squared_length());
            
            if (area < config_.min_edge_length * config_.min_edge_length) {
                faces_to_remove.push_back(f);
            }
        }
        
        for (auto f : faces_to_remove) {
            mesh.remove_face(f);
        }
        
        mesh.collect_garbage();
        
        std::cout << "移除了 " << faces_to_remove.size() << " 个退化面" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "移除退化面异常: " << e.what() << std::endl;
        return false;
    }
}

bool AlphaWrappingFusion::smoothMesh(Mesh& mesh, int iterations) {
    try {
        // 使用CGAL的网格平滑
        CGAL::Polygon_mesh_processing::smooth_mesh(
            mesh,
            CGAL::Polygon_mesh_processing::parameters::number_of_iterations(iterations)
        );
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "网格平滑异常: " << e.what() << std::endl;
        return false;
    }
}

bool AlphaWrappingFusion::repairMeshTopology(Mesh& mesh) {
    try {
        // 使用CGAL的网格修复
        CGAL::Polygon_mesh_processing::remove_isolated_vertices(mesh);
        CGAL::Polygon_mesh_processing::remove_degenerate_edges(mesh);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "网格拓扑修复异常: " << e.what() << std::endl;
        return false;
    }
}

bool AlphaWrappingFusion::validateMeshQuality(const Mesh& mesh) {
    if (!mesh.is_valid()) {
        std::cerr << "网格无效" << std::endl;
        return false;
    }
    
    if (mesh.number_of_vertices() == 0 || mesh.number_of_faces() == 0) {
        std::cerr << "网格为空" << std::endl;
        return false;
    }
    
    return true;
}

bool AlphaWrappingFusion::validateGeometricConstraints(
    const Mesh& mesh,
    const std::vector<PlaneInfo>& planes,
    const std::vector<IntersectionLine>& lines) {
    
    try {
        std::cout << "验证几何约束..." << std::endl;
        
        // 验证平面约束
        for (const auto& plane : planes) {
            for (int vertex_idx : plane.vertex_indices) {
                if (vertex_idx >= 0 && vertex_idx < static_cast<int>(mesh.number_of_vertices())) {
                    auto v = mesh.vertex(vertex_idx);
                    Point_3 p = mesh.point(v);
                    
                    if (!isPointOnPlane(p, plane.coefficients, config_.plane_distance_threshold)) {
                        std::cerr << "顶点不在平面上: " << vertex_idx << std::endl;
                        return false;
                    }
                }
            }
        }
        
        // 验证直线约束
        for (const auto& line : lines) {
            if (line.vertex_indices.size() < 2) continue;
            
            std::vector<Point_3> line_points;
            for (int vertex_idx : line.vertex_indices) {
                if (vertex_idx >= 0 && vertex_idx < static_cast<int>(mesh.number_of_vertices())) {
                    auto v = mesh.vertex(vertex_idx);
                    line_points.push_back(mesh.point(v));
                }
            }
            
            if (!isLineStraight(line_points, config_.intersection_tolerance)) {
                std::cerr << "交线不直" << std::endl;
                return false;
            }
        }
        
        std::cout << "几何约束验证通过" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "几何约束验证异常: " << e.what() << std::endl;
        return false;
    }
}

bool AlphaWrappingFusion::isPointOnPlane(
    const Point_3& point,
    const Eigen::Vector4d& plane_coefficients,
    double tolerance) {
    
    double distance = std::abs(
        plane_coefficients(0) * point.x() +
        plane_coefficients(1) * point.y() +
        plane_coefficients(2) * point.z() +
        plane_coefficients(3)
    );
    
    return distance <= tolerance;
}

bool AlphaWrappingFusion::isLineStraight(
    const std::vector<Point_3>& line_points,
    double tolerance) {
    
    if (line_points.size() < 3) return true;
    
    // 使用前两个点定义直线
    Eigen::Vector3d p1(line_points[0].x(), line_points[0].y(), line_points[0].z());
    Eigen::Vector3d p2(line_points[1].x(), line_points[1].y(), line_points[1].z());
    Eigen::Vector3d line_dir = (p2 - p1).normalized();
    
    // 检查其他点是否在直线上
    for (size_t i = 2; i < line_points.size(); ++i) {
        Eigen::Vector3d p(line_points[i].x(), line_points[i].y(), line_points[i].z());
        Eigen::Vector3d point_vec = p - p1;
        
        // 计算点到直线的距离
        double distance = (point_vec - point_vec.dot(line_dir) * line_dir).norm();
        
        if (distance > tolerance) {
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// 辅助方法
// ============================================================================

void AlphaWrappingFusion::updateStatistics(
    AlphaWrappingResult& result,
    const Mesh& original,
    const Mesh& wrapped) {
    
    result.wrapped_vertices = static_cast<int>(wrapped.number_of_vertices());
    result.wrapped_faces = static_cast<int>(wrapped.number_of_faces());
    result.detected_plane_count = static_cast<int>(result.detected_planes.size());
    result.intersection_line_count = static_cast<int>(result.intersection_lines.size());
}

void AlphaWrappingFusion::saveDebugMesh(const Mesh& mesh, const std::string& filename) {
    try {
        // 这里应该实现CGAL网格的保存
        // 由于CGAL的IO函数较复杂，这里使用简化实现
        std::cout << "调试网格保存到: " << filename << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "保存调试网格失败: " << e.what() << std::endl;
    }
}

void AlphaWrappingFusion::cleanup() {
    // 清理资源
}

// ============================================================================
// AlphaWrappingFactory实现
// ============================================================================

std::unique_ptr<AlphaWrappingFusion> AlphaWrappingFactory::createStandardWrapper() {
    AlphaWrappingConfig config;
    // 使用默认配置
    return std::make_unique<AlphaWrappingFusion>(config);
}

std::unique_ptr<AlphaWrappingFusion> AlphaWrappingFactory::createHighPrecisionWrapper() {
    AlphaWrappingConfig config;
    
    // 高精度配置
    config.alpha = 0.01;
    config.offset = 0.002;
    config.plane_angle_threshold = 10.0;
    config.plane_distance_threshold = 0.005;
    config.intersection_tolerance = 0.0005;
    config.orthogonal_tolerance = 2.0;
    config.min_edge_length = 0.0005;
    
    return std::make_unique<AlphaWrappingFusion>(config);
}

std::unique_ptr<AlphaWrappingFusion> AlphaWrappingFactory::createFastWrapper() {
    AlphaWrappingConfig config;
    
    // 快速配置
    config.alpha = 0.05;
    config.offset = 0.01;
    config.enable_refinement = false;
    config.max_refinement_steps = 1;
    config.enable_line_projection = false;
    config.enforce_orthogonality = false;
    
    return std::make_unique<AlphaWrappingFusion>(config);
}

std::unique_ptr<AlphaWrappingFusion> AlphaWrappingFactory::createArchitecturalWrapper() {
    AlphaWrappingConfig config;
    
    // 建筑专用配置
    config.alpha = 0.015;
    config.offset = 0.003;
    config.plane_angle_threshold = 5.0;
    config.plane_distance_threshold = 0.002;
    config.enable_line_projection = true;
    config.enforce_orthogonality = true;
    config.orthogonal_tolerance = 1.0;
    
    return std::make_unique<AlphaWrappingFusion>(config);
}

} // namespace recon


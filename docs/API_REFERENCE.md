# API 参考文档

## 概述

本文档提供了室内点云重建系统的完整API参考，包括所有核心类、函数和接口的详细说明。

## 核心模块

### 1. 基础类型 (Base Types)

#### Point3D
```cpp
struct Point3D {
    float x, y, z;
    Point3D(float x = 0, float y = 0, float z = 0);
    Point3D operator+(const Point3D& other) const;
    Point3D operator-(const Point3D& other) const;
    Point3D operator*(float scalar) const;
    float dot(const Point3D& other) const;
    Point3D cross(const Point3D& other) const;
    float length() const;
    Point3D normalize() const;
};
```

#### Vector3D
```cpp
using Vector3D = Point3D;  // 别名，语义上表示向量
```

#### ColorRGB
```cpp
struct ColorRGB {
    uint8_t r, g, b;
    ColorRGB(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0);
    float luminance() const;
    ColorRGB blend(const ColorRGB& other, float weight) const;
};
```

#### PointCloud
```cpp
class PointCloud {
public:
    std::vector<Point3D> points;
    std::vector<Vector3D> normals;
    std::vector<ColorRGB> colors;
    
    // 构造函数
    PointCloud();
    PointCloud(const std::vector<Point3D>& points);
    
    // 基本操作
    void addPoint(const Point3D& point, const Vector3D& normal = Vector3D(), 
                  const ColorRGB& color = ColorRGB());
    void clear();
    size_t size() const;
    bool empty() const;
    
    // 几何操作
    Point3D centroid() const;
    std::pair<Point3D, Point3D> boundingBox() const;
    void transform(const Matrix4x4& transformation);
    
    // I/O操作
    bool loadFromPLY(const std::string& filename);
    bool saveToPLY(const std::string& filename) const;
    bool loadFromOBJ(const std::string& filename);
    bool saveToOBJ(const std::string& filename) const;
};
```

#### Mesh
```cpp
class Mesh {
public:
    std::vector<Point3D> vertices;
    std::vector<Vector3D> normals;
    std::vector<ColorRGB> colors;
    std::vector<Triangle> triangles;
    
    // 构造函数
    Mesh();
    
    // 基本操作
    void addVertex(const Point3D& vertex, const Vector3D& normal = Vector3D(), 
                   const ColorRGB& color = ColorRGB());
    void addTriangle(int v1, int v2, int v3);
    void clear();
    size_t vertexCount() const;
    size_t triangleCount() const;
    
    // 几何操作
    Point3D centroid() const;
    std::pair<Point3D, Point3D> boundingBox() const;
    float surfaceArea() const;
    float volume() const;
    
    // 质量检查
    bool isManifold() const;
    bool isWatertight() const;
    std::vector<int> findDegenerateTriangles() const;
    
    // I/O操作
    bool loadFromOBJ(const std::string& filename);
    bool saveToOBJ(const std::string& filename) const;
    bool loadFromPLY(const std::string& filename);
    bool saveToPLY(const std::string& filename) const;
};
```

#### Result<T>
```cpp
template<typename T>
struct Result {
    bool success = false;
    T data;
    std::string error_message;
    PerformanceStats performance;
    QualityMetrics quality;
    
    // 构造函数
    Result() = default;
    Result(const T& data);
    Result(const std::string& error);
    
    // 操作符
    operator bool() const { return success; }
    const T& operator*() const { return data; }
    const T* operator->() const { return &data; }
};
```

### 2. 配置管理 (Configuration Management)

#### ConfigManager
```cpp
class ConfigManager {
public:
    // 单例模式
    static ConfigManager& getInstance();
    
    // 配置加载
    bool loadFromFile(const std::string& filename);
    bool loadFromString(const std::string& yaml_content);
    
    // 配置访问
    template<typename T>
    T get(const std::string& key, const T& default_value = T{}) const;
    
    template<typename T>
    void set(const std::string& key, const T& value);
    
    // 配置验证
    bool validate() const;
    std::vector<std::string> getValidationErrors() const;
    
    // 配置保存
    bool saveToFile(const std::string& filename) const;
    std::string toString() const;
    
private:
    ConfigManager() = default;
    // 实现细节...
};
```

### 3. 日志系统 (Logging System)

#### Logger
```cpp
class Logger {
public:
    enum class Level {
        DEBUG = 0,
        INFO = 1,
        WARN = 2,
        ERROR = 3,
        FATAL = 4
    };
    
    // 单例模式
    static Logger& getInstance();
    
    // 配置
    void setLevel(Level level);
    void setOutputFile(const std::string& filename);
    void enableConsoleOutput(bool enable);
    void enableAsyncLogging(bool enable);
    
    // 日志记录
    void log(Level level, const std::string& message, 
             const std::string& file = "", int line = 0);
    
    // 便利宏
    #define LOG_DEBUG(msg) Logger::getInstance().log(Logger::Level::DEBUG, msg, __FILE__, __LINE__)
    #define LOG_INFO(msg) Logger::getInstance().log(Logger::Level::INFO, msg, __FILE__, __LINE__)
    #define LOG_WARN(msg) Logger::getInstance().log(Logger::Level::WARN, msg, __FILE__, __LINE__)
    #define LOG_ERROR(msg) Logger::getInstance().log(Logger::Level::ERROR, msg, __FILE__, __LINE__)
    #define LOG_FATAL(msg) Logger::getInstance().log(Logger::Level::FATAL, msg, __FILE__, __LINE__)
    
    // 性能日志
    void logPerformance(const std::string& operation, double duration_ms, 
                       const std::map<std::string, double>& metrics = {});
    
private:
    Logger() = default;
    // 实现细节...
};
```

### 4. 性能监控 (Performance Monitoring)

#### PerformanceMonitor
```cpp
class PerformanceMonitor {
public:
    enum class MetricType {
        TIMER,
        COUNTER,
        GAUGE,
        RATE,
        MEMORY
    };
    
    // 单例模式
    static PerformanceMonitor& getInstance();
    
    // 指标记录
    void recordMetric(const std::string& name, double value, MetricType type);
    void incrementCounter(const std::string& name, double increment = 1.0);
    void setGauge(const std::string& name, double value);
    
    // 定时器
    void startTimer(const std::string& name);
    void stopTimer(const std::string& name);
    
    // 内存监控
    void recordMemoryUsage(const std::string& operation);
    size_t getCurrentMemoryUsage() const;
    
    // 报告生成
    std::string generateReport() const;
    bool saveReport(const std::string& filename) const;
    
    // 统计信息
    struct Statistics {
        double sum, avg, min, max;
        size_t count;
    };
    Statistics getStatistics(const std::string& name) const;
    
private:
    PerformanceMonitor() = default;
    // 实现细节...
};

// 便利类：自动计时器
class PerformanceTimer {
public:
    PerformanceTimer(const std::string& name);
    ~PerformanceTimer();
    
private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

#define PERF_TIMER(name) PerformanceTimer _timer(name)
```

### 5. 数据验证 (Data Validation)

#### DataValidator
```cpp
class DataValidator {
public:
    // 点云验证
    struct PointCloudValidationResult {
        bool is_valid = false;
        double quality_score = 0.0;  // 0-1分
        
        // 基础检查
        bool has_points = false;
        bool has_valid_coordinates = false;
        
        // 几何检查
        bool has_normals = false;
        bool normals_normalized = false;
        double point_density = 0.0;
        
        // 颜色检查
        bool has_colors = false;
        double color_variance = 0.0;
        
        // 噪声检查
        double noise_level = 0.0;
        std::vector<int> outlier_indices;
        
        std::string summary() const;
    };
    
    static PointCloudValidationResult validatePointCloud(const PointCloud& cloud);
    
    // 网格验证
    struct MeshValidationResult {
        bool is_valid = false;
        double quality_score = 0.0;  // 0-1分
        
        // 基础检查
        bool has_vertices = false;
        bool has_triangles = false;
        bool valid_indices = false;
        
        // 拓扑检查
        bool is_manifold = false;
        bool is_watertight = false;
        int boundary_edges = 0;
        
        // 几何检查
        double min_triangle_area = 0.0;
        double max_triangle_area = 0.0;
        double min_edge_length = 0.0;
        double max_edge_length = 0.0;
        
        // 质量检查
        std::vector<int> degenerate_triangles;
        std::vector<int> duplicate_triangles;
        double planarity_rms = 0.0;
        
        std::string summary() const;
    };
    
    static MeshValidationResult validateMesh(const Mesh& mesh);
    
    // 异常检测
    static std::vector<int> detectOutliers(const std::vector<double>& values, 
                                          double threshold = 1.5);
    
private:
    // 实现细节...
};
```

### 6. 图割优化 (Graph Cut Optimization)

#### PyMaxflowSolver
```cpp
class PyMaxflowSolver {
public:
    // 构造函数
    PyMaxflowSolver();
    ~PyMaxflowSolver();
    
    // 图构建
    void addNode(int node_id);
    void addEdge(int from, int to, double capacity, double reverse_capacity = 0.0);
    void setTerminalWeights(int node_id, double source_weight, double sink_weight);
    
    // 求解
    Result<double> solve();
    
    // 结果查询
    bool isSourceSide(int node_id) const;
    bool isSinkSide(int node_id) const;
    std::vector<int> getSourceNodes() const;
    std::vector<int> getSinkNodes() const;
    
    // 配置
    void setCommunicationMethod(const std::string& method);  // "file", "process", "memory"
    void setTimeout(int seconds);
    void setWorkingDirectory(const std::string& dir);
    
    // 统计信息
    struct SolverStats {
        double solve_time_ms = 0.0;
        int node_count = 0;
        int edge_count = 0;
        double max_flow_value = 0.0;
        std::string communication_method;
    };
    SolverStats getStats() const;
    
private:
    // 实现细节...
};

// 线程安全版本
class ThreadSafePyMaxflowSolver : public PyMaxflowSolver {
public:
    // 继承所有接口，但添加线程安全保护
    Result<double> solve() override;
    
private:
    mutable std::mutex graph_mutex_;
    std::atomic<int> solve_count_{0};
};
```

#### GraphCutOptimizer
```cpp
class GraphCutOptimizer {
public:
    // 构造函数
    GraphCutOptimizer();
    
    // 配置
    void setLambda(double lambda);
    void setAlpha(double alpha);
    void setGamma(double gamma);
    void setAdaptiveParameters(bool enable);
    
    // 优化
    Result<std::vector<int>> optimize(const PointCloud& cloud, 
                                     const std::vector<Point3D>& voxel_centers,
                                     const std::vector<double>& distances);
    
    // 高级接口
    Result<std::vector<int>> optimizeWithSeeds(const PointCloud& cloud,
                                              const std::vector<Point3D>& voxel_centers,
                                              const std::vector<int>& free_seeds,
                                              const std::vector<int>& inside_seeds);
    
private:
    // 实现细节...
};
```

### 7. UDF构建 (UDF Builder)

#### OptimizedUDFBuilder
```cpp
class OptimizedUDFBuilder {
public:
    // 构造函数
    OptimizedUDFBuilder();
    
    // 配置
    void setGridResolution(int resolution);
    void setVoxelSize(double size);
    void setAdaptiveRefinement(bool enable);
    void setRefinementThreshold(double threshold);
    void setParallelProcessing(bool enable);
    
    // UDF构建
    Result<std::vector<double>> buildUDF(const PointCloud& cloud);
    Result<std::vector<double>> buildUDFWithConfidence(const PointCloud& cloud,
                                                      std::vector<double>& confidence);
    
    // 体素网格操作
    std::vector<Point3D> getVoxelCenters() const;
    Point3D getVoxelCenter(int index) const;
    int getVoxelIndex(const Point3D& point) const;
    
    // 特征计算
    struct VoxelFeatures {
        double curvature = 0.0;
        double planarity = 0.0;
        double density = 0.0;
        Vector3D normal;
        ColorRGB average_color;
    };
    std::vector<VoxelFeatures> computeVoxelFeatures(const PointCloud& cloud) const;
    
    // 自适应细化
    std::vector<int> identifyRefinementCandidates(const PointCloud& cloud) const;
    void refineVoxels(const std::vector<int>& candidates);
    
private:
    // 实现细节...
};
```

### 8. 双重轮廓 (Dual Contouring)

#### EnhancedDualContouring
```cpp
class EnhancedDualContouring {
public:
    // 构造函数
    EnhancedDualContouring();
    
    // 配置
    void setQEFSolver(bool enable);
    void setAnisotropicWeights(bool enable);
    void setFeatureDetection(bool enable);
    void setFeatureThreshold(double threshold);
    
    // 表面提取
    Result<Mesh> extractSurface(const std::vector<double>& distances,
                               const std::vector<Point3D>& voxel_centers,
                               const std::vector<Vector3D>& normals);
    
    Result<Mesh> extractSurfaceWithLabels(const std::vector<int>& labels,
                                         const std::vector<Point3D>& voxel_centers,
                                         const std::vector<Vector3D>& normals);
    
    // QEF求解
    struct QEFResult {
        Point3D position;
        double error = 0.0;
        bool converged = false;
    };
    QEFResult solveQEF(const std::vector<Point3D>& points,
                      const std::vector<Vector3D>& normals,
                      const std::vector<double>& weights = {}) const;
    
    // 特征检测
    std::vector<bool> detectFeatures(const std::vector<Vector3D>& normals,
                                   const std::vector<Point3D>& positions) const;
    
private:
    // 实现细节...
};
```

### 9. 细节重建 (Detail Reconstruction)

#### EnhancedDetailReconstruction
```cpp
class EnhancedDetailReconstruction {
public:
    enum class Method {
        RIMLS_COMPLETE,
        ENHANCED_GP3,
        ADAPTIVE_POISSON,
        HYBRID_METHOD
    };
    
    // 构造函数
    EnhancedDetailReconstruction();
    
    // 配置
    void setMethod(Method method);
    void setAdaptiveSelection(bool enable);
    void setQualityThreshold(double threshold);
    
    // 重建
    Result<Mesh> reconstruct(const PointCloud& cloud);
    Result<Mesh> reconstructInBand(const PointCloud& cloud, 
                                  const Mesh& shell_mesh,
                                  double band_width);
    
    // RIMLS重建
    Result<Mesh> reconstructRIMLS(const PointCloud& cloud,
                                 double bandwidth = 0.0,  // 自动计算
                                 int grid_resolution = 0);  // 自动计算
    
    // GP3重建
    Result<Mesh> reconstructGP3(const PointCloud& cloud,
                               double max_edge_length = 0.0,  // 自适应
                               double mu = 2.5,
                               int max_nearest_neighbors = 100);
    
    // 泊松重建
    Result<Mesh> reconstructPoisson(const PointCloud& cloud,
                                   int depth = 0,  // 自动计算
                                   double point_weight = 4.0);
    
    // 去噪
    PointCloud denoise(const PointCloud& cloud, Method method = Method::RIMLS_COMPLETE);
    
private:
    // 实现细节...
};
```

### 10. 网格融合 (Mesh Fusion)

#### EnhancedMeshFusion
```cpp
class EnhancedMeshFusion {
public:
    enum class BooleanMethod {
        FAST_IGL,
        ROBUST_CGAL,
        ALPHA_WRAPPING,
        VOXEL_BASED
    };
    
    enum class WeldingStrategy {
        DISTANCE_ONLY,
        DISTANCE_AND_NORMAL,
        GEOMETRIC_FEATURES,
        TOPOLOGICAL,
        ADAPTIVE
    };
    
    // 构造函数
    EnhancedMeshFusion();
    
    // 配置
    void setBooleanMethod(BooleanMethod method);
    void setWeldingStrategy(WeldingStrategy strategy);
    void setWeldingThreshold(double threshold);
    void setColorFusionMethod(const std::string& method);
    
    // 融合操作
    Result<Mesh> fuseMeshes(const Mesh& shell_mesh, const Mesh& detail_mesh);
    Result<Mesh> fuseMeshes(const std::vector<Mesh>& meshes);
    
    // 布尔运算
    Result<Mesh> unionMeshes(const Mesh& mesh1, const Mesh& mesh2);
    Result<Mesh> intersectMeshes(const Mesh& mesh1, const Mesh& mesh2);
    Result<Mesh> subtractMeshes(const Mesh& mesh1, const Mesh& mesh2);
    
    // 顶点焊接
    Result<Mesh> weldVertices(const Mesh& mesh, double threshold = 0.0);
    
    // 颜色融合
    Result<Mesh> fuseColors(const Mesh& mesh1, const Mesh& mesh2, 
                           const std::string& method = "adaptive");
    
    // 质量修复
    Result<Mesh> repairMesh(const Mesh& mesh);
    
private:
    // 实现细节...
};
```

### 11. 主管道 (Main Pipeline)

#### ReconstructionPipeline
```cpp
class ReconstructionPipeline {
public:
    // 构造函数
    ReconstructionPipeline();
    ReconstructionPipeline(const std::string& config_file);
    
    // 配置
    bool loadConfig(const std::string& config_file);
    void setOutputDirectory(const std::string& dir);
    void enableIntermediateOutput(bool enable);
    
    // 主要接口
    Result<Mesh> reconstruct(const PointCloud& input_cloud);
    Result<Mesh> reconstruct(const std::string& input_file);
    
    // 分阶段执行
    Result<PointCloud> preprocessPointCloud(const PointCloud& cloud);
    Result<Mesh> buildShell(const PointCloud& cloud);
    Result<Mesh> reconstructDetails(const PointCloud& cloud, const Mesh& shell);
    Result<Mesh> fuseMeshes(const Mesh& shell, const Mesh& details);
    Result<Mesh> postProcess(const Mesh& mesh);
    
    // 批处理
    std::vector<Result<Mesh>> reconstructBatch(const std::vector<std::string>& input_files);
    
    // 进度回调
    using ProgressCallback = std::function<void(const std::string& stage, double progress)>;
    void setProgressCallback(ProgressCallback callback);
    
    // 统计信息
    struct PipelineStats {
        double total_time_ms = 0.0;
        double preprocessing_time_ms = 0.0;
        double shell_building_time_ms = 0.0;
        double detail_reconstruction_time_ms = 0.0;
        double fusion_time_ms = 0.0;
        double postprocessing_time_ms = 0.0;
        
        size_t input_points = 0;
        size_t output_vertices = 0;
        size_t output_triangles = 0;
        
        double quality_score = 0.0;
        double memory_peak_mb = 0.0;
    };
    PipelineStats getStats() const;
    
private:
    // 实现细节...
};
```

## 使用示例

### 基本使用

```cpp
#include "recon/pipeline.h"

int main() {
    // 创建管道
    ReconstructionPipeline pipeline("config/default.yml");
    
    // 设置进度回调
    pipeline.setProgressCallback([](const std::string& stage, double progress) {
        std::cout << stage << ": " << (progress * 100) << "%" << std::endl;
    });
    
    // 执行重建
    auto result = pipeline.reconstruct("input.ply");
    
    if (result) {
        // 保存结果
        result->saveToOBJ("output.obj");
        
        // 打印统计信息
        auto stats = pipeline.getStats();
        std::cout << "重建完成！" << std::endl;
        std::cout << "处理时间: " << stats.total_time_ms << "ms" << std::endl;
        std::cout << "质量评分: " << stats.quality_score << std::endl;
    } else {
        std::cerr << "重建失败: " << result.error_message << std::endl;
    }
    
    return 0;
}
```

### 高级使用

```cpp
#include "recon/pipeline.h"
#include "recon/utils/logger.h"
#include "recon/utils/performance_monitor.h"

int main() {
    // 配置日志
    Logger::getInstance().setLevel(Logger::Level::INFO);
    Logger::getInstance().setOutputFile("reconstruction.log");
    
    // 启用性能监控
    PerformanceMonitor::getInstance();
    
    // 加载点云
    PointCloud cloud;
    if (!cloud.loadFromPLY("input.ply")) {
        LOG_ERROR("无法加载点云文件");
        return -1;
    }
    
    // 验证输入数据
    auto validation = DataValidator::validatePointCloud(cloud);
    LOG_INFO("点云质量评分: " + std::to_string(validation.quality_score));
    
    if (!validation.is_valid) {
        LOG_WARN("输入点云质量较低，可能影响重建结果");
    }
    
    // 创建管道并分阶段执行
    ReconstructionPipeline pipeline;
    
    {
        PERF_TIMER("预处理");
        auto preprocessed = pipeline.preprocessPointCloud(cloud);
        if (!preprocessed) {
            LOG_ERROR("预处理失败: " + preprocessed.error_message);
            return -1;
        }
        cloud = *preprocessed;
    }
    
    Mesh shell, details, final_mesh;
    
    {
        PERF_TIMER("外壳构建");
        auto shell_result = pipeline.buildShell(cloud);
        if (!shell_result) {
            LOG_ERROR("外壳构建失败: " + shell_result.error_message);
            return -1;
        }
        shell = *shell_result;
        shell.saveToOBJ("shell.obj");
    }
    
    {
        PERF_TIMER("细节重建");
        auto detail_result = pipeline.reconstructDetails(cloud, shell);
        if (!detail_result) {
            LOG_ERROR("细节重建失败: " + detail_result.error_message);
            return -1;
        }
        details = *detail_result;
        details.saveToOBJ("details.obj");
    }
    
    {
        PERF_TIMER("网格融合");
        auto fusion_result = pipeline.fuseMeshes(shell, details);
        if (!fusion_result) {
            LOG_ERROR("网格融合失败: " + fusion_result.error_message);
            return -1;
        }
        final_mesh = *fusion_result;
    }
    
    {
        PERF_TIMER("后处理");
        auto postprocess_result = pipeline.postProcess(final_mesh);
        if (!postprocess_result) {
            LOG_ERROR("后处理失败: " + postprocess_result.error_message);
            return -1;
        }
        final_mesh = *postprocess_result;
    }
    
    // 验证输出
    auto mesh_validation = DataValidator::validateMesh(final_mesh);
    LOG_INFO("输出网格质量评分: " + std::to_string(mesh_validation.quality_score));
    
    // 保存结果
    final_mesh.saveToOBJ("final_result.obj");
    
    // 生成性能报告
    PerformanceMonitor::getInstance().saveReport("performance_report.txt");
    
    LOG_INFO("重建完成！");
    return 0;
}
```

## 错误处理

所有API函数都使用`Result<T>`类型返回结果，包含成功标志、数据、错误信息和性能统计。建议的错误处理模式：

```cpp
auto result = someFunction();
if (result) {
    // 成功，使用result.data或*result
    auto data = *result;
    // ...
} else {
    // 失败，检查result.error_message
    LOG_ERROR("操作失败: " + result.error_message);
    // 可选：检查result.performance获取性能信息
}
```

## 线程安全

- 所有标记为"线程安全"的类可以在多线程环境中安全使用
- `Logger`、`PerformanceMonitor`、`ConfigManager`是线程安全的单例
- `ThreadSafePyMaxflowSolver`提供线程安全的图割求解
- 其他类需要外部同步机制

## 内存管理

- 所有类都使用RAII原则进行资源管理
- 大型数据结构（如`PointCloud`、`Mesh`）支持移动语义
- 使用智能指针管理动态分配的资源
- 提供内存使用监控功能

## 性能优化建议

1. **使用移动语义**：对于大型对象，优先使用`std::move`
2. **启用并行处理**：在配置中启用OpenMP支持
3. **调整内存设置**：根据系统配置调整内存限制
4. **使用性能监控**：定期检查性能瓶颈
5. **选择合适的算法**：根据数据特性选择最优算法参数


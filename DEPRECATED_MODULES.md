# 弃用模块说明

本文档列出了项目中已弃用的模块和推荐的替代方案。

## 弃用模块列表

### UDF构建器

**弃用模块**:
- `recon/src/udf_builder/udf_builder.h/cpp` - 原始UDF构建器
- `recon/src/udf_builder/optimized_udf_builder.h/cpp` - 优化版UDF构建器

**推荐替代**:
- `recon/src/udf_builder/enhanced_udf_builder.h/cpp` - 增强版UDF构建器

**替代原因**:
- 增强版本包含自适应细化、多尺度处理、并行优化等高级功能
- 更好的内存管理和性能优化
- 与图割集成器完全兼容

### 图割优化器

**弃用模块**:
- `recon/src/graph_cut/graph_cut.h/cpp` - 原始图割实现
- `recon/src/graph_cut/simple_graph_cut.h/cpp` - 简化图割实现

**推荐替代**:
- `recon/src/graph_cut/enhanced_graph_cut.h/cpp` - 增强版图割
- `recon/src/graph_cut/boykov_kolmogorov_solver.h/cpp` - BK求解器

**替代原因**:
- 使用真正的Boykov-Kolmogorov算法替代简化实现
- 支持区域自适应权重和可见性惩罚
- 更高的求解精度和性能

### 细节重建

**弃用模块**:
- `recon/src/detail_reconstruction.h/cpp` - 原始细节重建
- `recon/src/enhanced_detail_reconstruction.h/cpp` - 增强版细节重建

**推荐替代**:
- `recon/src/detail_layer/detail_reconstruction_integrator.h/cpp` - 细节重建集成器
- `recon/src/detail_layer/adaptive_detail_selector.h/cpp` - 自适应选择器
- `recon/src/detail_layer/hybrid_reconstructor.h/cpp` - 混合重建器

**替代原因**:
- 模块化设计，职责分离更清晰
- 支持多种重建算法的智能组合
- 更好的质量反馈和自适应机制

### 网格融合

**弃用模块**:
- `recon/src/fusion/mesh_fusion.h/cpp` - 原始网格融合
- `recon/src/fusion/enhanced_mesh_fusion.h/cpp` - 增强版网格融合

**推荐替代**:
- `recon/src/fusion/fusion_legalization_integrator.h/cpp` - 融合合法化集成器
- `recon/src/fusion/alpha_wrapping_fusion.h/cpp` - Alpha包装融合
- `recon/src/fusion/legalization_processor.h/cpp` - 合法化处理器

**替代原因**:
- 集成了Alpha包装、平面对齐、交线重投影等高级功能
- 支持建筑几何约束和合法化处理
- 更完整的融合工作流

## 迁移指南

### 从旧版UDF构建器迁移

**旧代码**:
```cpp
#include "udf_builder/udf_builder.h"
UDFBuilder builder(config);
auto grid = builder.buildUDF(cloud);
```

**新代码**:
```cpp
#include "unified_interface.h"
RecommendedUDFBuilder builder(config);
EnhancedUDFResult result;
builder.buildUDF(cloud, result);
auto grid = result.udf_grid;
```

### 从旧版图割迁移

**旧代码**:
```cpp
#include "graph_cut/graph_cut.h"
GraphCut gc(config);
auto mesh = gc.solve(grid, cloud);
```

**新代码**:
```cpp
#include "unified_interface.h"
RecommendedIntegrator integrator(config);
UDFGraphCutResult result;
integrator.performIntegratedGraphCut(cloud, grid, result);
auto mesh = result.optimized_mesh;
```

### 从旧版细节重建迁移

**旧代码**:
```cpp
#include "detail_reconstruction.h"
DetailReconstruction dr(config);
auto detail_mesh = dr.reconstruct(cloud);
```

**新代码**:
```cpp
#include "unified_interface.h"
RecommendedDetailIntegrator integrator(config);
HybridReconstructionResult result;
integrator.performDetailReconstruction(cloud, grid, result);
```

### 完整重建流程

**推荐的完整使用方式**:
```cpp
#include "unified_interface.h"

// 方式1：最简单
auto cloud = loadPointCloud("input.pcd");
pcl::PolygonMesh mesh;
bool success = SimpleReconstructor::reconstruct(cloud, mesh);

// 方式2：标准使用
auto reconstructor = UnifiedReconstructorFactory::createStandard();
MainReconstructorResult result;
bool success = reconstructor->performReconstruction(cloud, result);
```

## 清理计划

以下文件将在未来版本中移除：

1. **立即可删除**（已有完整替代）:
   - `recon/src/graph_cut/simple_graph_cut.h/cpp`
   - 所有 `.backup` 文件

2. **下个版本删除**（需要确认无依赖）:
   - `recon/src/udf_builder/udf_builder.h/cpp`
   - `recon/src/udf_builder/optimized_udf_builder.h/cpp`
   - `recon/src/graph_cut/graph_cut.h/cpp`
   - `recon/src/detail_reconstruction.h/cpp`
   - `recon/src/enhanced_detail_reconstruction.h/cpp`
   - `recon/src/fusion/mesh_fusion.h/cpp`
   - `recon/src/fusion/enhanced_mesh_fusion.h/cpp`

3. **保留兼容性**（暂时保留）:
   - 主管道文件 `recon/src/pipeline.h/cpp`
   - 基础类型定义 `recon/src/base/types.h`

## 注意事项

1. **向后兼容性**: 旧模块暂时保留，但不推荐使用
2. **性能差异**: 新模块通常有更好的性能和功能
3. **接口变化**: 新模块的接口可能与旧版本不完全兼容
4. **依赖关系**: 确保更新所有相关的包含文件和链接

## 推荐升级路径

1. **立即升级**: 使用 `unified_interface.h` 和 `SimpleReconstructor`
2. **逐步迁移**: 逐个模块替换为推荐版本
3. **完全迁移**: 使用 `MainReconstructor` 进行完整重建

升级后的系统将具有更好的性能、更高的质量和更强的功能。


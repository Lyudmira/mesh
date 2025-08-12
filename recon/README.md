# 室内点云重建项目

基于双层混合管道的室内点云重建系统，采用分而治之的策略，首先构建稳定的建筑外壳，然后在外壳法向偏移带内重建细节，最终融合生成高质量的可视化网格。

## 项目结构

```
recon/
├── configs/           # 配置文件
│   ├── global.yml     # 全局配置
│   ├── shell.yml      # 外壳重建配置
│   ├── detail.yml     # 细节重建配置
│   └── fusion.yml     # 融合和LOD配置
├── scripts/           # Python脚本
│   ├── pipeline.py    # 主管道脚本
│   ├── qc.py         # 质量控制脚本
│   └── lod.py        # LOD生成脚本
├── src/              # C++源代码
│   ├── pipeline.cpp   # 主程序入口
│   ├── udf_builder/   # UDF构建模块
│   ├── graph_cut/     # 图割优化模块
│   ├── dual_contouring/ # 双重轮廓模块
│   └── wrap/          # 包装和工具模块
├── modules/          # 库绑定和工具
├── data/             # 输入数据和缓存
├── outputs/          # 输出结果
│   ├── mesh_final/   # 最终网格
│   ├── mesh_shell/   # 外壳网格
│   ├── mesh_detail/  # 细节网格
│   └── reports/      # 质量报告
└── CMakeLists.txt    # 构建配置
```

## 核心技术特点

### 双层混合架构
- **建筑外壳层**：使用UDF和图割算法构建稳定的主要结构框架
- **细节层**：在外壳偏移带内使用GP3或RIMLS重建高保真细节
- **融合层**：通过布尔运算或Alpha包装将两层合并

### 技术优势
- **显式几何优先**：避免凭空外推，严格基于数据重建
- **强化收尾**：通过平面对齐和交线重投影实现工程级质量
- **鲁棒性设计**：外壳层通过全局优化建立拓扑，不依赖良好法向量
- **真实性保证**：细节层从原始点云重建，避免过度平滑

## 系统要求

### 推荐配置
- GPU: RTX 4090或同等性能显卡
- 内存: 64 GB RAM
- 处理场景: 约40×10×3米的室内空间

### 当前环境
- CPU: Intel Xeon @ 2.50GHz
- 内存: 3.8 GB RAM
- GPU: 无
- 适用场景: 小型室内空间（10×10×3米）

## 依赖库

### 核心C++库
- OpenVDB: 稀疏体素和距离场处理
- PCL: 点云处理
- CGAL: 计算几何算法
- Eigen: 线性代数
- TBB: 并行计算

### Python库
- Open3D: 3D数据处理和可视化
- NumPy/SciPy: 数值计算
- PyMeshLab: 网格处理
- libigl: 几何处理Python绑定

## 快速开始

### 1. 环境配置
```bash
# 激活环境
eval "$(micromamba shell hook --shell bash)"
micromamba activate mesh-env
```

### 2. 编译项目
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 3. 运行管道
```bash
# C++版本
./pipeline input.ply output.obj

# Python版本
python scripts/pipeline.py input.ply output.obj
```

## 配置说明

### 全局配置 (global.yml)
- 项目基础设置
- 硬件和性能参数
- 输入输出格式
- 质量控制标准

### 外壳配置 (shell.yml)
- 预处理参数
- UDF构建设置
- 图割优化参数
- 合法化选项

### 细节配置 (detail.yml)
- 几何提取参数
- GP3/RIMLS重建设置
- 后处理选项
- 质量控制

### 融合配置 (fusion.yml)
- 融合策略
- 布尔运算参数
- LOD生成设置
- 导出格式

## 开发状态

- [x] 环境配置和依赖安装
- [x] 项目目录结构创建
- [x] 基础配置文件
- [ ] 核心C++代码实现
- [ ] Python包装器开发
- [ ] 构建系统配置
- [ ] 测试用例开发
- [ ] 文档完善

## 许可证

本项目基于技术文档实现，用于学习和研究目的。


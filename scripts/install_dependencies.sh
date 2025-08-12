#!/bin/bash

# 室内点云重建系统 - 依赖库安装脚本
# Indoor Point Cloud Reconstruction System - Dependencies Installation Script
# Author: Manus AI
# Date: 2025-08-12

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否为root用户
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "检测到root用户，建议使用普通用户运行此脚本"
        read -p "是否继续？(y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# 检测操作系统
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS="debian"
            log_info "检测到Debian/Ubuntu系统"
        elif [ -f /etc/redhat-release ]; then
            OS="redhat"
            log_info "检测到RedHat/CentOS/Fedora系统"
        else
            OS="linux"
            log_info "检测到其他Linux系统"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_info "检测到macOS系统"
    else
        log_error "不支持的操作系统: $OSTYPE"
        exit 1
    fi
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查包管理器
check_package_manager() {
    if [[ "$OS" == "debian" ]]; then
        if command_exists apt; then
            PKG_MANAGER="apt"
            PKG_UPDATE="sudo apt update"
            PKG_INSTALL="sudo apt install -y"
        else
            log_error "未找到apt包管理器"
            exit 1
        fi
    elif [[ "$OS" == "redhat" ]]; then
        if command_exists dnf; then
            PKG_MANAGER="dnf"
            PKG_UPDATE="sudo dnf update -y"
            PKG_INSTALL="sudo dnf install -y"
        elif command_exists yum; then
            PKG_MANAGER="yum"
            PKG_UPDATE="sudo yum update -y"
            PKG_INSTALL="sudo yum install -y"
        else
            log_error "未找到dnf或yum包管理器"
            exit 1
        fi
    elif [[ "$OS" == "macos" ]]; then
        if command_exists brew; then
            PKG_MANAGER="brew"
            PKG_UPDATE="brew update"
            PKG_INSTALL="brew install"
        else
            log_error "未找到Homebrew，请先安装: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
    fi
    
    log_info "使用包管理器: $PKG_MANAGER"
}

# 更新包管理器
update_package_manager() {
    log_info "更新包管理器..."
    eval $PKG_UPDATE
    log_success "包管理器更新完成"
}

# 安装基础依赖
install_basic_dependencies() {
    log_info "安装基础依赖..."
    
    if [[ "$OS" == "debian" ]]; then
        $PKG_INSTALL build-essential cmake git wget curl
        $PKG_INSTALL libboost-all-dev libeigen3-dev
        $PKG_INSTALL python3 python3-pip python3-dev
        $PKG_INSTALL pkg-config autoconf automake libtool
    elif [[ "$OS" == "redhat" ]]; then
        $PKG_INSTALL gcc gcc-c++ cmake git wget curl
        $PKG_INSTALL boost-devel eigen3-devel
        $PKG_INSTALL python3 python3-pip python3-devel
        $PKG_INSTALL pkgconfig autoconf automake libtool
    elif [[ "$OS" == "macos" ]]; then
        $PKG_INSTALL cmake git wget curl
        $PKG_INSTALL boost eigen
        $PKG_INSTALL python3
        $PKG_INSTALL autoconf automake libtool
    fi
    
    log_success "基础依赖安装完成"
}

# 安装Python依赖
install_python_dependencies() {
    log_info "安装Python依赖..."
    
    # 升级pip
    python3 -m pip install --upgrade pip
    
    # 安装核心Python包
    python3 -m pip install numpy scipy matplotlib
    python3 -m pip install PyMaxflow
    python3 -m pip install pyyaml
    python3 -m pip install opencv-python
    
    log_success "Python依赖安装完成"
}

# 创建构建目录
create_build_dirs() {
    log_info "创建构建目录..."
    
    BUILD_DIR="$HOME/mesh_recon_deps"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    log_info "构建目录: $BUILD_DIR"
}

# 安装CGAL
install_cgal() {
    log_info "安装CGAL..."
    
    if [[ "$OS" == "debian" ]]; then
        $PKG_INSTALL libcgal-dev libcgal-qt5-dev
    elif [[ "$OS" == "redhat" ]]; then
        $PKG_INSTALL CGAL-devel
    elif [[ "$OS" == "macos" ]]; then
        $PKG_INSTALL cgal
    else
        # 从源码编译
        log_info "从源码编译CGAL..."
        
        if [ ! -d "CGAL-5.6" ]; then
            wget https://github.com/CGAL/cgal/releases/download/v5.6/CGAL-5.6.tar.xz
            tar -xf CGAL-5.6.tar.xz
        fi
        
        cd CGAL-5.6
        mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
        make -j$(nproc)
        sudo make install
        cd ../..
    fi
    
    log_success "CGAL安装完成"
}

# 安装OpenVDB
install_openvdb() {
    log_info "安装OpenVDB..."
    
    if [[ "$OS" == "debian" ]]; then
        # Ubuntu 20.04+ 有预编译包
        if $PKG_INSTALL libopenvdb-dev 2>/dev/null; then
            log_success "从包管理器安装OpenVDB"
            return
        fi
    elif [[ "$OS" == "macos" ]]; then
        if $PKG_INSTALL openvdb 2>/dev/null; then
            log_success "从Homebrew安装OpenVDB"
            return
        fi
    fi
    
    # 从源码编译
    log_info "从源码编译OpenVDB..."
    
    # 安装依赖
    if [[ "$OS" == "debian" ]]; then
        $PKG_INSTALL libtbb-dev libhalf-dev libilmbase-dev libopenexr-dev
        $PKG_INSTALL libboost-iostreams-dev libboost-system-dev
    elif [[ "$OS" == "redhat" ]]; then
        $PKG_INSTALL tbb-devel OpenEXR-devel
    elif [[ "$OS" == "macos" ]]; then
        $PKG_INSTALL tbb openexr
    fi
    
    if [ ! -d "openvdb-10.0.1" ]; then
        wget https://github.com/AcademySoftwareFoundation/openvdb/archive/v10.0.1.tar.gz
        tar -xzf v10.0.1.tar.gz
    fi
    
    cd openvdb-10.0.1
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
             -DOPENVDB_BUILD_PYTHON_MODULE=OFF -DOPENVDB_BUILD_BINARIES=OFF
    make -j$(nproc)
    sudo make install
    cd ../..
    
    log_success "OpenVDB安装完成"
}

# 安装libigl
install_libigl() {
    log_info "安装libigl..."
    
    if [ ! -d "libigl-2.4.0" ]; then
        wget https://github.com/libigl/libigl/archive/v2.4.0.tar.gz
        tar -xzf v2.4.0.tar.gz
    fi
    
    cd libigl-2.4.0
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
             -DLIBIGL_WITH_CGAL=ON -DLIBIGL_WITH_OPENGL=OFF
    make -j$(nproc)
    sudo make install
    cd ../..
    
    log_success "libigl安装完成"
}

# 安装PCL (Point Cloud Library)
install_pcl() {
    log_info "安装PCL..."
    
    if [[ "$OS" == "debian" ]]; then
        $PKG_INSTALL libpcl-dev
    elif [[ "$OS" == "redhat" ]]; then
        $PKG_INSTALL pcl-devel
    elif [[ "$OS" == "macos" ]]; then
        $PKG_INSTALL pcl
    else
        # 从源码编译
        log_info "从源码编译PCL..."
        
        # 安装依赖
        if [[ "$OS" == "debian" ]]; then
            $PKG_INSTALL libflann-dev libvtk7-dev libqhull-dev
        fi
        
        if [ ! -d "pcl-1.13.1" ]; then
            wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.13.1.tar.gz
            tar -xzf pcl-1.13.1.tar.gz
        fi
        
        cd pcl-pcl-1.13.1
        mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
                 -DBUILD_GPU=OFF -DBUILD_apps=OFF -DBUILD_examples=OFF
        make -j$(nproc)
        sudo make install
        cd ../..
    fi
    
    log_success "PCL安装完成"
}

# 安装nlohmann/json
install_nlohmann_json() {
    log_info "安装nlohmann/json..."
    
    if [[ "$OS" == "debian" ]]; then
        $PKG_INSTALL nlohmann-json3-dev
    elif [[ "$OS" == "redhat" ]]; then
        $PKG_INSTALL json-devel
    elif [[ "$OS" == "macos" ]]; then
        $PKG_INSTALL nlohmann-json
    else
        # 从源码安装
        if [ ! -d "json-3.11.2" ]; then
            wget https://github.com/nlohmann/json/archive/v3.11.2.tar.gz
            tar -xzf v3.11.2.tar.gz
        fi
        
        cd json-3.11.2
        mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
                 -DJSON_BuildTests=OFF
        make -j$(nproc)
        sudo make install
        cd ../..
    fi
    
    log_success "nlohmann/json安装完成"
}

# 安装yaml-cpp
install_yaml_cpp() {
    log_info "安装yaml-cpp..."
    
    if [[ "$OS" == "debian" ]]; then
        $PKG_INSTALL libyaml-cpp-dev
    elif [[ "$OS" == "redhat" ]]; then
        $PKG_INSTALL yaml-cpp-devel
    elif [[ "$OS" == "macos" ]]; then
        $PKG_INSTALL yaml-cpp
    else
        # 从源码安装
        if [ ! -d "yaml-cpp-0.7.0" ]; then
            wget https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.7.0.tar.gz
            tar -xzf yaml-cpp-0.7.0.tar.gz
        fi
        
        cd yaml-cpp-yaml-cpp-0.7.0
        mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
                 -DYAML_BUILD_SHARED_LIBS=ON
        make -j$(nproc)
        sudo make install
        cd ../..
    fi
    
    log_success "yaml-cpp安装完成"
}

# 验证安装
verify_installation() {
    log_info "验证安装..."
    
    # 检查头文件
    HEADERS=(
        "/usr/local/include/CGAL/CGAL_config.h"
        "/usr/local/include/openvdb/openvdb.h"
        "/usr/local/include/igl/igl_inline.h"
        "/usr/local/include/nlohmann/json.hpp"
        "/usr/local/include/yaml-cpp/yaml.h"
    )
    
    for header in "${HEADERS[@]}"; do
        if [ -f "$header" ]; then
            log_success "找到头文件: $header"
        else
            log_warning "未找到头文件: $header"
        fi
    done
    
    # 检查库文件
    LIBS=(
        "CGAL"
        "openvdb"
        "yaml-cpp"
    )
    
    for lib in "${LIBS[@]}"; do
        if ldconfig -p | grep -q "$lib" 2>/dev/null || \
           find /usr/local/lib /usr/lib -name "lib${lib}*" 2>/dev/null | grep -q .; then
            log_success "找到库文件: $lib"
        else
            log_warning "未找到库文件: $lib"
        fi
    done
    
    # 检查Python包
    PYTHON_PACKAGES=(
        "numpy"
        "scipy"
        "maxflow"
        "yaml"
        "cv2"
    )
    
    for package in "${PYTHON_PACKAGES[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            log_success "找到Python包: $package"
        else
            log_warning "未找到Python包: $package"
        fi
    done
}

# 生成配置文件
generate_config() {
    log_info "生成配置文件..."
    
    CONFIG_FILE="$HOME/.mesh_recon_config"
    cat > "$CONFIG_FILE" << EOF
# 室内点云重建系统配置文件
# 由install_dependencies.sh自动生成

export CGAL_DIR=/usr/local
export OpenVDB_DIR=/usr/local
export libigl_DIR=/usr/local
export nlohmann_json_DIR=/usr/local
export yaml-cpp_DIR=/usr/local

# 添加库路径
export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:\$PKG_CONFIG_PATH

# Python路径
export PYTHONPATH=/usr/local/lib/python3/site-packages:\$PYTHONPATH

# 编译器标志
export CXXFLAGS="-I/usr/local/include \$CXXFLAGS"
export LDFLAGS="-L/usr/local/lib \$LDFLAGS"
EOF
    
    log_success "配置文件生成: $CONFIG_FILE"
    log_info "请运行 'source $CONFIG_FILE' 或将其添加到 ~/.bashrc"
}

# 清理临时文件
cleanup() {
    log_info "清理临时文件..."
    cd "$HOME"
    if [ -d "$BUILD_DIR" ]; then
        read -p "是否删除构建目录 $BUILD_DIR？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$BUILD_DIR"
            log_success "构建目录已删除"
        fi
    fi
}

# 显示帮助信息
show_help() {
    cat << EOF
室内点云重建系统 - 依赖库安装脚本

用法: $0 [选项]

选项:
    -h, --help          显示此帮助信息
    -q, --quick         快速安装（跳过可选依赖）
    -f, --force         强制重新安装所有依赖
    --no-cleanup        不清理临时文件
    --verify-only       仅验证已安装的依赖

支持的系统:
    - Ubuntu/Debian (apt)
    - CentOS/RHEL/Fedora (yum/dnf)
    - macOS (Homebrew)

安装的依赖库:
    - CGAL (计算几何算法库)
    - OpenVDB (体素数据结构)
    - libigl (几何处理库)
    - PCL (点云库)
    - nlohmann/json (JSON解析)
    - yaml-cpp (YAML解析)
    - PyMaxflow (图割算法)

EOF
}

# 主函数
main() {
    local quick_install=false
    local force_install=false
    local no_cleanup=false
    local verify_only=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -q|--quick)
                quick_install=true
                shift
                ;;
            -f|--force)
                force_install=true
                shift
                ;;
            --no-cleanup)
                no_cleanup=true
                shift
                ;;
            --verify-only)
                verify_only=true
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    log_info "开始安装室内点云重建系统依赖库..."
    
    if [ "$verify_only" = true ]; then
        verify_installation
        exit 0
    fi
    
    check_root
    detect_os
    check_package_manager
    
    if [ "$force_install" = true ]; then
        log_info "强制重新安装模式"
    fi
    
    update_package_manager
    install_basic_dependencies
    install_python_dependencies
    
    create_build_dirs
    
    install_nlohmann_json
    install_yaml_cpp
    install_cgal
    
    if [ "$quick_install" = false ]; then
        install_openvdb
        install_libigl
        install_pcl
    else
        log_info "快速安装模式，跳过可选依赖"
    fi
    
    verify_installation
    generate_config
    
    if [ "$no_cleanup" = false ]; then
        cleanup
    fi
    
    log_success "依赖库安装完成！"
    log_info "请运行以下命令设置环境变量:"
    log_info "source $HOME/.mesh_recon_config"
    log_info ""
    log_info "然后可以编译项目:"
    log_info "mkdir build && cd build"
    log_info "cmake .. && make -j\$(nproc)"
}

# 错误处理
trap 'log_error "安装过程中发生错误，退出码: $?"' ERR

# 运行主函数
main "$@"


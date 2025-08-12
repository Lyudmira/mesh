#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <map>
#include <vector>
#include <mutex>
#include <memory>

/**
 * 增强的日志系统，支持多级别日志、性能监控、数据验证
 */
class EnhancedLogger {
public:
    enum class Level {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3,
        CRITICAL = 4
    };
    
    enum class Category {
        GENERAL,
        PERFORMANCE,
        QUALITY,
        MEMORY,
        ALGORITHM,
        VALIDATION
    };
    
private:
    static std::unique_ptr<EnhancedLogger> instance_;
    static std::mutex mutex_;
    
    Level min_level_;
    std::ofstream log_file_;
    std::map<Category, std::ofstream> category_files_;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> timers_;
    std::map<std::string, std::vector<double>> metrics_;
    
    EnhancedLogger(const std::string& log_dir = "logs/") {
        min_level_ = Level::INFO;
        
        // 创建日志目录
        system(("mkdir -p " + log_dir).c_str());
        
        // 打开主日志文件
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << log_dir << "reconstruction_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".log";
        log_file_.open(ss.str());
        
        // 打开分类日志文件
        category_files_[Category::PERFORMANCE].open(log_dir + "performance.log");
        category_files_[Category::QUALITY].open(log_dir + "quality.log");
        category_files_[Category::MEMORY].open(log_dir + "memory.log");
        category_files_[Category::ALGORITHM].open(log_dir + "algorithm.log");
        category_files_[Category::VALIDATION].open(log_dir + "validation.log");
    }
    
public:
    static EnhancedLogger& getInstance() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (instance_ == nullptr) {
            instance_ = std::unique_ptr<EnhancedLogger>(new EnhancedLogger());
        }
        return *instance_;
    }
    
    void setMinLevel(Level level) {
        min_level_ = level;
    }
    
    // 基础日志记录
    void log(Level level, Category category, const std::string& message, 
             const std::map<std::string, double>& data = {}) {
        if (level < min_level_) return;
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
        ss << "[" << levelToString(level) << "] ";
        ss << "[" << categoryToString(category) << "] ";
        ss << message;
        
        // 添加数据
        if (!data.empty()) {
            ss << " | Data: ";
            for (const auto& [key, value] : data) {
                ss << key << "=" << value << " ";
            }
        }
        
        std::string log_line = ss.str();
        
        // 写入主日志
        log_file_ << log_line << std::endl;
        log_file_.flush();
        
        // 写入分类日志
        if (category_files_.find(category) != category_files_.end()) {
            category_files_[category] << log_line << std::endl;
            category_files_[category].flush();
        }
        
        // 控制台输出（ERROR及以上级别）
        if (level >= Level::ERROR) {
            std::cerr << log_line << std::endl;
        } else if (level >= Level::INFO) {
            std::cout << log_line << std::endl;
        }
    }
    
    // 性能监控
    void startTimer(const std::string& name) {
        timers_[name] = std::chrono::high_resolution_clock::now();
    }
    
    double endTimer(const std::string& name) {
        auto it = timers_.find(name);
        if (it == timers_.end()) {
            log(Level::WARNING, Category::PERFORMANCE, "Timer not found: " + name);
            return -1.0;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - it->second).count();
        double duration_ms = duration / 1000.0;
        
        log(Level::INFO, Category::PERFORMANCE, "Timer " + name + " completed", 
            {{"duration_ms", duration_ms}});
        
        timers_.erase(it);
        return duration_ms;
    }
    
    // 指标记录
    void recordMetric(const std::string& name, double value) {
        metrics_[name].push_back(value);
        log(Level::DEBUG, Category::PERFORMANCE, "Metric recorded: " + name, 
            {{"value", value}});
    }
    
    // 指标统计
    void reportMetrics() {
        log(Level::INFO, Category::PERFORMANCE, "=== PERFORMANCE METRICS REPORT ===");
        
        for (const auto& [name, values] : metrics_) {
            if (values.empty()) continue;
            
            double sum = 0.0, min_val = values[0], max_val = values[0];
            for (double val : values) {
                sum += val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
            double avg = sum / values.size();
            
            log(Level::INFO, Category::PERFORMANCE, "Metric: " + name, {
                {"count", static_cast<double>(values.size())},
                {"average", avg},
                {"minimum", min_val},
                {"maximum", max_val},
                {"total", sum}
            });
        }
    }
    
    // 数据验证日志
    void validateData(const std::string& data_name, double value, 
                     double expected_min, double expected_max, 
                     const std::string& unit = "") {
        bool valid = (value >= expected_min && value <= expected_max);
        Level level = valid ? Level::INFO : Level::WARNING;
        
        std::string message = "Data validation: " + data_name + 
                             (valid ? " PASSED" : " FAILED");
        
        log(level, Category::VALIDATION, message, {
            {"value", value},
            {"expected_min", expected_min},
            {"expected_max", expected_max},
            {"valid", valid ? 1.0 : 0.0}
        });
        
        if (!unit.empty()) {
            log(Level::DEBUG, Category::VALIDATION, "Unit: " + unit);
        }
    }
    
    // 内存使用监控
    void logMemoryUsage(const std::string& stage) {
        // 简化的内存使用获取（实际实现需要平台特定代码）
        size_t memory_kb = getCurrentMemoryUsage();
        double memory_mb = memory_kb / 1024.0;
        double memory_gb = memory_mb / 1024.0;
        
        log(Level::INFO, Category::MEMORY, "Memory usage at " + stage, {
            {"memory_kb", static_cast<double>(memory_kb)},
            {"memory_mb", memory_mb},
            {"memory_gb", memory_gb}
        });
        
        // 内存警告
        if (memory_gb > 32.0) {
            log(Level::WARNING, Category::MEMORY, "High memory usage detected", 
                {{"memory_gb", memory_gb}});
        }
    }
    
    // 算法质量验证
    void validateAlgorithmOutput(const std::string& algorithm, 
                                const std::map<std::string, double>& quality_metrics) {
        log(Level::INFO, Category::QUALITY, "Algorithm quality validation: " + algorithm);
        
        for (const auto& [metric, value] : quality_metrics) {
            log(Level::INFO, Category::QUALITY, "Quality metric: " + metric, 
                {{"value", value}});
            
            // 基于指标名称的自动验证
            if (metric == "planarity_rms" && value > 0.01) {
                log(Level::WARNING, Category::QUALITY, 
                    "Planarity RMS exceeds threshold", {{"value", value}, {"threshold", 0.01}});
            } else if (metric == "hausdorff_distance" && value > 0.02) {
                log(Level::WARNING, Category::QUALITY, 
                    "Hausdorff distance exceeds threshold", {{"value", value}, {"threshold", 0.02}});
            } else if (metric.find("error") != std::string::npos && value > 0.05) {
                log(Level::WARNING, Category::QUALITY, 
                    "Error metric exceeds threshold", {{"metric", 0}, {"value", value}, {"threshold", 0.05}});
            }
        }
    }
    
    // 处理阶段日志
    void logStageStart(const std::string& stage, const std::map<std::string, double>& input_stats = {}) {
        log(Level::INFO, Category::GENERAL, "=== STAGE START: " + stage + " ===");
        startTimer(stage);
        
        if (!input_stats.empty()) {
            log(Level::INFO, Category::GENERAL, "Input statistics for " + stage, input_stats);
        }
        
        logMemoryUsage(stage + "_start");
    }
    
    void logStageEnd(const std::string& stage, const std::map<std::string, double>& output_stats = {}) {
        double duration = endTimer(stage);
        
        log(Level::INFO, Category::GENERAL, "=== STAGE END: " + stage + " ===", 
            {{"duration_ms", duration}});
        
        if (!output_stats.empty()) {
            log(Level::INFO, Category::GENERAL, "Output statistics for " + stage, output_stats);
        }
        
        logMemoryUsage(stage + "_end");
    }
    
    ~EnhancedLogger() {
        reportMetrics();
        log_file_.close();
        for (auto& [category, file] : category_files_) {
            file.close();
        }
    }
    
private:
    std::string levelToString(Level level) {
        switch (level) {
            case Level::DEBUG: return "DEBUG";
            case Level::INFO: return "INFO";
            case Level::WARNING: return "WARN";
            case Level::ERROR: return "ERROR";
            case Level::CRITICAL: return "CRIT";
            default: return "UNKNOWN";
        }
    }
    
    std::string categoryToString(Category category) {
        switch (category) {
            case Category::GENERAL: return "GEN";
            case Category::PERFORMANCE: return "PERF";
            case Category::QUALITY: return "QUAL";
            case Category::MEMORY: return "MEM";
            case Category::ALGORITHM: return "ALGO";
            case Category::VALIDATION: return "VALID";
            default: return "UNK";
        }
    }
    
    size_t getCurrentMemoryUsage() {
        // 简化实现，实际需要平台特定代码
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line);
                std::string label, value, unit;
                iss >> label >> value >> unit;
                return std::stoul(value);
            }
        }
        return 0;
    }
};

// 静态成员定义
std::unique_ptr<EnhancedLogger> EnhancedLogger::instance_ = nullptr;
std::mutex EnhancedLogger::mutex_;

// 便捷宏定义
#define LOG_DEBUG(category, message, ...) \
    EnhancedLogger::getInstance().log(EnhancedLogger::Level::DEBUG, category, message, ##__VA_ARGS__)

#define LOG_INFO(category, message, ...) \
    EnhancedLogger::getInstance().log(EnhancedLogger::Level::INFO, category, message, ##__VA_ARGS__)

#define LOG_WARNING(category, message, ...) \
    EnhancedLogger::getInstance().log(EnhancedLogger::Level::WARNING, category, message, ##__VA_ARGS__)

#define LOG_ERROR(category, message, ...) \
    EnhancedLogger::getInstance().log(EnhancedLogger::Level::ERROR, category, message, ##__VA_ARGS__)

#define LOG_CRITICAL(category, message, ...) \
    EnhancedLogger::getInstance().log(EnhancedLogger::Level::CRITICAL, category, message, ##__VA_ARGS__)

#define START_TIMER(name) EnhancedLogger::getInstance().startTimer(name)
#define END_TIMER(name) EnhancedLogger::getInstance().endTimer(name)
#define RECORD_METRIC(name, value) EnhancedLogger::getInstance().recordMetric(name, value)
#define VALIDATE_DATA(name, value, min, max, unit) \
    EnhancedLogger::getInstance().validateData(name, value, min, max, unit)

#define LOG_STAGE_START(stage, ...) \
    EnhancedLogger::getInstance().logStageStart(stage, ##__VA_ARGS__)
#define LOG_STAGE_END(stage, ...) \
    EnhancedLogger::getInstance().logStageEnd(stage, ##__VA_ARGS__)

// 使用示例
int main() {
    auto& logger = EnhancedLogger::getInstance();
    logger.setMinLevel(EnhancedLogger::Level::DEBUG);
    
    // 阶段日志示例
    LOG_STAGE_START("preprocessing", {{"input_points", 10000}, {"input_size_mb", 15.2}});
    
    // 性能监控示例
    START_TIMER("voxelization");
    // ... 执行体素化算法
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 模拟处理时间
    END_TIMER("voxelization");
    
    // 指标记录示例
    RECORD_METRIC("voxel_count", 125000);
    RECORD_METRIC("processing_speed_voxels_per_sec", 1250000);
    
    // 数据验证示例
    VALIDATE_DATA("voxel_size", 0.05, 0.01, 0.1, "meters");
    VALIDATE_DATA("point_density", 850.5, 100, 2000, "points/m²");
    
    // 质量验证示例
    logger.validateAlgorithmOutput("graph_cut", {
        {"energy_value", 12345.67},
        {"inside_voxels", 85000},
        {"free_voxels", 40000},
        {"convergence_iterations", 15}
    });
    
    LOG_STAGE_END("preprocessing", {{"output_points", 9500}, {"removed_outliers", 500}});
    
    // 算法日志示例
    LOG_INFO(EnhancedLogger::Category::ALGORITHM, "Graph cut optimization completed", {
        {"nodes", 125000},
        {"edges", 750000},
        {"solve_time_ms", 45.2},
        {"energy", 12345.67}
    });
    
    // 质量控制日志示例
    LOG_INFO(EnhancedLogger::Category::QUALITY, "Mesh quality assessment", {
        {"triangle_count", 250000},
        {"vertex_count", 125000},
        {"planarity_rms", 0.008},
        {"hausdorff_distance", 0.015},
        {"manifold_edges_ratio", 0.98}
    });
    
    return 0;
}


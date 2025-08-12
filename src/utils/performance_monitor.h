/**
 * @file performance_monitor.h
 * @brief 性能监控和指标收集系统
 * 
 * 提供了全面的性能监控功能，包括：
 * - 实时性能指标收集
 * - 内存使用监控
 * - 算法性能分析
 * - 瓶颈识别
 * - 性能报告生成
 * 
 * @version 2.0
 * @date 2025-08-12
 */

#ifndef RECON_UTILS_PERFORMANCE_MONITOR_H
#define RECON_UTILS_PERFORMANCE_MONITOR_H

#include "../base/types.h"
#include "logger.h"
#include <chrono>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <iomanip>

namespace recon {
namespace utils {

// 前向声明
class PerformanceMonitor;

/**
 * @brief 性能计数器类型
 */
enum class CounterType {
    TIMER,          // 时间计数器
    COUNTER,        // 数值计数器
    GAUGE,          // 瞬时值
    RATE,           // 速率计数器
    MEMORY          // 内存使用
};

/**
 * @brief 性能指标
 */
struct PerformanceMetric {
    std::string name;
    CounterType type;
    double value;
    std::string unit;
    std::chrono::system_clock::time_point timestamp;
    std::string category;
    std::map<std::string, std::string> tags;
    
    PerformanceMetric(const std::string& n, CounterType t, double v, 
                     const std::string& u = "", const std::string& cat = "")
        : name(n), type(t), value(v), unit(u), category(cat),
          timestamp(std::chrono::system_clock::now()) {}
    
    void addTag(const std::string& key, const std::string& value) {
        tags[key] = value;
    }
};

/**
 * @brief 性能计时器
 */
class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::string name_;
    std::string category_;
    bool stopped_;
    
public:
    explicit PerformanceTimer(const std::string& name, const std::string& category = "")
        : name_(name), category_(category), stopped_(false) {
        start_time_ = std::chrono::high_resolution_clock::now();
        
        LOG_DEBUG("PerformanceTimer", "Started timer: " + name_);
    }
    
    ~PerformanceTimer() {
        if (!stopped_) {
            stop();
        }
    }
    
    double stop() {
        if (stopped_) return 0.0;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end_time - start_time_).count();
        stopped_ = true;
        
        // 记录到性能监控器（延迟到类定义之后）
        recordToMonitor(name_, duration, category_);
        
        LOG_DEBUG("PerformanceTimer", "Stopped timer: " + name_ + 
                 ", duration: " + std::to_string(duration) + "s");
        
        return duration;
    }
    
    double elapsed() const {
        auto current_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(current_time - start_time_).count();
    }
    
private:
    void recordToMonitor(const std::string& name, double duration, const std::string& category);
};

/**
 * @brief 内存使用监控器
 */
class MemoryMonitor {
private:
    static size_t getCurrentMemoryUsage() {
        // 简化的内存使用获取（在实际项目中可以使用更精确的方法）
        std::ifstream status("/proc/self/status");
        std::string line;
        
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line);
                std::string label, value, unit;
                iss >> label >> value >> unit;
                
                size_t memory_kb = std::stoull(value);
                return memory_kb * 1024;  // 转换为字节
            }
        }
        
        return 0;
    }
    
public:
    static size_t getMemoryUsage() {
        return getCurrentMemoryUsage();
    }
    
    static void recordMemoryUsage(const std::string& context) {
        size_t memory_usage = getCurrentMemoryUsage();
        
        // 记录到性能监控器（延迟到类定义之后）
        recordMemoryToMonitor(memory_usage);
        
        LOG_DEBUG("MemoryMonitor", "Memory usage at " + context + ": " + 
                 std::to_string(memory_usage / 1024 / 1024) + " MB");
    }
    
private:
    static void recordMemoryToMonitor(size_t memory_usage);
};

/**
 * @brief 性能监控器主类
 */
class PerformanceMonitor {
private:
    static std::unique_ptr<PerformanceMonitor> instance_;
    static std::mutex instance_mutex_;
    
    std::vector<PerformanceMetric> metrics_;
    mutable std::mutex metrics_mutex_;
    
    // 实时监控
    std::atomic<bool> monitoring_enabled_{false};
    std::thread monitoring_thread_;
    std::condition_variable monitoring_cv_;
    std::mutex monitoring_mutex_;
    
    // 配置
    size_t max_metrics_count_ = 10000;
    std::chrono::seconds monitoring_interval_{1};
    
    PerformanceMonitor() = default;
    
public:
    static PerformanceMonitor& getInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = std::unique_ptr<PerformanceMonitor>(new PerformanceMonitor());
        }
        return *instance_;
    }
    
    ~PerformanceMonitor() {
        stopMonitoring();
    }
    
    void recordMetric(const PerformanceMetric& metric) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        metrics_.push_back(metric);
        
        // 限制指标数量，避免内存无限增长
        if (metrics_.size() > max_metrics_count_) {
            metrics_.erase(metrics_.begin(), metrics_.begin() + (metrics_.size() - max_metrics_count_));
        }
    }
    
    void recordCounter(const std::string& name, double value, 
                      const std::string& unit = "", const std::string& category = "") {
        recordMetric(PerformanceMetric(name, CounterType::COUNTER, value, unit, category));
    }
    
    void recordGauge(const std::string& name, double value, 
                    const std::string& unit = "", const std::string& category = "") {
        recordMetric(PerformanceMetric(name, CounterType::GAUGE, value, unit, category));
    }
    
    void recordRate(const std::string& name, double value, 
                   const std::string& unit = "", const std::string& category = "") {
        recordMetric(PerformanceMetric(name, CounterType::RATE, value, unit, category));
    }
    
    std::unique_ptr<PerformanceTimer> createTimer(const std::string& name, 
                                                 const std::string& category = "") {
        return std::make_unique<PerformanceTimer>(name, category);
    }
    
    void startMonitoring() {
        if (monitoring_enabled_) return;
        
        monitoring_enabled_ = true;
        monitoring_thread_ = std::thread(&PerformanceMonitor::monitoringLoop, this);
        
        LOG_INFO("PerformanceMonitor", "Started real-time monitoring");
    }
    
    void stopMonitoring() {
        if (!monitoring_enabled_) return;
        
        monitoring_enabled_ = false;
        monitoring_cv_.notify_all();
        
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
        
        LOG_INFO("PerformanceMonitor", "Stopped real-time monitoring");
    }
    
    void setMonitoringInterval(std::chrono::seconds interval) {
        monitoring_interval_ = interval;
    }
    
    std::vector<PerformanceMetric> getMetrics(const std::string& category = "") const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        if (category.empty()) {
            return metrics_;
        }
        
        std::vector<PerformanceMetric> filtered_metrics;
        std::copy_if(metrics_.begin(), metrics_.end(), std::back_inserter(filtered_metrics),
            [&category](const PerformanceMetric& metric) {
                return metric.category == category;
            });
        
        return filtered_metrics;
    }
    
    std::map<std::string, double> getAggregatedMetrics(const std::string& category = "") const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        std::map<std::string, std::vector<double>> metric_values;
        
        for (const auto& metric : metrics_) {
            if (category.empty() || metric.category == category) {
                metric_values[metric.name].push_back(metric.value);
            }
        }
        
        std::map<std::string, double> aggregated;
        for (const auto& [name, values] : metric_values) {
            if (!values.empty()) {
                double sum = std::accumulate(values.begin(), values.end(), 0.0);
                aggregated[name + "_sum"] = sum;
                aggregated[name + "_avg"] = sum / values.size();
                aggregated[name + "_count"] = static_cast<double>(values.size());
                
                if (values.size() > 1) {
                    auto minmax = std::minmax_element(values.begin(), values.end());
                    aggregated[name + "_min"] = *minmax.first;
                    aggregated[name + "_max"] = *minmax.second;
                }
            }
        }
        
        return aggregated;
    }
    
    void generateReport(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            LOG_ERROR("PerformanceMonitor", "Failed to open report file: " + filename);
            return;
        }
        
        auto metrics = getMetrics();
        auto aggregated = getAggregatedMetrics();
        
        // 写入报告头
        file << "Performance Report\n";
        file << "==================\n";
        file << "Generated at: " << getCurrentTimeString() << "\n";
        file << "Total metrics: " << metrics.size() << "\n\n";
        
        // 写入聚合统计
        file << "Aggregated Statistics:\n";
        file << "----------------------\n";
        for (const auto& [name, value] : aggregated) {
            file << std::setw(30) << std::left << name << ": " 
                 << std::setw(15) << std::right << std::fixed << std::setprecision(6) << value << "\n";
        }
        file << "\n";
        
        // 按类别分组
        std::map<std::string, std::vector<PerformanceMetric>> metrics_by_category;
        for (const auto& metric : metrics) {
            metrics_by_category[metric.category.empty() ? "General" : metric.category].push_back(metric);
        }
        
        // 写入详细指标
        file << "Detailed Metrics:\n";
        file << "-----------------\n";
        for (const auto& [category, cat_metrics] : metrics_by_category) {
            file << "\nCategory: " << category << "\n";
            file << std::string(category.length() + 10, '-') << "\n";
            
            for (const auto& metric : cat_metrics) {
                file << std::setw(25) << std::left << metric.name << ": "
                     << std::setw(15) << std::right << std::fixed << std::setprecision(6) << metric.value;
                
                if (!metric.unit.empty()) {
                    file << " " << metric.unit;
                }
                
                file << " (" << formatTimestamp(metric.timestamp) << ")\n";
            }
        }
        
        file.close();
        LOG_INFO("PerformanceMonitor", "Performance report generated: " + filename);
    }
    
    void generateJsonReport(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            LOG_ERROR("PerformanceMonitor", "Failed to open JSON report file: " + filename);
            return;
        }
        
        auto metrics = getMetrics();
        auto aggregated = getAggregatedMetrics();
        
        file << "{\n";
        file << "  \"report_info\": {\n";
        file << "    \"generated_at\": \"" << getCurrentTimeString() << "\",\n";
        file << "    \"total_metrics\": " << metrics.size() << "\n";
        file << "  },\n";
        
        // 聚合统计
        file << "  \"aggregated_statistics\": {\n";
        bool first = true;
        for (const auto& [name, value] : aggregated) {
            if (!first) file << ",\n";
            file << "    \"" << name << "\": " << value;
            first = false;
        }
        file << "\n  },\n";
        
        // 详细指标
        file << "  \"detailed_metrics\": [\n";
        first = true;
        for (const auto& metric : metrics) {
            if (!first) file << ",\n";
            
            file << "    {\n";
            file << "      \"name\": \"" << metric.name << "\",\n";
            file << "      \"type\": \"" << counterTypeToString(metric.type) << "\",\n";
            file << "      \"value\": " << metric.value << ",\n";
            file << "      \"unit\": \"" << metric.unit << "\",\n";
            file << "      \"category\": \"" << metric.category << "\",\n";
            file << "      \"timestamp\": \"" << formatTimestamp(metric.timestamp) << "\"";
            
            if (!metric.tags.empty()) {
                file << ",\n      \"tags\": {\n";
                bool first_tag = true;
                for (const auto& [key, value] : metric.tags) {
                    if (!first_tag) file << ",\n";
                    file << "        \"" << key << "\": \"" << value << "\"";
                    first_tag = false;
                }
                file << "\n      }";
            }
            
            file << "\n    }";
            first = false;
        }
        file << "\n  ]\n";
        file << "}\n";
        
        file.close();
        LOG_INFO("PerformanceMonitor", "JSON performance report generated: " + filename);
    }
    
    void clearMetrics() {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        metrics_.clear();
        LOG_INFO("PerformanceMonitor", "Cleared all performance metrics");
    }
    
private:
    void monitoringLoop() {
        while (monitoring_enabled_) {
            // 记录系统指标
            MemoryMonitor::recordMemoryUsage("monitoring_loop");
            
            // 记录CPU使用率（简化版本）
            recordGauge("cpu_usage", getCpuUsage(), "percent", "system");
            
            // 等待下一个监控周期
            std::unique_lock<std::mutex> lock(monitoring_mutex_);
            monitoring_cv_.wait_for(lock, monitoring_interval_, 
                [this] { return !monitoring_enabled_; });
        }
    }
    
    double getCpuUsage() const {
        // 简化的CPU使用率获取（在实际项目中可以使用更精确的方法）
        static auto last_time = std::chrono::high_resolution_clock::now();
        static double last_cpu_time = 0.0;
        
        auto current_time = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(current_time - last_time).count();
        
        if (elapsed < 0.1) return last_cpu_time;  // 避免频繁计算
        
        std::ifstream stat("/proc/self/stat");
        if (!stat.is_open()) return 0.0;
        
        std::string line;
        std::getline(stat, line);
        std::istringstream iss(line);
        
        // 跳过前13个字段，获取utime和stime
        std::string field;
        for (int i = 0; i < 13; ++i) {
            iss >> field;
        }
        
        long utime, stime;
        iss >> utime >> stime;
        
        double cpu_time = (utime + stime) / 100.0;  // 转换为秒
        double cpu_usage = (cpu_time - last_cpu_time) / elapsed * 100.0;
        
        last_time = current_time;
        last_cpu_time = cpu_time;
        
        return std::max(0.0, std::min(100.0, cpu_usage));
    }
    
    std::string getCurrentTimeString() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }
    
    std::string formatTimestamp(const std::chrono::system_clock::time_point& tp) const {
        auto time_t = std::chrono::system_clock::to_time_t(tp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            tp.time_since_epoch()) % 1000;
        
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }
    
    std::string counterTypeToString(CounterType type) const {
        switch (type) {
            case CounterType::TIMER: return "timer";
            case CounterType::COUNTER: return "counter";
            case CounterType::GAUGE: return "gauge";
            case CounterType::RATE: return "rate";
            case CounterType::MEMORY: return "memory";
            default: return "unknown";
        }
    }
};

// 静态成员定义
std::unique_ptr<PerformanceMonitor> PerformanceMonitor::instance_ = nullptr;
std::mutex PerformanceMonitor::instance_mutex_;

// PerformanceTimer的延迟实现
inline void PerformanceTimer::recordToMonitor(const std::string& name, double duration, const std::string& category) {
    PerformanceMonitor::getInstance().recordMetric(
        PerformanceMetric(name, CounterType::TIMER, duration, "seconds", category));
}

// MemoryMonitor的延迟实现
inline void MemoryMonitor::recordMemoryToMonitor(size_t memory_usage) {
    PerformanceMonitor::getInstance().recordMetric(
        PerformanceMetric("memory_usage", CounterType::MEMORY, 
                        static_cast<double>(memory_usage), "bytes", "memory"));
}

// 便利宏定义
#define PERF_TIMER(name) \
    auto __perf_timer_##__LINE__ = recon::utils::PerformanceMonitor::getInstance().createTimer(name)

#define PERF_TIMER_CATEGORY(name, category) \
    auto __perf_timer_##__LINE__ = recon::utils::PerformanceMonitor::getInstance().createTimer(name, category)

#define PERF_COUNTER(name, value) \
    recon::utils::PerformanceMonitor::getInstance().recordCounter(name, value)

#define PERF_GAUGE(name, value) \
    recon::utils::PerformanceMonitor::getInstance().recordGauge(name, value)

#define PERF_RATE(name, value) \
    recon::utils::PerformanceMonitor::getInstance().recordRate(name, value)

#define PERF_MEMORY(context) \
    recon::utils::MemoryMonitor::recordMemoryUsage(context)

} // namespace utils
} // namespace recon

#endif // RECON_UTILS_PERFORMANCE_MONITOR_H


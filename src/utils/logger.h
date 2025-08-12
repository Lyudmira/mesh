/**
 * @file logger.h
 * @brief 增强的分层日志系统
 * 
 * 提供了完整的日志记录功能，支持：
 * - 分层日志级别（DEBUG, INFO, WARN, ERROR, FATAL）
 * - 模块化日志流
 * - 结构化日志输出（JSON格式）
 * - 性能监控集成
 * - 线程安全操作
 * 
 * @version 2.0
 * @date 2025-08-12
 */

#ifndef RECON_UTILS_LOGGER_H
#define RECON_UTILS_LOGGER_H

#include "../base/types.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <memory>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <iomanip>

namespace recon {
namespace utils {

/**
 * @brief 日志级别枚举
 */
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3,
    FATAL = 4
};

/**
 * @brief 日志条目结构
 */
struct LogEntry {
    LogLevel level;
    std::string module;
    std::string message;
    std::string function;
    std::string file;
    int line;
    std::chrono::system_clock::time_point timestamp;
    std::thread::id thread_id;
    
    // 可选的结构化数据
    std::map<std::string, std::string> metadata;
    
    LogEntry(LogLevel lvl, const std::string& mod, const std::string& msg,
             const std::string& func = "", const std::string& f = "", int l = 0)
        : level(lvl), module(mod), message(msg), function(func), file(f), line(l),
          timestamp(std::chrono::system_clock::now()), thread_id(std::this_thread::get_id()) {}
    
    void addMetadata(const std::string& key, const std::string& value) {
        metadata[key] = value;
    }
    
    void addMetadata(const std::string& key, double value) {
        metadata[key] = std::to_string(value);
    }
    
    void addMetadata(const std::string& key, int value) {
        metadata[key] = std::to_string(value);
    }
};

/**
 * @brief 日志格式化器接口
 */
class ILogFormatter {
public:
    virtual ~ILogFormatter() = default;
    virtual std::string format(const LogEntry& entry) = 0;
};

/**
 * @brief 文本格式化器
 */
class TextFormatter : public ILogFormatter {
public:
    std::string format(const LogEntry& entry) override {
        std::ostringstream oss;
        
        // 时间戳
        auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            entry.timestamp.time_since_epoch()) % 1000;
        
        oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count();
        
        // 日志级别
        oss << " [" << levelToString(entry.level) << "]";
        
        // 模块名
        if (!entry.module.empty()) {
            oss << " [" << entry.module << "]";
        }
        
        // 线程ID
        oss << " [T:" << entry.thread_id << "]";
        
        // 消息
        oss << " " << entry.message;
        
        // 位置信息（DEBUG级别显示）
        if (entry.level == LogLevel::DEBUG && !entry.function.empty()) {
            oss << " (" << entry.function;
            if (!entry.file.empty()) {
                oss << " at " << entry.file << ":" << entry.line;
            }
            oss << ")";
        }
        
        // 元数据
        if (!entry.metadata.empty()) {
            oss << " {";
            bool first = true;
            for (const auto& [key, value] : entry.metadata) {
                if (!first) oss << ", ";
                oss << key << "=" << value;
                first = false;
            }
            oss << "}";
        }
        
        return oss.str();
    }
    
private:
    std::string levelToString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO ";
            case LogLevel::WARN:  return "WARN ";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::FATAL: return "FATAL";
            default: return "UNKNOWN";
        }
    }
};

/**
 * @brief JSON格式化器
 */
class JsonFormatter : public ILogFormatter {
public:
    std::string format(const LogEntry& entry) override {
        std::ostringstream oss;
        
        oss << "{";
        
        // 时间戳（ISO 8601格式）
        auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            entry.timestamp.time_since_epoch()) % 1000;
        
        oss << "\"timestamp\":\"";
        oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
        oss << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z\"";
        
        // 基本字段
        oss << ",\"level\":\"" << levelToString(entry.level) << "\"";
        oss << ",\"module\":\"" << escapeJson(entry.module) << "\"";
        oss << ",\"message\":\"" << escapeJson(entry.message) << "\"";
        oss << ",\"thread_id\":\"" << entry.thread_id << "\"";
        
        // 位置信息
        if (!entry.function.empty()) {
            oss << ",\"function\":\"" << escapeJson(entry.function) << "\"";
        }
        if (!entry.file.empty()) {
            oss << ",\"file\":\"" << escapeJson(entry.file) << "\"";
            oss << ",\"line\":" << entry.line;
        }
        
        // 元数据
        if (!entry.metadata.empty()) {
            oss << ",\"metadata\":{";
            bool first = true;
            for (const auto& [key, value] : entry.metadata) {
                if (!first) oss << ",";
                oss << "\"" << escapeJson(key) << "\":\"" << escapeJson(value) << "\"";
                first = false;
            }
            oss << "}";
        }
        
        oss << "}";
        return oss.str();
    }
    
private:
    std::string levelToString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO";
            case LogLevel::WARN:  return "WARN";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::FATAL: return "FATAL";
            default: return "UNKNOWN";
        }
    }
    
    std::string escapeJson(const std::string& str) {
        std::string result;
        for (char c : str) {
            switch (c) {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default: result += c; break;
            }
        }
        return result;
    }
};

/**
 * @brief 日志输出器接口
 */
class ILogAppender {
public:
    virtual ~ILogAppender() = default;
    virtual void append(const LogEntry& entry) = 0;
    virtual void flush() = 0;
};

/**
 * @brief 控制台输出器
 */
class ConsoleAppender : public ILogAppender {
private:
    std::unique_ptr<ILogFormatter> formatter_;
    LogLevel min_level_;
    mutable std::mutex mutex_;
    
public:
    explicit ConsoleAppender(std::unique_ptr<ILogFormatter> formatter, LogLevel min_level = LogLevel::INFO)
        : formatter_(std::move(formatter)), min_level_(min_level) {}
    
    void append(const LogEntry& entry) override {
        if (entry.level < min_level_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::string formatted = formatter_->format(entry);
        
        if (entry.level >= LogLevel::ERROR) {
            std::cerr << formatted << std::endl;
        } else {
            std::cout << formatted << std::endl;
        }
    }
    
    void flush() override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout.flush();
        std::cerr.flush();
    }
};

/**
 * @brief 文件输出器
 */
class FileAppender : public ILogAppender {
private:
    std::unique_ptr<ILogFormatter> formatter_;
    std::ofstream file_;
    LogLevel min_level_;
    mutable std::mutex mutex_;
    size_t max_file_size_;
    size_t current_size_;
    std::string base_filename_;
    int file_index_;
    
public:
    explicit FileAppender(const std::string& filename, 
                         std::unique_ptr<ILogFormatter> formatter,
                         LogLevel min_level = LogLevel::DEBUG,
                         size_t max_file_size = 100 * 1024 * 1024)  // 100MB
        : formatter_(std::move(formatter)), min_level_(min_level),
          max_file_size_(max_file_size), current_size_(0),
          base_filename_(filename), file_index_(0) {
        
        openFile();
    }
    
    ~FileAppender() {
        if (file_.is_open()) {
            file_.close();
        }
    }
    
    void append(const LogEntry& entry) override {
        if (entry.level < min_level_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 检查文件大小，必要时轮转
        if (current_size_ > max_file_size_) {
            rotateFile();
        }
        
        std::string formatted = formatter_->format(entry);
        file_ << formatted << std::endl;
        current_size_ += formatted.length() + 1;
    }
    
    void flush() override {
        std::lock_guard<std::mutex> lock(mutex_);
        file_.flush();
    }
    
private:
    void openFile() {
        std::string filename = base_filename_;
        if (file_index_ > 0) {
            size_t dot_pos = base_filename_.find_last_of('.');
            if (dot_pos != std::string::npos) {
                filename = base_filename_.substr(0, dot_pos) + 
                          "_" + std::to_string(file_index_) + 
                          base_filename_.substr(dot_pos);
            } else {
                filename = base_filename_ + "_" + std::to_string(file_index_);
            }
        }
        
        file_.open(filename, std::ios::app);
        current_size_ = file_.tellp();
    }
    
    void rotateFile() {
        file_.close();
        file_index_++;
        openFile();
        current_size_ = 0;
    }
};

/**
 * @brief 异步日志器
 */
class AsyncLogger {
private:
    std::vector<std::unique_ptr<ILogAppender>> appenders_;
    std::queue<LogEntry> log_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread worker_thread_;
    std::atomic<bool> stop_flag_{false};
    LogLevel min_level_;
    
public:
    explicit AsyncLogger(LogLevel min_level = LogLevel::DEBUG)
        : min_level_(min_level), worker_thread_(&AsyncLogger::workerLoop, this) {}
    
    ~AsyncLogger() {
        stop();
    }
    
    void addAppender(std::unique_ptr<ILogAppender> appender) {
        appenders_.push_back(std::move(appender));
    }
    
    void log(const LogEntry& entry) {
        if (entry.level < min_level_) return;
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            log_queue_.push(entry);
        }
        queue_cv_.notify_one();
    }
    
    void flush() {
        // 等待队列清空
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return log_queue_.empty(); });
        
        // 刷新所有appender
        for (auto& appender : appenders_) {
            appender->flush();
        }
    }
    
    void stop() {
        stop_flag_ = true;
        queue_cv_.notify_all();
        
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        
        // 处理剩余的日志条目
        while (!log_queue_.empty()) {
            LogEntry entry = log_queue_.front();
            log_queue_.pop();
            
            for (auto& appender : appenders_) {
                appender->append(entry);
            }
        }
        
        // 最终刷新
        for (auto& appender : appenders_) {
            appender->flush();
        }
    }
    
private:
    void workerLoop() {
        while (!stop_flag_) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            queue_cv_.wait(lock, [this] { 
                return !log_queue_.empty() || stop_flag_; 
            });
            
            while (!log_queue_.empty()) {
                LogEntry entry = log_queue_.front();
                log_queue_.pop();
                lock.unlock();
                
                // 发送到所有appender
                for (auto& appender : appenders_) {
                    appender->append(entry);
                }
                
                lock.lock();
            }
        }
    }
};

/**
 * @brief 模块化日志管理器
 */
class LogManager {
private:
    static std::unique_ptr<LogManager> instance_;
    static std::mutex instance_mutex_;
    
    std::unique_ptr<AsyncLogger> logger_;
    std::map<std::string, LogLevel> module_levels_;
    mutable std::mutex config_mutex_;
    
    LogManager() {
        logger_ = std::make_unique<AsyncLogger>(LogLevel::DEBUG);
        
        // 默认添加控制台输出器
        logger_->addAppender(std::make_unique<ConsoleAppender>(
            std::make_unique<TextFormatter>(), LogLevel::INFO));
    }
    
public:
    static LogManager& getInstance() {
        std::lock_guard<std::mutex> lock(instance_mutex_);
        if (!instance_) {
            instance_ = std::unique_ptr<LogManager>(new LogManager());
        }
        return *instance_;
    }
    
    void addFileAppender(const std::string& filename, bool json_format = false, 
                        LogLevel min_level = LogLevel::DEBUG) {
        std::unique_ptr<ILogFormatter> formatter;
        if (json_format) {
            formatter = std::make_unique<JsonFormatter>();
        } else {
            formatter = std::make_unique<TextFormatter>();
        }
        
        logger_->addAppender(std::make_unique<FileAppender>(
            filename, std::move(formatter), min_level));
    }
    
    void setModuleLevel(const std::string& module, LogLevel level) {
        std::lock_guard<std::mutex> lock(config_mutex_);
        module_levels_[module] = level;
    }
    
    LogLevel getModuleLevel(const std::string& module) const {
        std::lock_guard<std::mutex> lock(config_mutex_);
        auto it = module_levels_.find(module);
        return (it != module_levels_.end()) ? it->second : LogLevel::DEBUG;
    }
    
    void log(LogLevel level, const std::string& module, const std::string& message,
             const std::string& function = "", const std::string& file = "", int line = 0) {
        
        // 检查模块级别
        if (level < getModuleLevel(module)) {
            return;
        }
        
        LogEntry entry(level, module, message, function, file, line);
        logger_->log(entry);
    }
    
    void logWithMetadata(LogLevel level, const std::string& module, const std::string& message,
                        const std::map<std::string, std::string>& metadata,
                        const std::string& function = "", const std::string& file = "", int line = 0) {
        
        if (level < getModuleLevel(module)) {
            return;
        }
        
        LogEntry entry(level, module, message, function, file, line);
        entry.metadata = metadata;
        logger_->log(entry);
    }
    
    void flush() {
        logger_->flush();
    }
    
    void shutdown() {
        logger_->stop();
    }
};

// 静态成员定义
std::unique_ptr<LogManager> LogManager::instance_ = nullptr;
std::mutex LogManager::instance_mutex_;

// 便利宏定义
#define LOG_DEBUG(module, message) \
    recon::utils::LogManager::getInstance().log(recon::utils::LogLevel::DEBUG, module, message, __FUNCTION__, __FILE__, __LINE__)

#define LOG_INFO(module, message) \
    recon::utils::LogManager::getInstance().log(recon::utils::LogLevel::INFO, module, message, __FUNCTION__, __FILE__, __LINE__)

#define LOG_WARN(module, message) \
    recon::utils::LogManager::getInstance().log(recon::utils::LogLevel::WARN, module, message, __FUNCTION__, __FILE__, __LINE__)

#define LOG_ERROR(module, message) \
    recon::utils::LogManager::getInstance().log(recon::utils::LogLevel::ERROR, module, message, __FUNCTION__, __FILE__, __LINE__)

#define LOG_FATAL(module, message) \
    recon::utils::LogManager::getInstance().log(recon::utils::LogLevel::FATAL, module, message, __FUNCTION__, __FILE__, __LINE__)

// 带元数据的日志宏
#define LOG_WITH_METADATA(level, module, message, metadata) \
    recon::utils::LogManager::getInstance().logWithMetadata(level, module, message, metadata, __FUNCTION__, __FILE__, __LINE__)

} // namespace utils
} // namespace recon

#endif // RECON_UTILS_LOGGER_H


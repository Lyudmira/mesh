/**
 * @file config_manager.h
 * @brief 配置管理系统 - 统一的配置加载和管理
 * @author Manus AI
 * @date 2025-08-12
 */

#ifndef RECON_CONFIG_MANAGER_H
#define RECON_CONFIG_MANAGER_H

#include "../base/types.h"
#include <yaml-cpp/yaml.h>
#include <string>
#include <map>
#include <memory>
#include <mutex>

namespace recon {
namespace config {

/**
 * @brief 配置节点类型
 */
enum class ConfigType {
    STRING,
    INTEGER,
    FLOAT,
    BOOLEAN,
    ARRAY,
    OBJECT
};

/**
 * @brief 配置值包装器
 */
class ConfigValue {
public:
    ConfigValue() = default;
    explicit ConfigValue(const YAML::Node& node);
    
    // 类型检查
    bool isString() const;
    bool isInt() const;
    bool isFloat() const;
    bool isBool() const;
    bool isArray() const;
    bool isObject() const;
    
    // 值获取
    std::string asString(const std::string& default_value = "") const;
    int asInt(int default_value = 0) const;
    float asFloat(float default_value = 0.0f) const;
    double asDouble(double default_value = 0.0) const;
    bool asBool(bool default_value = false) const;
    
    // 数组操作
    size_t size() const;
    ConfigValue operator[](size_t index) const;
    ConfigValue operator[](const std::string& key) const;
    
    // 存在性检查
    bool exists() const;
    bool hasKey(const std::string& key) const;
    
    // 迭代器支持
    std::vector<std::string> getKeys() const;
    std::vector<ConfigValue> getValues() const;
    
private:
    YAML::Node node_;
    bool valid_ = false;
};

/**
 * @brief 配置管理器 - 单例模式
 */
class ConfigManager {
public:
    // 单例访问
    static ConfigManager& getInstance();
    
    // 配置加载
    bool loadFromFile(const std::string& config_path);
    bool loadFromString(const std::string& yaml_content);
    bool loadDefaults();
    
    // 配置访问
    ConfigValue get(const std::string& path) const;
    std::string getString(const std::string& path, const std::string& default_value = "") const;
    int getInt(const std::string& path, int default_value = 0) const;
    float getFloat(const std::string& path, float default_value = 0.0f) const;
    double getDouble(const std::string& path, double default_value = 0.0) const;
    bool getBool(const std::string& path, bool default_value = false) const;
    
    // 配置设置
    void set(const std::string& path, const std::string& value);
    void set(const std::string& path, int value);
    void set(const std::string& path, float value);
    void set(const std::string& path, double value);
    void set(const std::string& path, bool value);
    
    // 配置保存
    bool saveToFile(const std::string& config_path) const;
    std::string saveToString() const;
    
    // 配置验证
    bool validate() const;
    std::vector<std::string> getValidationErrors() const;
    
    // 配置合并
    void merge(const ConfigManager& other);
    void mergeFromFile(const std::string& config_path);
    
    // 环境变量支持
    void loadEnvironmentOverrides();
    void setEnvironmentPrefix(const std::string& prefix);
    
    // 配置监听
    using ConfigChangeCallback = std::function<void(const std::string& path, const ConfigValue& old_value, const ConfigValue& new_value)>;
    void addChangeListener(const std::string& path, ConfigChangeCallback callback);
    void removeChangeListener(const std::string& path);
    
    // 配置信息
    bool isLoaded() const;
    std::string getConfigPath() const;
    std::chrono::system_clock::time_point getLoadTime() const;
    
    // 调试和诊断
    void printConfig() const;
    void printConfigTree() const;
    std::map<std::string, std::string> getAllSettings() const;
    
private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;
    
    // 内部方法
    ConfigValue getNode(const std::string& path) const;
    void setNode(const std::string& path, const YAML::Node& value);
    std::vector<std::string> splitPath(const std::string& path) const;
    YAML::Node navigateToNode(const std::vector<std::string>& path_parts, bool create = false);
    const YAML::Node navigateToNode(const std::vector<std::string>& path_parts) const;
    
    // 验证方法
    bool validateSystemConfig() const;
    bool validateAlgorithmConfig() const;
    bool validateQualityConfig() const;
    bool validateOutputConfig() const;
    
    // 环境变量处理
    std::string getEnvironmentVariable(const std::string& var_name) const;
    void applyEnvironmentOverride(const std::string& env_var, const std::string& config_path);
    
    // 通知方法
    void notifyConfigChange(const std::string& path, const ConfigValue& old_value, const ConfigValue& new_value);
    
    // 成员变量
    YAML::Node config_;
    mutable std::mutex config_mutex_;
    bool loaded_ = false;
    std::string config_path_;
    std::chrono::system_clock::time_point load_time_;
    std::string environment_prefix_ = "RECON_";
    
    // 监听器
    std::map<std::string, std::vector<ConfigChangeCallback>> change_listeners_;
    mutable std::mutex listeners_mutex_;
    
    // 验证错误
    mutable std::vector<std::string> validation_errors_;
    
    // 静态成员
    static std::unique_ptr<ConfigManager> instance_;
    static std::mutex instance_mutex_;
};

/**
 * @brief 配置作用域 - RAII风格的临时配置修改
 */
class ConfigScope {
public:
    explicit ConfigScope(const std::string& path, const ConfigValue& temporary_value);
    ~ConfigScope();
    
    ConfigScope(const ConfigScope&) = delete;
    ConfigScope& operator=(const ConfigScope&) = delete;
    ConfigScope(ConfigScope&& other) noexcept;
    ConfigScope& operator=(ConfigScope&& other) noexcept;
    
private:
    std::string path_;
    ConfigValue original_value_;
    bool active_ = true;
};

/**
 * @brief 配置验证器
 */
class ConfigValidator {
public:
    struct ValidationRule {
        std::string path;
        ConfigType expected_type;
        bool required = true;
        std::function<bool(const ConfigValue&)> custom_validator;
        std::string error_message;
    };
    
    static bool validate(const ConfigManager& config, const std::vector<ValidationRule>& rules);
    static std::vector<ValidationRule> getDefaultRules();
    static std::vector<ValidationRule> getSystemRules();
    static std::vector<ValidationRule> getAlgorithmRules();
    static std::vector<ValidationRule> getQualityRules();
    
private:
    static bool validateType(const ConfigValue& value, ConfigType expected_type);
    static bool validateRange(const ConfigValue& value, double min_val, double max_val);
    static bool validateEnum(const ConfigValue& value, const std::vector<std::string>& allowed_values);
};

/**
 * @brief 配置工厂 - 预定义配置的创建
 */
class ConfigFactory {
public:
    // 预定义配置
    static ConfigManager createDefaultConfig();
    static ConfigManager createHighQualityConfig();
    static ConfigManager createFastProcessingConfig();
    static ConfigManager createLowMemoryConfig();
    static ConfigManager createDebugConfig();
    
    // 场景特定配置
    static ConfigManager createIndoorSceneConfig();
    static ConfigManager createLargeScaleConfig();
    static ConfigManager createRealTimeConfig();
    
    // 硬件特定配置
    static ConfigManager createMultiCoreConfig(int num_cores);
    static ConfigManager createLowEndHardwareConfig();
    static ConfigManager createHighEndHardwareConfig();
    
private:
    static void applyCommonSettings(ConfigManager& config);
    static void applyPerformanceSettings(ConfigManager& config, const std::string& profile);
    static void applyQualitySettings(ConfigManager& config, const std::string& profile);
};

// 便利宏定义
#define CONFIG_GET(path, type, default_val) \
    recon::config::ConfigManager::getInstance().get##type(path, default_val)

#define CONFIG_SET(path, value) \
    recon::config::ConfigManager::getInstance().set(path, value)

#define CONFIG_STRING(path, default_val) CONFIG_GET(path, String, default_val)
#define CONFIG_INT(path, default_val) CONFIG_GET(path, Int, default_val)
#define CONFIG_FLOAT(path, default_val) CONFIG_GET(path, Float, default_val)
#define CONFIG_BOOL(path, default_val) CONFIG_GET(path, Bool, default_val)

// 配置路径常量
namespace ConfigPaths {
    // 系统配置路径
    constexpr const char* SYSTEM_LOG_LEVEL = "system.logging.level";
    constexpr const char* SYSTEM_NUM_THREADS = "system.parallel.num_threads";
    constexpr const char* SYSTEM_ENABLE_MONITORING = "system.performance.enable_monitoring";
    
    // 算法配置路径
    constexpr const char* GRAPH_CUT_SOLVER = "algorithms.graph_cut.solver_type";
    constexpr const char* UDF_RESOLUTION = "algorithms.udf_builder.grid_resolution";
    constexpr const char* DC_ENABLE_QEF = "algorithms.dual_contouring.enable_qef_solver";
    constexpr const char* DETAIL_METHOD = "algorithms.detail_reconstruction.method";
    constexpr const char* FUSION_METHOD = "algorithms.mesh_fusion.method";
    
    // 质量配置路径
    constexpr const char* QUALITY_THRESHOLD = "data.validation.quality_threshold";
    constexpr const char* MAX_HAUSDORFF = "quality.geometric_accuracy.max_hausdorff_distance";
    constexpr const char* MIN_FEATURE_PRESERVATION = "quality.geometric_accuracy.min_feature_preservation";
    
    // 输出配置路径
    constexpr const char* OUTPUT_FORMATS = "output.formats.mesh";
    constexpr const char* VERTEX_PRECISION = "output.quality.vertex_precision";
    constexpr const char* ENABLE_REPORTS = "output.reports.enable_quality_report";
}

} // namespace config
} // namespace recon

#endif // RECON_CONFIG_MANAGER_H


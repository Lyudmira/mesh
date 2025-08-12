#include <iostream>
#include <string>
#include <vector>

// 复制JSON解析器代码进行调试
class SimpleJsonParser {
private:
    std::string json_;
    
public:
    explicit SimpleJsonParser(const std::string& json) : json_(json) {
        std::cout << "JSON input: " << json_ << std::endl;
    }
    
    std::vector<int> getIntArray(const std::string& key) {
        std::cout << "Looking for key: " << key << std::endl;
        
        std::string pattern = "\"" + key + "\":[";
        std::cout << "Pattern: " << pattern << std::endl;
        
        size_t start = json_.find(pattern);
        if (start == std::string::npos) {
            std::cout << "Pattern not found!" << std::endl;
            return {};
        }
        
        std::cout << "Pattern found at position: " << start << std::endl;
        
        start += pattern.length();
        size_t end = json_.find("]", start);
        if (end == std::string::npos) {
            std::cout << "Closing bracket not found!" << std::endl;
            return {};
        }
        
        std::cout << "Array end at position: " << end << std::endl;
        
        std::string array_content = json_.substr(start, end - start);
        std::cout << "Array content: '" << array_content << "'" << std::endl;
        
        std::vector<int> result;
        
        // 简单的数组解析
        std::string current_number;
        for (char c : array_content) {
            std::cout << "Processing char: '" << c << "'" << std::endl;
            if (std::isdigit(c) || c == '-') {
                current_number += c;
                std::cout << "  Current number: '" << current_number << "'" << std::endl;
            } else if (c == ',' || c == ' ') {
                if (!current_number.empty()) {
                    int num = std::stoi(current_number);
                    std::cout << "  Adding number: " << num << std::endl;
                    result.push_back(num);
                    current_number.clear();
                }
            }
        }
        // 处理最后一个数字
        if (!current_number.empty()) {
            int num = std::stoi(current_number);
            std::cout << "  Adding final number: " << num << std::endl;
            result.push_back(num);
        }
        
        std::cout << "Final result size: " << result.size() << std::endl;
        return result;
    }
};

int main() {
    std::string test_json = "{\"success\": true, \"flow_value\": 2.6, \"segments\": [1, 1, 1, 1], \"statistics\": {\"num_nodes\": 4}}";
    
    SimpleJsonParser parser(test_json);
    auto segments = parser.getIntArray("segments");
    
    std::cout << "Parsed segments: ";
    for (int seg : segments) {
        std::cout << seg << " ";
    }
    std::cout << std::endl;
    
    return 0;
}


#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char *argv[]) {
    std::string json_string = R"(
        {
            "name": "John",
            "age": 30,
            "is_student": false,
            "address": {
                "street": "123 Main St",
                "city": "New York",
                "zip": "10001"
            },
            "scores": [90, 85, 88]
        }
    )";

    try {
        // 解析 JSON 字符串
        nlohmann::json j = nlohmann::json::parse(json_string);

        // 访问 JSON 对象的属性
        std::string name = j["name"];
        int age = j["age"];
        bool is_student = j["is_student"];
        std::string street = j["address"]["street"];
        std::string city = j["address"]["city"];
        std::string zip = j["address"]["zip"];

        // 访问 JSON 数组的元素
        std::vector<int> scores = j["scores"];

        // 打印解析后的数据
        std::cout << "Name: " << name << std::endl;
        std::cout << "Age: " << age << std::endl;
        std::cout << "Is Student: " << std::boolalpha << is_student
                  << std::endl;
        std::cout << "Address: " << street << ", " << city << ", " << zip
                  << std::endl;
        std::cout << "Scores: ";
        for (int score : scores) {
            std::cout << score << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
    }
    return 0;
}
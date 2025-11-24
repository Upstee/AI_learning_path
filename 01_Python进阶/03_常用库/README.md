# Python常用库

## 1. 课程概述

### 课程目标
1. 掌握requests库进行HTTP请求和数据获取
2. 熟练使用json和csv模块处理数据文件
3. 理解和使用正则表达式进行文本处理
4. 了解其他常用工具库（datetime、collections等）
5. 能够综合使用这些库解决实际问题

### 预计学习时间
- **理论学习**：6-8小时
- **代码实践**：10-12小时
- **练习巩固**：4-6小时
- **总计**：20-26小时（约1-2周）

### 难度等级
- **简单到中等** - 主要是API使用，需要多练习

### 课程定位
- **前置课程**：01_面向对象编程、02_高级特性
- **后续课程**：03_数据处理基础（NumPy、Pandas）
- **在体系中的位置**：AI开发的基础工具，数据获取和处理必备

### 学完能做什么
- 能够从网络获取数据（API、网页）
- 能够处理JSON和CSV格式的数据
- 能够使用正则表达式提取和处理文本
- 能够使用常用工具库提高开发效率

---

## 2. 前置知识检查

### 必备前置概念清单
- **函数和类**：基本的函数定义和类使用
- **文件操作**：open、read、write
- **异常处理**：try/except
- **字符串操作**：基本的字符串方法

### 回顾链接/跳转
- 如果不熟悉文件操作：Python基础教程
- 如果不熟悉异常处理：Python基础教程

### 入门小测

**选择题**（每题2分，共10分）

1. Python中如何打开文件？
   A. open()  B. file()  C. read()  D. write()
   **答案**：A

2. JSON是什么格式？
   A. 文本  B. 数据交换  C. 图片  D. 视频
   **答案**：B

3. HTTP GET请求的作用是？
   A. 发送数据  B. 获取数据  C. 删除数据  D. 更新数据
   **答案**：B

4. 正则表达式中.表示什么？
   A. 点号  B. 任意字符  C. 换行  D. 空格
   **答案**：B

5. CSV文件的分隔符通常是？
   A. 逗号  B. 分号  C. 制表符  D. 以上都可以
   **答案**：D

**评分标准**：≥8分（80%）为通过

### 不会时的补救指引
如果小测不通过，建议：
1. 复习Python基础（文件操作、字符串）
2. 了解HTTP基础知识
3. 完成基础练习后再继续

---

## 3. 核心知识点详解

### 3.1 requests库

#### 概念引入与直观类比

**类比**：requests就像"邮递员"，帮你发送请求并带回响应。

- **发送请求**：告诉服务器你想要什么
- **接收响应**：服务器返回数据给你

#### 逐步理论推导

**步骤1：安装requests**
```bash
pip install requests
```

**步骤2：发送GET请求**
```python
import requests

response = requests.get("https://api.github.com")
print(response.status_code)  # 200
print(response.text)  # 响应内容
```

**步骤3：发送POST请求**
```python
data = {"name": "Alice", "age": 25}
response = requests.post("https://httpbin.org/post", json=data)
print(response.json())
```

**步骤4：处理响应**
```python
response = requests.get("https://api.github.com")
if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"错误: {response.status_code}")
```

#### 关键性质

**优点**：
- **简单易用**：API简洁直观
- **功能完整**：支持所有HTTP方法
- **自动处理**：自动处理编码、JSON等

**适用场景**：
- API调用
- 网页爬虫
- 数据获取

---

### 3.2 JSON处理

#### 概念引入与直观类比

**类比**：JSON就像"通用语言"，让不同系统能够交换数据。

- **Python对象**：字典、列表
- **JSON字符串**：文本格式
- **转换**：Python ↔ JSON

#### 逐步理论推导

**步骤1：Python对象转JSON**
```python
import json

data = {
    "name": "Alice",
    "age": 25,
    "hobbies": ["reading", "coding"]
}

json_str = json.dumps(data, ensure_ascii=False, indent=2)
print(json_str)
```

**步骤2：JSON转Python对象**
```python
json_str = '{"name": "Alice", "age": 25}'
data = json.loads(json_str)
print(data["name"])  # Alice
```

**步骤3：文件操作**
```python
# 写入JSON文件
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# 读取JSON文件
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
```

---

### 3.3 CSV处理

#### 概念引入与直观类比

**类比**：CSV就像"表格"，用逗号分隔的表格数据。

- **行**：一条记录
- **列**：一个字段
- **分隔符**：逗号（或其他）

#### 逐步理论推导

**步骤1：读取CSV**
```python
import csv

with open("data.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["name"], row["age"])
```

**步骤2：写入CSV**
```python
data = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30}
]

with open("output.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "age"])
    writer.writeheader()
    writer.writerows(data)
```

---

### 3.4 正则表达式

#### 概念引入与直观类比

**类比**：正则表达式就像"模板"，用来匹配符合模式的文本。

- **模式**：描述文本的规则
- **匹配**：找到符合规则的文本
- **提取**：从文本中提取信息

#### 逐步理论推导

**步骤1：基本匹配**
```python
import re

text = "我的电话是13812345678"
pattern = r"\d{11}"  # 11位数字
match = re.search(pattern, text)
if match:
    print(match.group())  # 13812345678
```

**步骤2：分组提取**
```python
text = "日期：2024-01-27"
pattern = r"(\d{4})-(\d{2})-(\d{2})"
match = re.search(pattern, text)
if match:
    year, month, day = match.groups()
    print(f"年：{year}, 月：{month}, 日：{day}")
```

**步骤3：查找所有匹配**
```python
text = "电话：13812345678, 13987654321"
pattern = r"\d{11}"
matches = re.findall(pattern, text)
print(matches)  # ['13812345678', '13987654321']
```

#### 常用模式

- `\d`：数字
- `\w`：字母、数字、下划线
- `\s`：空白字符
- `.`：任意字符
- `*`：0次或多次
- `+`：1次或多次
- `?`：0次或1次
- `{n}`：恰好n次
- `{n,m}`：n到m次

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：
  - requests（需要安装：`pip install requests`）

### 4.2 从零开始的完整可运行示例

#### 示例1：使用requests获取API数据

```python
import requests
import json

def get_github_user(username):
    """获取GitHub用户信息"""
    url = f"https://api.github.com/users/{username}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # 检查HTTP错误
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None

# 使用
user_info = get_github_user("octocat")
if user_info:
    print(f"用户名: {user_info['login']}")
    print(f"关注者: {user_info['followers']}")
    print(json.dumps(user_info, indent=2, ensure_ascii=False))
```

#### 示例2：JSON数据处理

```python
import json

# 创建数据
data = {
    "students": [
        {"name": "Alice", "age": 20, "scores": [85, 90, 88]},
        {"name": "Bob", "age": 21, "scores": [92, 87, 91]}
    ]
}

# 保存到文件
with open("students.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# 从文件读取
with open("students.json", "r", encoding="utf-8") as f:
    loaded_data = json.load(f)

# 处理数据
for student in loaded_data["students"]:
    avg_score = sum(student["scores"]) / len(student["scores"])
    print(f"{student['name']}: 平均分 {avg_score:.2f}")
```

#### 示例3：CSV数据处理

```python
import csv

# 写入CSV
students = [
    {"name": "Alice", "age": 20, "score": 85},
    {"name": "Bob", "age": 21, "score": 92},
    {"name": "Charlie", "age": 19, "score": 78}
]

with open("students.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "age", "score"])
    writer.writeheader()
    writer.writerows(students)

# 读取CSV
with open("students.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"{row['name']}: {row['score']}分")
```

#### 示例4：正则表达式应用

```python
import re

def extract_emails(text):
    """提取文本中的邮箱地址"""
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, text)
    return emails

def extract_phones(text):
    """提取文本中的手机号"""
    pattern = r'1[3-9]\d{9}'
    phones = re.findall(pattern, text)
    return phones

# 使用
text = """
联系方式：
邮箱：alice@example.com, bob@test.org
电话：13812345678, 13987654321
"""

emails = extract_emails(text)
phones = extract_phones(text)

print("邮箱:", emails)
print("电话:", phones)
```

### 4.3 常见错误与排查

**错误1**：requests未安装
```python
# 错误
import requests  # ModuleNotFoundError

# 解决
# pip install requests
```

**错误2**：JSON编码问题
```python
# 错误：中文乱码
json.dumps({"name": "张三"})  # {"name": "\u5f20\u4e09"}

# 正确
json.dumps({"name": "张三"}, ensure_ascii=False)  # {"name": "张三"}
```

**错误3**：CSV文件换行问题
```python
# 错误：Windows下多空行
with open("data.csv", "w") as f:
    writer = csv.writer(f)

# 正确
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
```

### 4.4 性能/工程化小技巧

1. **使用Session复用连接**
```python
session = requests.Session()
session.get("https://api.example.com/data1")
session.get("https://api.example.com/data2")  # 复用连接
```

2. **使用生成器处理大文件**
```python
def read_large_csv(filename):
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row
```

3. **编译正则表达式提高性能**
```python
pattern = re.compile(r'\d+')
# 多次使用时，编译后的pattern更快
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：API数据获取**
使用requests获取天气API数据并解析。

**练习2：JSON文件处理**
创建一个学生信息管理系统，使用JSON存储数据。

**练习3：CSV数据分析**
读取CSV文件，计算平均值、最大值等统计信息。

**练习4：正则表达式提取**
从文本中提取所有URL、邮箱、电话等信息。

### 进阶练习（2-3题）

**练习1：综合数据获取**
从多个API获取数据，合并并保存到JSON文件。

**练习2：数据清洗**
使用正则表达式清洗CSV数据，去除无效记录。

### 挑战练习（1-2题）

**练习1：简单的数据采集系统**
实现一个完整的数据采集系统，包括数据获取、处理、存储。

---

## 6. 实际案例

### 案例：天气数据获取和分析系统

**业务背景**：
需要获取多个城市的天气数据，进行分析和存储。

**问题抽象**：
- 从API获取天气数据
- 解析JSON响应
- 存储到CSV文件
- 进行数据分析

**端到端实现**：
```python
import requests
import csv
import json
from datetime import datetime

class WeatherCollector:
    """天气数据收集器"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self, city):
        """获取城市天气"""
        params = {
            "q": city,
            "appid": self.api_key,
            "units": "metric",
            "lang": "zh_cn"
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"获取{city}天气失败: {e}")
            return None
    
    def save_to_csv(self, data_list, filename):
        """保存到CSV"""
        if not data_list:
            return
        
        fieldnames = ["城市", "温度", "湿度", "描述", "时间"]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_list)
    
    def collect_multiple_cities(self, cities):
        """收集多个城市的数据"""
        results = []
        for city in cities:
            data = self.get_weather(city)
            if data:
                results.append({
                    "城市": city,
                    "温度": data["main"]["temp"],
                    "湿度": data["main"]["humidity"],
                    "描述": data["weather"][0]["description"],
                    "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        return results

# 使用（需要API key）
# collector = WeatherCollector("your_api_key")
# cities = ["北京", "上海", "广州"]
# data = collector.collect_multiple_cities(cities)
# collector.save_to_csv(data, "weather.csv")
```

**结果解读**：
- 系统能够获取多个城市的天气数据
- 数据保存到CSV文件，便于分析

**改进方向**：
- 添加数据缓存
- 添加错误重试
- 添加数据可视化

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. requests.get()返回什么？
   A. 字符串  B. Response对象  C. 字典  D. 列表
   **答案**：B

2. json.dumps()的作用是？
   A. 解析JSON  B. 转换为JSON字符串  C. 读取文件  D. 写入文件
   **答案**：B

3. CSV文件的默认分隔符是？
   A. 逗号  B. 分号  C. 制表符  D. 空格
   **答案**：A

4. 正则表达式\d表示什么？
   A. 字母  B. 数字  C. 空白  D. 任意字符
   **答案**：B

5. response.json()的作用是？
   A. 获取文本  B. 解析JSON  C. 获取状态码  D. 获取头部
   **答案**：B

**简答题**（每题10分，共40分）

1. 说明requests库的主要用途。
   **参考答案**：用于发送HTTP请求，获取网络数据，调用API等。

2. 解释JSON和CSV的区别。
   **参考答案**：JSON是嵌套的数据结构，适合复杂数据；CSV是表格格式，适合简单表格数据。

3. 说明正则表达式的应用场景。
   **参考答案**：文本匹配、数据提取、格式验证、文本替换等。

4. 解释为什么需要处理编码问题。
   **参考答案**：不同系统使用不同编码，正确处理编码可以避免乱码问题。

### 编程实践题（20分）

编写一个函数，从API获取数据并保存到JSON文件。

**参考答案**：
```python
import requests
import json

def fetch_and_save(url, filename):
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
```

### 综合应用题（20分）

实现一个简单的数据采集系统，能够从API获取数据，使用正则表达式处理，保存到CSV文件。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《Python网络数据采集》
- 《Python数据处理》

**在线资源**：
- requests官方文档
- Python官方文档：json, csv, re模块

### 相关工具与库

- **pandas**：更强大的数据处理（后续课程）
- **beautifulsoup4**：HTML解析
- **scrapy**：网页爬虫框架

### 进阶话题指引

完成本课程后，可以学习：
- **网页爬虫**：BeautifulSoup、Scrapy
- **数据处理**：pandas、numpy
- **API设计**：Flask、FastAPI

### 下节课预告

下一课将学习：
- **04_项目实战**：综合运用所学知识完成实际项目
- 通过项目巩固和提升Python技能

### 学习建议

1. **多实践**：这些库需要大量练习
2. **阅读文档**：学会查阅官方文档
3. **做项目**：通过项目综合运用
4. **持续学习**：这些是AI开发的基础工具

---

**恭喜完成第三课！你已经掌握了Python常用库，准备好进行项目实战了！**


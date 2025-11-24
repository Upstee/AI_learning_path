# Python高级特性

## 1. 课程概述

### 课程目标
1. 掌握装饰器的定义和使用，理解装饰器的原理
2. 理解生成器和迭代器的概念，能够创建和使用生成器
3. 掌握上下文管理器的使用，理解with语句的工作原理
4. 了解元编程的基本概念和应用
5. 理解并发编程的基础，能够使用多线程和多进程

### 预计学习时间
- **理论学习**：10-12小时
- **代码实践**：15-18小时
- **练习巩固**：8-10小时
- **总计**：33-40小时（约2-3周）

### 难度等级
- **较难** - 需要理解抽象概念和Python内部机制

### 课程定位
- **前置课程**：01_面向对象编程
- **后续课程**：03_常用库、04_项目实战
- **在体系中的位置**：Python进阶的核心，提升代码质量和效率

### 学完能做什么
- 能够使用装饰器增强函数功能
- 能够使用生成器处理大数据
- 能够使用上下文管理器管理资源
- 能够编写更优雅、更高效的Python代码

---

## 2. 前置知识检查

### 必备前置概念清单
- **面向对象编程**：类、对象、方法
- **函数**：函数定义、参数、返回值、作用域
- **闭包**：理解闭包的概念
- **异常处理**：try/except/finally

### 回顾链接/跳转
- 如果不熟悉面向对象编程：`01_Python进阶/01_面向对象编程/`
- 如果不熟悉函数：Python基础教程

### 入门小测

**选择题**（每题2分，共10分）

1. Python中如何定义一个函数？
   A. function name():  B. def name():  C. define name():  D. func name():
   **答案**：B

2. 什么是闭包？
   A. 函数内部定义的函数  B. 函数返回另一个函数  C. 函数访问外部变量  D. 以上都是
   **答案**：D

3. 如何捕获异常？
   A. try/except  B. catch  C. error  D. exception
   **答案**：A

4. 类的方法第一个参数通常是什么？
   A. cls  B. self  C. this  D. me
   **答案**：B

5. Python中如何创建列表？
   A. []  B. list()  C. 以上都可以  D. 以上都不可以
   **答案**：C

**简答题**（每题5分，共10分）

1. 解释什么是闭包，并给出一个例子。
   **参考答案**：闭包是内部函数访问外部函数变量的机制。例如：
   ```python
   def outer(x):
       def inner(y):
           return x + y
       return inner
   ```

2. 说明try/except/finally的作用。
   **参考答案**：try执行可能出错的代码，except捕获异常，finally无论是否异常都会执行。

**编程题**（10分）

编写一个函数，接收一个函数和两个数字，返回函数对这两个数字的计算结果。

**参考答案**：
```python
def apply_func(func, a, b):
    return func(a, b)

# 使用
result = apply_func(lambda x, y: x + y, 3, 4)  # 7
```

**评分标准**：≥24分（80%）为通过

### 不会时的补救指引
如果小测不通过，建议：
1. 复习01_面向对象编程
2. 复习Python基础（函数、异常处理）
3. 完成基础练习后再继续

---

## 3. 核心知识点详解

### 3.1 装饰器

#### 概念引入与直观类比

**类比**：装饰器就像"包装纸"，在不改变原物品的情况下，给它添加新的功能。

例如：
- **原函数**：一个礼物
- **装饰器**：包装纸
- **装饰后的函数**：包装好的礼物（功能更多，但核心不变）

#### 逐步理论推导

**步骤1：理解函数是一等对象**
```python
def greet(name):
    return f"Hello, {name}!"

# 函数可以赋值给变量
my_func = greet
print(my_func("Alice"))  # Hello, Alice!
```

**步骤2：函数可以作为参数**
```python
def call_twice(func, arg):
    return func(arg), func(arg)

result = call_twice(greet, "Bob")
```

**步骤3：函数可以作为返回值**
```python
def create_greeter(prefix):
    def greeter(name):
        return f"{prefix}, {name}!"
    return greeter

hello = create_greeter("Hello")
print(hello("Charlie"))  # Hello, Charlie!
```

**步骤4：创建装饰器**
```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("函数执行前")
        result = func(*args, **kwargs)
        print("函数执行后")
        return result
    return wrapper

@my_decorator
def greet(name):
    return f"Hello, {name}!"

greet("David")  # 函数执行前\nHello, David!\n函数执行后
```

#### 数学公式与必要证明

装饰器本质上是函数组合：
- 如果f是原函数，d是装饰器
- 装饰后的函数 = d(f)
- 调用时：d(f)(args) = wrapper(args)

#### 图解/可视化

```
原函数
┌─────────────┐
│   greet()   │
└─────────────┘
       │
       │ @decorator
       ▼
装饰器
┌─────────────┐
│  decorator  │
│   wrapper   │
└─────────────┘
       │
       ▼
装饰后的函数
┌─────────────┐
│ wrapper()   │
│  - 前处理   │
│  - greet()  │
│  - 后处理   │
└─────────────┘
```

#### 算法伪代码

```
装饰器模式：
1. 定义装饰器函数（接收函数作为参数）
2. 定义内部包装函数（wrapper）
3. 在wrapper中执行额外操作
4. 调用原函数
5. 返回wrapper函数
6. 使用@语法糖应用装饰器
```

#### 关键性质

**优点**：
- **不修改原函数**：保持原函数不变
- **代码复用**：一个装饰器可以用于多个函数
- **功能增强**：轻松添加新功能

**缺点**：
- **调试困难**：函数名可能改变
- **性能开销**：额外的函数调用

**适用场景**：
- 日志记录
- 性能计时
- 权限检查
- 缓存

#### 常见误区与对比

**误区1**：装饰器会修改原函数
- **正确理解**：装饰器返回新函数，原函数不变

**误区2**：装饰器只能用于函数
- **正确理解**：也可以用于类和方法

---

### 3.2 生成器

#### 概念引入与直观类比

**类比**：生成器就像"懒人清单"，需要时才计算，而不是一次性全部计算。

- **列表**：一次性把所有东西都准备好（占用内存）
- **生成器**：需要时才产生下一个值（节省内存）

#### 逐步理论推导

**步骤1：理解问题（列表的缺点）**
```python
# 创建大列表占用大量内存
big_list = [x**2 for x in range(1000000)]  # 占用内存
```

**步骤2：使用生成器表达式**
```python
# 生成器不立即创建所有值
gen = (x**2 for x in range(1000000))  # 不占用内存
```

**步骤3：使用yield创建生成器函数**
```python
def squares(n):
    for i in range(n):
        yield i**2  # yield产生值并暂停

gen = squares(1000000)  # 不执行，返回生成器对象
print(next(gen))  # 0
print(next(gen))  # 1
```

**步骤4：使用生成器**
```python
for value in squares(10):
    print(value)  # 0, 1, 4, 9, ...
```

#### 数学公式与必要证明

生成器是惰性求值的实现：
- 列表：Eager Evaluation（立即求值）
- 生成器：Lazy Evaluation（惰性求值）

内存复杂度：
- 列表：O(n)
- 生成器：O(1)

#### 图解/可视化

```
列表（立即求值）
┌─────────────────┐
│ [0, 1, 4, 9, ...] │  ← 所有值都在内存中
└─────────────────┘

生成器（惰性求值）
┌─────────────────┐
│ 生成器对象      │
│ 状态：暂停      │
│ 当前位置：0     │
└─────────────────┘
       │
       │ next()
       ▼
   产生值：0
       │
       │ next()
       ▼
   产生值：1
```

#### 关键性质

**优点**：
- **内存效率**：不占用大量内存
- **惰性计算**：需要时才计算
- **无限序列**：可以表示无限序列

**缺点**：
- **只能遍历一次**：生成器用完后不能再次使用
- **不能索引**：不能像列表那样用索引访问

**适用场景**：
- 处理大文件
- 无限序列
- 数据流处理
- 节省内存

---

### 3.3 上下文管理器

#### 概念引入与直观类比

**类比**：上下文管理器就像"自动门"，进入时自动开门，离开时自动关门。

- **进入**：自动执行初始化（开门）
- **使用**：在上下文中操作
- **离开**：自动执行清理（关门）

#### 逐步理论推导

**步骤1：理解资源管理问题**
```python
# 需要手动关闭文件
file = open("data.txt", "r")
content = file.read()
file.close()  # 容易忘记
```

**步骤2：使用try/finally**
```python
file = open("data.txt", "r")
try:
    content = file.read()
finally:
    file.close()  # 确保关闭
```

**步骤3：使用with语句**
```python
with open("data.txt", "r") as file:
    content = file.read()
# 自动关闭文件
```

**步骤4：自定义上下文管理器**
```python
class MyContext:
    def __enter__(self):
        print("进入上下文")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("离开上下文")
        return False

with MyContext():
    print("在上下文中")
```

#### 关键性质

**优点**：
- **自动管理**：自动处理资源的获取和释放
- **异常安全**：即使发生异常也能正确清理
- **代码简洁**：代码更清晰

**适用场景**：
- 文件操作
- 数据库连接
- 线程锁
- 临时设置

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：无（使用标准库）

### 4.2 从零开始的完整可运行示例

#### 示例1：装饰器

```python
import time
from functools import wraps

def timer(func):
    """计时装饰器"""
    @wraps(func)  # 保持原函数信息
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end - start:.4f}秒")
        return result
    return wrapper

def logger(func):
    """日志装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"调用函数: {func.__name__}")
        print(f"参数: args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"返回值: {result}")
        return result
    return wrapper

@timer
@logger
def add(a, b):
    """加法函数"""
    time.sleep(0.1)  # 模拟耗时操作
    return a + b

result = add(3, 4)
```

**运行结果**：
```
调用函数: add
参数: args=(3, 4), kwargs={}
返回值: 7
add 执行时间: 0.1001秒
```

#### 示例2：生成器

```python
def fibonacci(n):
    """生成斐波那契数列"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 使用生成器
for num in fibonacci(10):
    print(num, end=" ")
print()

# 生成器表达式
squares = (x**2 for x in range(10))
print(list(squares))
```

**运行结果**：
```
0 1 1 2 3 5 8 13 21 34 
[0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

#### 示例3：上下文管理器

```python
class FileManager:
    """文件管理器"""
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        print(f"打开文件: {self.filename}")
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            print(f"关闭文件: {self.filename}")
        return False  # 不抑制异常

# 使用
with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")
# 自动关闭文件
```

**运行结果**：
```
打开文件: test.txt
关闭文件: test.txt
```

### 4.3 常见错误与排查

**错误1**：装饰器忘记使用@wraps
```python
# 错误：函数名变成wrapper
@timer
def my_func():
    pass

print(my_func.__name__)  # wrapper（错误）

# 正确
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ...
    return wrapper
```

**错误2**：生成器只能遍历一次
```python
gen = (x for x in range(5))
list(gen)  # [0, 1, 2, 3, 4]
list(gen)  # []（空，因为已经用完了）
```

**错误3**：上下文管理器忘记实现__exit__
```python
# 错误
class BadContext:
    def __enter__(self):
        return self
    # 缺少__exit__

# 正确
class GoodContext:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
```

### 4.4 性能/工程化小技巧

1. **使用@lru_cache装饰器**：缓存函数结果
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_func(n):
    # 复杂计算
    return result
```

2. **使用生成器处理大文件**
```python
def read_large_file(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()
```

3. **使用contextlib简化上下文管理器**
```python
from contextlib import contextmanager

@contextmanager
def my_context():
    print("进入")
    yield
    print("离开")
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：创建缓存装饰器**
创建一个装饰器，缓存函数的返回值。

**练习2：创建生成器函数**
创建一个生成器，生成所有偶数。

**练习3：创建上下文管理器**
创建一个上下文管理器，用于临时修改工作目录。

### 进阶练习（2-3题）

**练习1：带参数的装饰器**
创建一个装饰器，可以指定重试次数。

**练习2：生成器管道**
创建多个生成器，实现数据处理的管道。

### 挑战练习（1-2题）

**练习1：完整的装饰器系统**
实现日志、计时、缓存、重试等多个装饰器，并能够组合使用。

---

## 6. 实际案例

### 案例：性能监控装饰器

**业务背景**：
需要监控函数的执行时间和调用次数。

**问题抽象**：
- 需要记录函数执行时间
- 需要统计函数调用次数
- 需要记录函数参数和返回值

**端到端实现**：
```python
import time
from functools import wraps
from collections import defaultdict

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self):
        self.call_count = defaultdict(int)
        self.total_time = defaultdict(float)
    
    def monitor(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.call_count[func.__name__] += 1
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            self.total_time[func.__name__] += elapsed
            return result
        return wrapper
    
    def get_stats(self):
        """获取统计信息"""
        stats = {}
        for func_name in self.call_count:
            stats[func_name] = {
                'calls': self.call_count[func_name],
                'total_time': self.total_time[func_name],
                'avg_time': self.total_time[func_name] / self.call_count[func_name]
            }
        return stats

# 使用
monitor = PerformanceMonitor()

@monitor.monitor
def slow_function(n):
    time.sleep(0.1)
    return n * 2

@monitor.monitor
def fast_function(n):
    return n + 1

# 执行函数
slow_function(5)
fast_function(10)
slow_function(3)

# 查看统计
print(monitor.get_stats())
```

**结果解读**：
- 可以监控每个函数的调用次数和执行时间
- 可以分析性能瓶颈

**改进方向**：
- 添加内存使用监控
- 添加异常统计
- 导出到文件

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. 装饰器的@符号是什么？
   A. 语法糖  B. 运算符  C. 注释  D. 字符串
   **答案**：A

2. yield关键字的作用是？
   A. 返回  B. 产生值并暂停  C. 停止  D. 继续
   **答案**：B

3. with语句需要实现哪些方法？
   A. __enter__和__exit__  B. __init__和__del__  C. __get__和__set__  D. __call__
   **答案**：A

4. 生成器的主要优势是？
   A. 速度快  B. 内存效率  C. 功能多  D. 易用
   **答案**：B

5. @wraps装饰器的作用是？
   A. 包装函数  B. 保持原函数信息  C. 加速执行  D. 缓存结果
   **答案**：B

**简答题**（每题10分，共40分）

1. 解释装饰器的工作原理。
   **参考答案**：装饰器是接收函数作为参数，返回新函数的函数。新函数包装原函数，添加额外功能。

2. 说明生成器和列表的区别。
   **参考答案**：列表立即计算所有值并存储在内存中；生成器惰性计算，需要时才产生值，节省内存。

3. 解释上下文管理器的作用。
   **参考答案**：上下文管理器自动管理资源的获取和释放，确保资源正确清理，即使发生异常也能处理。

4. 说明装饰器的应用场景。
   **参考答案**：日志记录、性能计时、权限检查、缓存、重试机制等。

### 编程实践题（20分）

创建一个装饰器，实现函数调用的重试机制。

**参考答案**：
```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
```

### 综合应用题（20分）

设计一个完整的日志系统，使用装饰器记录函数调用。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《流畅的Python》- 第7、9、15、16章
- 《Effective Python》- 第31-40条

**在线资源**：
- Python官方文档：Decorators, Generators
- Real Python：Decorators, Generators

### 相关工具与库

- **functools**：装饰器工具（lru_cache, wraps）
- **contextlib**：上下文管理器工具
- **asyncio**：异步编程（高级话题）

### 进阶话题指引

完成本课程后，可以学习：
- **元类**：类的类
- **描述符**：属性访问控制
- **异步编程**：async/await
- **并发编程**：多线程、多进程

### 下节课预告

下一课将学习：
- **03_常用库**：requests、json/csv处理、正则表达式
- 这些库是AI开发中常用的工具

### 学习建议

1. **多实践**：装饰器和生成器需要大量练习
2. **理解原理**：不要只记住用法，要理解为什么
3. **阅读代码**：阅读使用这些特性的开源代码
4. **持续学习**：这些是Python的核心特性，需要深入理解

---

**恭喜完成第二课！你已经掌握了Python的高级特性，准备好学习常用库了！**


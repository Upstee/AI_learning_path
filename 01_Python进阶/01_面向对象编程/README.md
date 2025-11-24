# 面向对象编程

## 1. 课程概述

### 课程目标
1. 理解面向对象编程的核心概念（类、对象、封装、继承、多态）
2. 能够定义和使用类，创建对象实例
3. 掌握继承机制，能够设计类层次结构
4. 理解和使用特殊方法（`__init__`、`__str__`、`__repr__`等）
5. 能够应用面向对象思想解决实际问题

### 预计学习时间
- **理论学习**：8-10小时
- **代码实践**：12-15小时
- **练习巩固**：6-8小时
- **总计**：26-33小时（约2-3周）

### 难度等级
- **中等** - 需要理解抽象概念，但通过实践可以掌握

### 课程定位
- **前置课程**：Python基础（变量、函数、数据结构）
- **后续课程**：02_高级特性、03_常用库
- **在体系中的位置**：Python进阶的基础，AI开发必备技能

### 学完能做什么
- 能够使用面向对象思想组织代码
- 能够设计和实现类层次结构
- 能够使用继承和多态提高代码复用性
- 能够阅读和理解使用OOP的Python代码库

---

## 2. 前置知识检查

### 必备前置概念清单
- **变量和数据类型**：int, float, str, list, dict等
- **函数定义**：def关键字、参数、返回值
- **控制流**：if/else, for, while
- **数据结构**：列表、字典的基本操作

### 回顾链接/跳转
- 如果对Python基础不熟悉，请先学习Python基础课程
- 参考：`00_开始_AI全景导览/03_必备知识概览/编程技能要求.md`

### 入门小测

**选择题**（每题2分，共10分）

1. Python中如何定义一个函数？
   A. function name():  B. def name():  C. define name():  D. func name():
   **答案**：B

2. 以下哪个是字典？
   A. [1, 2, 3]  B. (1, 2, 3)  C. {1: 'a', 2: 'b'}  D. "123"
   **答案**：C

3. 如何访问列表的第一个元素？
   A. list[0]  B. list[1]  C. list.first()  D. list.get(0)
   **答案**：A

4. 以下哪个关键字用于定义函数？
   A. function  B. def  C. define  D. func
   **答案**：B

5. Python中如何创建一个空列表？
   A. []  B. list()  C. 以上都可以  D. 以上都不可以
   **答案**：C

**简答题**（每题5分，共10分）

1. 解释什么是函数参数，什么是函数返回值。
   **参考答案**：函数参数是传递给函数的数据，函数返回值是函数执行后返回的结果。

2. 说明列表和字典的区别。
   **参考答案**：列表是有序的序列，通过索引访问；字典是键值对的集合，通过键访问值。

**编程题**（10分）

编写一个函数，接收一个数字列表，返回列表中的最大值。

**参考答案**：
```python
def find_max(numbers):
    if not numbers:
        return None
    max_value = numbers[0]
    for num in numbers:
        if num > max_value:
            max_value = num
    return max_value

# 或者使用内置函数
def find_max(numbers):
    return max(numbers) if numbers else None
```

**评分标准**：
- 选择题：每题2分，共10分
- 简答题：每题5分，共10分
- 编程题：10分
- **总分**：30分，≥24分（80%）为通过

### 不会时的补救指引
如果小测不通过，建议：
1. 复习Python基础语法
2. 完成更多基础练习
3. 参考：`00_开始_AI全景导览/03_必备知识概览/编程技能要求.md`
4. 完成基础练习后再继续本课程

---

## 3. 核心知识点详解

### 3.1 类与对象

#### 概念引入与直观类比

**类比**：类就像是一个"模板"或"蓝图"，对象是根据这个模板创建的"实例"。

- **类（Class）**：定义了一类对象的共同属性和方法
  - 例如："汽车"类定义了所有汽车都有颜色、品牌、速度等属性
- **对象（Object）**：根据类创建的具体实例
  - 例如：根据"汽车"类可以创建"我的红色奔驰"这个具体对象

#### 逐步理论推导

**步骤1：定义类**
```python
class Car:
    pass
```

**步骤2：添加属性（实例变量）**
```python
class Car:
    def __init__(self, color, brand):
        self.color = color  # 实例变量
        self.brand = brand
```

**步骤3：添加方法**
```python
class Car:
    def __init__(self, color, brand):
        self.color = color
        self.brand = brand
    
    def start(self):  # 方法
        print(f"{self.brand}汽车启动了")
```

**步骤4：创建对象**
```python
my_car = Car("红色", "奔驰")
my_car.start()  # 调用方法
```

#### 数学公式与必要证明

面向对象编程主要是概念性的，不涉及复杂的数学公式。但我们可以用集合论来理解：

- **类**：定义了一个集合（所有可能的对象）
- **对象**：集合中的一个元素
- **继承**：子集关系（子类是父类的子集）

#### 图解/可视化

```
类（Class）
┌─────────────────┐
│   Car           │
├─────────────────┤
│ 属性：          │
│ - color         │
│ - brand         │
│                 │
│ 方法：          │
│ - __init__()    │
│ - start()       │
└─────────────────┘
        │
        │ 实例化
        ▼
对象（Object）
┌─────────────────┐
│ my_car          │
├─────────────────┤
│ color = "红色"  │
│ brand = "奔驰"  │
└─────────────────┘
```

#### 算法伪代码

```
定义类：
1. 使用class关键字
2. 定义__init__方法（构造函数）
3. 定义其他方法

创建对象：
1. 调用类名（传入参数）
2. Python自动调用__init__
3. 返回对象实例

使用对象：
1. 通过对象访问属性：object.attribute
2. 通过对象调用方法：object.method()
```

#### 关键性质

**优点**：
- **封装**：数据和操作封装在一起
- **复用**：可以创建多个对象
- **组织**：代码结构清晰

**缺点**：
- **复杂度**：简单问题可能过度设计
- **性能**：比函数调用稍慢（通常可忽略）

**适用场景**：
- 需要表示现实世界的实体
- 需要管理复杂的状态
- 需要代码复用和组织

#### 常见误区与对比

**误区1**：认为类必须很复杂
- **正确理解**：简单的类也很有用

**误区2**：过度使用继承
- **正确理解**：优先使用组合而非继承

**对比**：面向对象 vs 面向过程
- **面向过程**：以函数为中心，适合简单问题
- **面向对象**：以对象为中心，适合复杂问题

---

### 3.2 继承

#### 概念引入与直观类比

**类比**：继承就像"遗传"，子类继承父类的特征，但可以有自己独特的特征。

- **父类（基类）**：定义通用特征
- **子类（派生类）**：继承父类特征，添加新特征

例如：
- **父类**：动物（有名字、会吃）
- **子类**：狗（继承动物的特征，添加"会叫"）

#### 逐步理论推导

**步骤1：定义父类**
```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def eat(self):
        print(f"{self.name}在吃东西")
```

**步骤2：定义子类（继承父类）**
```python
class Dog(Animal):  # 继承Animal
    def __init__(self, name, breed):
        super().__init__(name)  # 调用父类构造函数
        self.breed = breed
    
    def bark(self):  # 子类特有方法
        print(f"{self.name}在叫")
```

**步骤3：使用继承**
```python
my_dog = Dog("旺财", "金毛")
my_dog.eat()   # 继承自父类
my_dog.bark()  # 子类特有
```

#### 数学公式与必要证明

继承关系可以用集合论表示：
- 如果B继承A，则B ⊆ A（B是A的子集）
- 所有B的对象都是A的对象

#### 图解/可视化

```
父类：Animal
┌──────────────┐
│ name         │
│ eat()        │
└──────────────┘
       ▲
       │ 继承
       │
子类：Dog
┌──────────────┐
│ name (继承)  │
│ eat() (继承) │
│ breed (新增) │
│ bark() (新增)│
└──────────────┘
```

#### 关键性质

**优点**：
- **代码复用**：子类自动获得父类功能
- **扩展性**：容易添加新功能
- **多态**：可以用父类引用子类对象

**缺点**：
- **耦合**：子类依赖父类
- **复杂性**：继承层次过深难以理解

---

### 3.3 特殊方法

#### 概念引入与直观类比

**类比**：特殊方法就像"魔法方法"，让对象能够响应Python的内置操作。

例如：
- `__init__`：对象创建时的"初始化魔法"
- `__str__`：打印对象时的"字符串魔法"
- `__len__`：使用len()时的"长度魔法"

#### 常用特殊方法

**`__init__`**：构造函数
```python
class Person:
    def __init__(self, name):
        self.name = name
```

**`__str__`**：字符串表示（用户友好）
```python
def __str__(self):
    return f"Person(name={self.name})"
```

**`__repr__`**：对象表示（开发者友好）
```python
def __repr__(self):
    return f"Person('{self.name}')"
```

**`__len__`**：长度
```python
def __len__(self):
    return len(self.items)
```

**`__getitem__`**：索引访问
```python
def __getitem__(self, index):
    return self.items[index]
```

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：无（使用标准库）

### 4.2 从零开始的完整可运行示例

#### 示例1：基础类定义

```python
# 定义Person类
class Person:
    """人员类"""
    
    def __init__(self, name, age):
        """
        构造函数
        :param name: 姓名
        :param age: 年龄
        """
        self.name = name  # 实例变量
        self.age = age
    
    def introduce(self):
        """自我介绍"""
        return f"我是{self.name}，{self.age}岁"
    
    def have_birthday(self):
        """过生日，年龄+1"""
        self.age += 1
        print(f"{self.name}过生日了，现在{self.age}岁")

# 创建对象
person1 = Person("张三", 20)
person2 = Person("李四", 25)

# 使用对象
print(person1.introduce())  # 我是张三，20岁
person1.have_birthday()      # 张三过生日了，现在21岁
print(person1.introduce())  # 我是张三，21岁
```

**运行结果**：
```
我是张三，20岁
张三过生日了，现在21岁
我是张三，21岁
```

#### 示例2：继承

```python
# 父类：动物
class Animal:
    def __init__(self, name):
        self.name = name
    
    def eat(self):
        print(f"{self.name}在吃东西")
    
    def sleep(self):
        print(f"{self.name}在睡觉")

# 子类：狗
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # 调用父类构造函数
        self.breed = breed
    
    def bark(self):
        print(f"{self.name}（{self.breed}）在叫：汪汪！")
    
    def eat(self):  # 重写父类方法
        print(f"{self.name}在吃狗粮")

# 使用
my_dog = Dog("旺财", "金毛")
my_dog.eat()    # 旺财在吃狗粮（重写的方法）
my_dog.sleep()  # 旺财在睡觉（继承的方法）
my_dog.bark()   # 旺财（金毛）在叫：汪汪！（子类特有）
```

**运行结果**：
```
旺财在吃狗粮
旺财在睡觉
旺财（金毛）在叫：汪汪！
```

#### 示例3：特殊方法

```python
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
    
    def __str__(self):
        """用户友好的字符串表示"""
        return f"《{self.title}》- {self.author}"
    
    def __repr__(self):
        """开发者友好的对象表示"""
        return f"Book('{self.title}', '{self.author}', {self.pages})"
    
    def __len__(self):
        """返回页数"""
        return self.pages
    
    def __eq__(self, other):
        """判断两本书是否相同（标题和作者相同）"""
        if not isinstance(other, Book):
            return False
        return self.title == other.title and self.author == other.author

# 使用
book1 = Book("Python编程", "张三", 300)
book2 = Book("Python编程", "张三", 250)
book3 = Book("Java编程", "李四", 400)

print(book1)           # 《Python编程》- 张三（调用__str__）
print(repr(book1))     # Book('Python编程', '张三', 300)（调用__repr__）
print(len(book1))      # 300（调用__len__）
print(book1 == book2)  # True（调用__eq__）
print(book1 == book3)  # False
```

**运行结果**：
```
《Python编程》- 张三
Book('Python编程', '张三', 300)
300
True
False
```

### 4.3 常见错误与排查

**错误1**：忘记self参数
```python
# 错误
def introduce(self):
    return f"我是{name}"  # NameError: name 'name' is not defined

# 正确
def introduce(self):
    return f"我是{self.name}"
```

**错误2**：忘记调用super().__init__()
```python
# 错误
class Dog(Animal):
    def __init__(self, name, breed):
        self.breed = breed  # 没有初始化name

# 正确
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)
        self.breed = breed
```

**错误3**：混淆类变量和实例变量
```python
class Person:
    count = 0  # 类变量（所有对象共享）
    
    def __init__(self, name):
        self.name = name  # 实例变量（每个对象独有）
        Person.count += 1
```

### 4.4 性能/工程化小技巧

1. **使用@property装饰器**：将方法转换为属性
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def area(self):
        return 3.14 * self._radius ** 2
```

2. **使用__slots__**：限制实例变量，节省内存
```python
class Point:
    __slots__ = ['x', 'y']  # 只能有x和y属性
```

3. **使用dataclass**（Python 3.7+）：简化类定义
```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
```

### 4.5 建议的动手修改点

1. 修改Person类，添加更多属性和方法
2. 创建更多子类，练习继承
3. 实现更多特殊方法
4. 尝试使用@property装饰器

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：创建Student类**
创建一个Student类，包含姓名、学号、成绩列表。实现方法：
- `add_score(score)`: 添加成绩
- `get_average()`: 计算平均分
- `get_max_score()`: 获取最高分

**参考答案**：
```python
class Student:
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id
        self.scores = []
    
    def add_score(self, score):
        self.scores.append(score)
    
    def get_average(self):
        if not self.scores:
            return 0
        return sum(self.scores) / len(self.scores)
    
    def get_max_score(self):
        return max(self.scores) if self.scores else 0
```

**练习2：继承练习**
创建Vehicle父类和Car、Bike子类，实现继承关系。

**练习3：特殊方法练习**
为Student类实现`__str__`和`__repr__`方法。

### 进阶练习（2-3题）

**练习1：银行账户系统**
创建一个BankAccount类，实现存款、取款、查询余额功能，并添加异常处理。

**练习2：图书管理系统**
创建Book类和Library类，实现图书的添加、删除、查询功能。

### 挑战练习（1-2题）

**练习1：完整的学校管理系统**
创建Student、Teacher、Course等类，实现完整的学校管理系统。

---

## 6. 实际案例

### 案例：简单的图书管理系统

**业务背景**：
图书馆需要管理系统来管理图书的借阅和归还。

**问题抽象**：
- 图书：有书名、作者、ISBN、是否借出
- 图书馆：管理多本图书
- 操作：借书、还书、查询

**端到端实现**：
```python
class Book:
    def __init__(self, title, author, isbn):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.is_borrowed = False
    
    def __str__(self):
        status = "已借出" if self.is_borrowed else "可借"
        return f"《{self.title}》- {self.author} [{status}]"

class Library:
    def __init__(self):
        self.books = []
    
    def add_book(self, book):
        self.books.append(book)
        print(f"添加图书：{book.title}")
    
    def borrow_book(self, isbn):
        for book in self.books:
            if book.isbn == isbn:
                if book.is_borrowed:
                    print(f"《{book.title}》已被借出")
                else:
                    book.is_borrowed = True
                    print(f"成功借出《{book.title}》")
                return
        print(f"未找到ISBN为{isbn}的图书")
    
    def return_book(self, isbn):
        for book in self.books:
            if book.isbn == isbn:
                if book.is_borrowed:
                    book.is_borrowed = False
                    print(f"成功归还《{book.title}》")
                else:
                    print(f"《{book.title}》未被借出")
                return
        print(f"未找到ISBN为{isbn}的图书")
    
    def list_books(self):
        print("\n图书列表：")
        for book in self.books:
            print(f"  {book}")

# 使用
library = Library()
book1 = Book("Python编程", "张三", "001")
book2 = Book("Java编程", "李四", "002")

library.add_book(book1)
library.add_book(book2)
library.list_books()
library.borrow_book("001")
library.list_books()
library.return_book("001")
library.list_books()
```

**结果解读**：
- 系统能够管理图书的添加、借阅、归还
- 可以查询图书状态

**改进方向**：
- 添加借阅者信息
- 添加借阅期限
- 添加逾期处理
- 使用数据库存储

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. Python中如何定义类？
   A. class MyClass:  B. def MyClass:  C. Class MyClass:  D. myclass:
   **答案**：A

2. `__init__`方法的作用是？
   A. 初始化对象  B. 销毁对象  C. 打印对象  D. 比较对象
   **答案**：A

3. 如何调用父类的方法？
   A. parent.method()  B. super().method()  C. Parent.method()  D. self.parent.method()
   **答案**：B

4. `self`参数代表什么？
   A. 类本身  B. 对象实例  C. 父类  D. 子类
   **答案**：B

5. 继承的语法是？
   A. class Child(Parent):  B. class Child: Parent  C. Child extends Parent  D. Child inherits Parent
   **答案**：A

**简答题**（每题10分，共40分）

1. 解释类和对象的区别。
   **参考答案**：类是模板，对象是根据模板创建的实例。

2. 说明继承的作用和好处。
   **参考答案**：继承允许子类复用父类的代码，提高代码复用性和可维护性。

3. 解释`__init__`、`__str__`、`__repr__`的区别。
   **参考答案**：`__init__`是构造函数，`__str__`是用户友好的字符串表示，`__repr__`是开发者友好的对象表示。

4. 说明封装的概念。
   **参考答案**：封装是将数据和操作数据的方法封装在一起，隐藏内部实现细节。

### 编程实践题（20分）

创建一个Shape父类和Circle、Rectangle子类，实现多态。

**参考答案**：
```python
class Shape:
    def area(self):
        raise NotImplementedError
    
    def perimeter(self):
        raise NotImplementedError

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14 * self.radius

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)
```

### 综合应用题（20分）

设计一个简单的学生成绩管理系统，包含Student类和GradeBook类。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《Python编程：从入门到实践》- 面向对象章节
- 《流畅的Python》- 第1、9、10章
- 《Effective Python》- 第25-30条

**在线课程**：
- Python官方教程：Classes
- Real Python：Object-Oriented Programming

### 相关工具与库

- **dataclasses**：简化类定义
- **attrs**：更强大的类定义工具
- **pydantic**：数据验证

### 进阶话题指引

完成本课程后，可以学习：
- **设计模式**：单例、工厂、观察者等
- **元类**：类的类
- **描述符**：属性访问控制
- **抽象基类**：定义接口

### 下节课预告

下一课将学习：
- **02_高级特性**：装饰器、生成器、上下文管理器
- 这些特性将让你的Python代码更优雅、更强大

### 学习建议

1. **多实践**：每学一个概念，立即写代码
2. **做项目**：通过项目巩固知识
3. **阅读代码**：阅读开源项目的代码
4. **持续学习**：面向对象是编程的基础，需要持续练习

---

**恭喜完成第一课！你已经掌握了面向对象编程的基础，准备好学习Python的高级特性了！**


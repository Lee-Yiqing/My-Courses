# Python基础入门课程

## 课程信息
- **讲师**：待填写
- **平台**：待填写
- **时长**：待填写
- **难度**：初级
- **学习时间**：待填写

## 课程大纲
1. Python简介和环境搭建
2. 基本数据类型
3. 控制流程
4. 函数和模块
5. 文件操作
6. 异常处理

## 重点内容

### 1. Python环境搭建
- 安装Python解释器
- 配置IDE（推荐VS Code或PyCharm）
- 虚拟环境的使用

```python
# 创建虚拟环境
python -m venv myenv

# 激活虚拟环境（Windows）
myenv\Scripts\activate

# 激活虚拟环境（Linux/Mac）
source myenv/bin/activate
```

### 2. 基本数据类型

```python
# 数字类型
integer_num = 10
float_num = 3.14

# 字符串
string = "Hello, Python!"
multi_line = """这是
多行字符串"""

# 列表
my_list = [1, 2, 3, 4, 5]

# 字典
my_dict = {"name": "张三", "age": 25}

# 集合
my_set = {1, 2, 3, 4, 5}
```

### 3. 控制流程

```python
# 条件判断
score = 85
if score >= 90:
    print("优秀")
elif score >= 60:
    print("及格")
else:
    print("不及格")

# for循环
for i in range(5):
    print(f"第{i+1}次循环")

# while循环
count = 0
while count < 5:
    print(count)
    count += 1
```

### 4. 函数定义

```python
# 基本函数
def greet(name):
    """问候函数"""
    return f"你好, {name}!"

# 带默认参数的函数
def power(base, exponent=2):
    """计算幂"""
    return base ** exponent

# Lambda函数
square = lambda x: x ** 2
print(square(5))  # 输出: 25
```

### 5. 文件操作

```python
# 写入文件
with open('example.txt', 'w', encoding='utf-8') as f:
    f.write('这是一行文本\n')

# 读取文件
with open('example.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    print(content)
```

### 6. 异常处理

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("错误：除数不能为零")
except Exception as e:
    print(f"发生错误：{e}")
finally:
    print("执行清理操作")
```

## 实践项目

### 项目：简单计算器
实现一个支持加减乘除的计算器程序。

**实现要点**：
- 用户输入两个数字和运算符
- 根据运算符执行相应计算
- 处理除零异常
- 支持循环计算

**遇到的问题及解决方案**：
1. 输入验证：使用try-except处理非法输入
2. 浮点数精度：使用round()函数控制小数位数

## 学习心得

**收获**：
- 掌握了Python基本语法和数据结构
- 理解了函数式编程的思想
- 学会了使用异常处理机制

**感悟**：
- Python语法简洁优雅，易于学习
- 动手实践非常重要，要多写代码
- 阅读官方文档能够深入理解语言特性

**需要改进的地方**：
- 需要加强算法和数据结构的学习
- 要多做项目实践，提升编程能力
- 培养良好的代码规范习惯

## 参考资料
- [Python官方文档](https://docs.python.org/zh-cn/)
- [廖雪峰Python教程](https://www.liaoxuefeng.com/wiki/1016959663602400)
- [Python编程：从入门到实践](https://book.douban.com/subject/26829016/)

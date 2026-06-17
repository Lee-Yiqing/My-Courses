# DSA 机考 Cheatsheet — Part 1: Python 基础 + 语法速查

---

## 一、Python 基础语法速查

### 1. 变量与类型

```python
x = 10          # int, Python整数无溢出(自动变bigint)
x = 3.14        # float
x = "hello"     # str, 不可变
x = True        # bool, True=1, False=0
x = None        # 空值, 类似null

# 类型转换
int("42")       # str → int, 得42
float("3.14")   # str → float
str(42)         # int → str, 得"42"
bool(0)         # → False; bool(1) → True; bool("") → False; bool("x") → True

# 整数除法 vs 浮点除法
7 // 2          # 整除=3 (向下取整,注意负数: -7//2=-4)
7 / 2           # 浮点=3.5
7 % 2           # 取余=1
divmod(7, 2)    # 同时返回(3, 1)
```

### 2. 字符串操作

```python
s = "hello world"
s[0]           # 'h', 索引从0开始
s[-1]          # 'd', 倒数第一个
s[2:5]         # 'llo', 切片[起:止](不含止)
s[::-1]        # 'dlrow olleh', 整串反转
s[:3]          # 'hel', 前3个
s[3:]          # 'lo world', 第3个之后

len(s)         # 11, 长度
s.upper()      # 'HELLO WORLD'
s.lower()      # 'hello world'
s.strip()      # 去两端空白; lstrip()去左; rstrip()去右
s.lstrip('0')  # 去左端0, "007"→"7"; 空串→""
s.split(',')   # 按逗号分割→列表; 无参数=按空白分割
','.join(['a','b','c'])  # 'a,b,c', 用逗号连接列表
s.replace('o','0')       # 替换所有o为0
s.find('wor')            # 6, 找子串位置; 找不到→-1
s.count('o')             # 2, 计数
s.startswith('hel')      # True
s.endswith('ld')         # True
s.isdigit()              # 是否全数字字符
s.isalpha()              # 是否全字母

# 格式化
f"result={x}"           # f-string, 最推荐
f"{x:.2f}"              # 保留2位小数
f"{x:>10}"              # 右对齐宽度10
f"{x:<10}"              # 左对齐宽度10
"{}+{}={}".format(a,b,a+b)  # format方法,备选
```

### 3. 列表 (list) — 最核心的数据结构

```python
# 创建
a = [1, 2, 3]           # 直接创建
a = [0] * 10            # 10个0, 注意:嵌套时[[0]*3]*3是浅拷贝!
a = [[0]*3 for _ in range(3)]  # 正确创建3x3零矩阵

# 访问
a[0]                    # 第1个元素
a[-1]                   # 最后一个
a[1:3]                  # 切片[起:止],不含止

# 修改
a.append(4)             # 尾部添加, O(1)
a.pop()                 # 弹出尾部, O(1)
a.pop(0)                # 弹出头部, O(n)! 大数据别用
a.insert(0, x)          # 头部插入, O(n)! 大数据别用
a[i] = x                # 直接赋值, O(1)
a.extend([4,5])         # 批量追加, 等价 a+=b

# 查找
a.index(x)              # 第一个x的位置, 找不到报ValueError
x in a                  # 是否存在, O(n)
a.count(x)              # x出现次数

# 排序
a.sort()                # 原地升序
a.sort(reverse=True)    # 原地降序
a.sort(key=lambda x: x[1])  # 按第2个元素排序
a.sort(key=lambda x: -x)    # 按负值=降序
sorted(a)               # 返回新列表,不改原列表
b = sorted(a, key=lambda x: (x[0], -x[1]))  # 多关键字:先按x[0]升,再按x[1]降

# 其他
len(a)                  # 长度
a.reverse()             # 原地反转
a[::-1]                 # 返回反转副本(不改原)
sum(a)                  # 求和
max(a); min(a)          # 最大/最小

# 列表推导式(简洁创建)
[x*x for x in range(10)]          # [0,1,4,...,81]
[x for x in range(10) if x%2==0]  # [0,2,4,6,8]
[i for i,j in enumerate(a)]       # 得到所有索引
```

### 4. 字典 (dict) — 哈希表

```python
# 创建
d = {'a': 1, 'b': 2}
d = {}                  # 空字典

# 操作
d['a']                  # 取值, key不存在→KeyError
d.get('a', 0)           # 取值, 不存在→返回0(不报错)
d['c'] = 3              # 插入/修改
d.pop('a')              # 删除并返回值
del d['a']              # 删除(不返回)

# 遍历
for k in d:             # 遍历key
for k, v in d.items():  # 遍历key-value
d.keys()                # 所有key的view
d.values()              # 所有value的view

# 判断
'a' in d                # key是否存在, O(1)

# defaultdict — 自动创建默认值(考试常用!)
from collections import defaultdict
d = defaultdict(list)   # 不存在的key→自动创建空列表[]
d = defaultdict(int)    # 不存在的key→0
d = defaultdict(set)    # 不存在的key→空集合()
d['new_key'].append(x)  # 不需要先初始化,直接append

# Counter — 计数器(统计频率)
from collections import Counter
cnt = Counter([1,2,2,3,3,3])  # Counter({3:3, 2:2, 1:1})
cnt[3]                  # 3, 3出现3次
cnt.most_common(2)      # [(3,3),(2,2)], 出现最多的2个
cnt.update([3,3,4])     # 添加更多元素
```

### 5. 集合 (set)

```python
# 创建
s = {1, 2, 3}           # 直接创建
s = set([1,2,2,3])      # 从列表创建, 自动去重→{1,2,3}
s = set()               # 空集合, 注意{}是空字典不是空集合!

# 操作
s.add(4)                # 添加元素
s.remove(4)             # 删除(不存在→KeyError)
s.discard(4)            # 删除(不存在→不报错,更安全)
4 in s                  # 是否存在, O(1)! 比列表O(n)快

# 集合运算
s1 & s2                 # 交集
s1 | s2                 # 并集
s1 - s2                 # 差集(s1有但s2没有)
s1 ^ s2                 # 异或(只在其中一个中)
len(s)                  # 元素个数

# frozen set — 不可变集合, 可做字典key
fs = frozenset([1,2,3])
```

### 6. 元组 (tuple) — 不可变列表

```python
t = (1, 2, 3)           # 创建, 不可变
t = tuple([1,2,3])      # 从列表转
a, b, c = t             # 解包赋值, a=1,b=2,c=3
t[0]                    # 索引, 同list
len(t)                  # 长度

# 常用技巧: 多变量同时赋值
a, b = b, a             # 交换! 不需要临时变量
x, y = 1, 2             # 同时赋值
# 元组可做字典key(因为不可变), 列表不行!
```

### 7. 常见踩坑

```python
# 浮点精度: 0.1+0.2 != 0.3
0.1 + 0.2 == 0.3        # False!
round(0.1 + 0.2, 10) == 0.3  # True, 用round处理

# 整数除法方向: 向下取整(地板除)
-7 // 2                 # -4, 不是-3! (向下取整)
7 // 2                  # 3
# 如果要数学意义上的取整: int(7/2)=3, int(-7/2)=-3

# list乘法的陷阱
a = [[0]] * 3           # 3个引用指向同一个[0]!
a[0][0] = 1             # a变成[[1],[1],[1]] 全改了!
# 正确方式:
a = [[0] for _ in range(3)]  # 3个独立[0]

# range的范围: 含起不含止
range(5)                 # 0,1,2,3,4 (不含5)
range(1,5)               # 1,2,3,4
range(0,10,2)            # 0,2,4,6,8 (步长2)

# global vs nonlocal
# 函数内修改外层变量: global=模块级, nonlocal=外层函数级
count = 0
def inc():
    global count         # 声明后才能修改
    count += 1

# 递归深度限制
import sys
sys.setrecursionlimit(100000)  # 默认1000, 大数据/深树需要调高

# 常用数学
abs(x)                  # 绝对值
max(a,b); min(a,b)      # 两数比较
pow(x, y)               # x的y次方
x ** y                  # 同pow
pow(x, y, mod)          # x^y mod mod, 快速幂(O(log y))
math.inf                # 正无穷; float('inf')同样
-math.inf               # 负无穷; float('-inf')同样
1e9                     # 10^9, 写成1000000000也行
10**9+7                 # 常用取模数
```

### 8. 枚举与迭代

```python
# enumerate: 同时获取索引和值
for i, val in enumerate(arr):
    # i=索引, val=值
for i, val in enumerate(arr, 1):  # 索引从1开始

# zip: 同时遍历多个序列
for a, b in zip(arr1, arr2):
    # a=arr1[i], b=arr2[i]
list(zip([1,2,3], ['a','b','c']))  # [(1,'a'),(2,'b'),(3,'c')]

# reversed: 反向遍历
for x in reversed(arr):
    # 从末尾到开头

# sorted: 排序后遍历(不改原)
for x in sorted(arr, reverse=True):
    # 从大到小

# any/all: 快速判断
any(x > 0 for x in arr)   # 是否存在>0的元素
all(x > 0 for x in arr)   # 是否全部>0
```

### 9. 异常处理(读取输入时常用)

```python
try:
    n = int(input())
except ValueError:         # 转换失败
    pass
except EOFError:           # 输入结束(多组输入终止信号)
    break

# 最常用: 多组输入直到EOF
while True:
    try:
        line = input()
    except EOFError:
        break
    # 处理line...
```

### 10. 位运算(考试高频)

```python
# 基本操作
x & y          # 按位与: 两位都1才1
x | y          # 按位或: 任一位1就1
x ^ y          # 按位异或: 不同为1, 相同为0
~x             # 按位取反: 包括符号位
x << k         # 左移k位 = x * 2^k
x >> k         # 右移k位 = x // 2^k (对正数)

# 常用技巧
x & -x         # lowbit: 取最低位1的值, 如6&-6=2
x & (x-1)      # 去掉最低位1, 如6&5=4
x & (x-1)==0   # 判断x是否为2的幂(只有一个1)
x >> k & 1     # 取x的第k位(0还是1)
1 << k         # 只有第k位为1的数
x | (1<<k)     # 把x的第k位设为1
x & ~(1<<k)    # 把x的第k位清0

# 状态压缩: 用整数表示集合
mask = 0                    # 空集
mask | (1<<i)               # 加入元素i
mask & (1<<i)               # 查询i是否在集合中(非0=在)
mask ^ (1<<i)               # 删除元素i(或切换)
mask & ~(1<<i)              # 删除元素i(另一种写法)
(1<<n) - 1                  # 包含0..n-1的全集

# 枚举所有子集
for mask in range(1<<n):
    subset = [items[i] for i in range(n) if mask>>i & 1]

# 枚举mask的所有真子集(不含空集不含mask本身)
sub = (mask-1) & mask
while sub:
    # 处理sub
    sub = (sub-1) & mask
```
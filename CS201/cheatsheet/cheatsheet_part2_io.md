# DSA 机考 Cheatsheet — Part 2: I/O 指南 + 输入输出格式

---

## 一、输入方式

### 1. 基础输入 (小数据量, 几十行以内)

```python
n = int(input())                # 读一行,转整数
s = input()                     # 读一行字符串
a, b = map(int, input().split())  # 读一行两个整数
arr = list(map(int, input().split()))  # 读一行转整数列表
# 例: 输入"3 5 7" → arr=[3,5,7]
```

### 2. 快速输入 (大数据量, 必须用!)

```python
import sys
input = sys.stdin.readline      # 替换input函数, 快10倍以上
# 用法和普通input一样, 但注意: readline末尾带'\n'!
n = int(input())                # int()会自动去掉换行符, 没问题
s = input().strip()             # string必须strip去掉换行符!
arr = list(map(int, input().split()))  # split自动去换行,没问题

# 更快: 一次性读全部(超大数据)
data = sys.stdin.read()          # 读全部内容为一个字符串
nums = list(map(int, data.split()))  # 全部数字一次解析
```

### 3. 多组输入模板 (最常见格式)

```python
# 格式A: 给定组数T
import sys; input = sys.stdin.readline
T = int(input())
for _ in range(T):
    n = int(input())
    arr = list(map(int, input().split()))
    # 处理每组...

# 格式B: 不知组数, 读到EOF结束(最常见!)
import sys; input = sys.stdin.readline
while True:
    try:
        n = int(input())
    except EOFError:             # 没更多输入了
        break
    arr = list(map(int, input().split()))
    # 处理每组...

# 格式B变体: 可能有空行
import sys; input = sys.stdin.readline
while True:
    line = input()
    if not line:                 # 空行或EOF
        break
    n = int(line)
    if n == 0:                   # 有些题用n=0作为终止信号
        break
    arr = list(map(int, input().split()))
    # 处理每组...

# 格式C: 每行两个数,读到EOF
import sys; input = sys.stdin.readline
for line in sys.stdin:          # 逐行迭代,自动处理EOF
    a, b = map(int, line.split())
    # 处理...

# 格式D: 图的输入(读n个节点m条边)
import sys; input = sys.stdin.readline
n, m = map(int, input().split())
graph = [[] for _ in range(n+1)]  # 1-indexed邻接表
for _ in range(m):
    u, v, w = map(int, input().split())
    graph[u].append((v, w))      # 有向图
    # graph[v].append((u, w))    # 无向图加这句
```

### 4. 复杂输入格式

```python
# 矩阵输入(每行一行矩阵)
n = int(input())
matrix = []
for _ in range(n):
    row = list(map(int, input().split()))
    matrix.append(row)

# 字符矩阵
n, m = map(int, input().split())
grid = []
for _ in range(n):
    grid.append(input().strip())  # 每行一个字符串, 如"##.##"

# 混合类型输入
line = input().split()           # ['3', 'hello', '5.7']
n = int(line[0])
s = line[1]
f = float(line[2])

# 读整个文件(字符串处理题)
import sys
text = sys.stdin.read()           # 全部内容为一个字符串
```

---

## 二、输出方式

### 1. 基础输出

```python
print(x)                       # 输出x, 自动换行
print(x, y, z)                 # 输出多个值, 用空格分隔
print(x, end=' ')              # 不换行, 用空格结尾(循环中逐个输出)
print(x, end='')               # 不换行, 无结尾字符
print(x, sep=',')              # 多值用逗号分隔而非空格

# 列表输出
print(*arr)                    # 展开列表,空格分隔: 1 2 3 4 5
print(*arr, sep=' ')           # 同上,明确指定分隔符
print('\n'.join(map(str, arr)))  # 每个元素一行

# 格式化输出(重点!)
x = 3.14159265
print(f"{x}")                  # 3.14159265, 原样输出
print(f"{x:.2f}")              # 3.14, 保留2位小数
print(f"{x:.0f}")              # 3, 0位小数=取整
print(f"{x:.6f}")              # 3.141593, 6位小数(四舍五入)
print(f"{x:.10f}")             # 10位小数, 高精度需求

# 整数格式化
n = 42
print(f"{n}")                  # 42
print(f"{n:05d}")              # 00042, 宽度5,前补0
print(f"{n:>5d}")              # "   42", 宽度5,右对齐
print(f"{n:<5d}")              # "42   ", 宽度5,左对齐

# 百分比
print(f"{0.1234:.2%}")         # 12.34%, 自动乘100加%

# 多变量格式化
print(f"{a} {b} {c}")          # 用空格分隔三个变量
print(f"Case {i}: {ans}")      # 常见格式 "Case 1: 42"
```

### 2. 常见输出格式对照

| 题目要求 | Python代码 | 输出效果 |
|---|---|---|
| 输出一个整数 | `print(ans)` | `42` |
| 输出两个整数空格分隔 | `print(a, b)` 或 `print(f"{a} {b}")` | `3 5` |
| 输出保留2位小数 | `print(f"{ans:.2f}")` | `3.14` |
| 输出保留6位小数 | `print(f"{ans:.6f}")` | `3.141593` |
| 输出YES/NO | `print("YES" if cond else "NO")` | `YES` |
| 每行一个数 | `for x in arr: print(x)` | 逐行 |
| 空格分隔一行 | `print(*arr)` | `1 2 3 4 5` |
| Case编号 | `print(f"Case {i}: {ans}")` | `Case 1: 42` |
| Yes/No(首字母大写) | `print("Yes" if ok else "No")` | `Yes` |
| 空行分隔每组 | `print(ans); print()` | 输出后额外空行 |

### 3. 特殊输出技巧

```python
# 逆序输出列表
print(*reversed(arr))          # 5 4 3 2 1

# 输出字典序最小的结果
results.sort()                  # 先排序
print(results[0])              # 取第一个

# 不确定格式时: 模拟样例
# 先仔细看题目样例的输出格式, 包括空格/换行/大小写
# 然后用print精确匹配样例

# 精度陷阱: 整数运算不要转float
# 例: 计算平均值, 用分数表示或最后才转float
avg = sum(arr) / len(arr)       # Python3自动float除法
print(f"{avg:.2f}")             # 保留2位

# 大数取模
MOD = 10**9 + 7
ans = (ans % MOD + MOD) % MOD   # 确保正数(处理负数结果)
print(ans % MOD)                # 正数直接取模就行

# 输出True/False → 需要转为题目要求的格式
# 很多题要求输出 "Yes"/"No" 或 "1"/"0", 不是True/False!
```

---

## 三、OpenJudge 特别注意

### 1. 提交语言选择
- **Python3**: 推荐, 最稳定
- **PyPy3**: 可能更快但兼容性略差, 优先Python3
- **G++**: C++, 如果Python TLE可考虑

### 2. 常见WA原因 (答案错误)

| 原因 | 解决方案 |
|---|---|
| 输出格式不匹配样例 | 逐字符对比你的输出和样例输出(空格/换行/大小写) |
| 多输出了空行 | 检查是否不该print空行 |
| YES写成Yes | 看清楚题目要求的大小写! |
| 浮点精度不够 | 用f-string指定足够位数: `f"{ans:.10f}"` |
| 整数溢出 | Python没有这个问题,放心 |
| 漏处理边界(N=0/N=1) | 特判: if n==0: print(0); return |
| 多组输出间缺少空行 | 有些题要求每组间有空行 |

### 3. 常见TLE原因 (超时)

| 原因 | 解决方案 |
|---|---|
| input()太慢 | 改用 `sys.stdin.readline` |
| list.pop(0) | 用deque代替list做队列操作 |
| 嵌套循环O(n^2)太大 | 想办法优化算法,或用更高效数据结构 |
| 递归太深 | `sys.setrecursionlimit(100000)` |
| Python本身慢 | 考虑换PyPy3或关键部分用更优算法 |

### 4. 常见RE原因 (运行时错误)

| 原因 | 解决方案 |
|---|---|
| 空列表取[0] | 先检查 `if arr:` |
| 字典key不存在 | 用 `d.get(key, default)` 或 `defaultdict` |
| 索引越界 | 检查循环边界, 确认0-based还是1-based |
| 除零 | 检查除数是否为0 |
| 递归深度超限 | `sys.setrecursionlimit(100000)` |

### 5. 完整比赛代码模板

```python
import sys
input = sys.stdin.readline

def solve():
    # 把解题逻辑写在这里
    n = int(input())
    # ...

def main():
    # 处理多组输入
    while True:
        try:
            solve()
        except EOFError:
            break

if __name__ == '__main__':
    main()
```

### 6. 调试技巧

```python
# 本地调试: 打印中间变量
# print(f"debug: n={n}, arr={arr}")  # 提交前删掉所有debug print!

# 快速测试: 在文件末尾加
# if __name__ == '__main__':
#     # 本地测试用例
#     print(solve(测试输入))

# 处理超大输入: 一次性读入
# data = sys.stdin.buffer.read().decode()  # 二进制读取更快
```
# DSA 机考 Cheatsheet — Part3: 数据结构 API (栈/堆/队列/字典)

---

## 一、栈 (Stack) — Python用list模拟

> **核心**: 后进先出(LIFO)。用list的append/pop即可,尾部操作O(1)。

```python
stack = []              # 空栈
stack.append(x)         # 入栈(push), O(1), x放到末尾
stack.pop()             # 出栈(pop), O(1), 弹出末尾元素并返回
stack[-1]               # 看栈顶(peek), 不弹出, O(1)
len(stack)              # 栈大小
if not stack: ...       # 判断栈空

# 示例: 括号匹配
def is_valid(s):
    pairs = {')':'(', ']':'[', '}':'{'}   # 右括号→对应左括号
    stack = []
    for ch in s:
        if ch in pairs:                     # 右括号→检查栈顶
            if not stack or stack[-1] != pairs[ch]:  # 栈空或不匹配
                return False
            stack.pop()                      # 匹配,弹出左括号
        else:                                # 左括号→入栈
            stack.append(ch)
    return not stack                         # 栈空=全部匹配

# 示例: 单调栈(找下一个更大元素)
# LC739 每日温度
def dailyTemperatures(T):
    n = len(T); ans = [0]*n; stack = []      # stack存索引,维持温度递减
    for i in range(n):
        while stack and T[stack[-1]] < T[i]:  # 栈顶温度比当前低→找到了!
            prev = stack.pop()                 # 弹出栈顶
            ans[prev] = i - prev               # 距离=当前位置-栈顶位置
        stack.append(i)                        # 当前入栈,等待更高温度
    return ans
# LC参考: 739(单调栈-下一个更大), 84(柱状图最大矩形), 20(括号匹配)

# 示例: 中缀→后缀(调度场算法)
# 详见Part4, 核心就是栈存操作符,按优先级弹出
```

**栈常见考法一览:**

| 题型 | 识别线索 | LC参考 |
|---|---|---|
| 括号匹配 | 含括号、合法序列 | LC20 |
| 单调栈 | 下一个更大/更小、删数字使最小 | LC739, 84 |
| 表达式求值 | 中缀/后缀表达式 | hw4: 24591 |
| 状态历史 | 浏览器前进后退、撤销 | 题库8号 |
| 字符串处理 | 解码字符串 `3[a2[bc]]` | LC394 |

---

## 二、堆 (Heap) — heapq模块

> **核心**: Python的heapq是小顶堆(最小值在堆顶)。大顶堆=push负数。
> **关键API**: heapify建堆, heappush入堆, heappop出堆, heapreplace替换堆顶。

```python
import heapq

# 小顶堆(最小值在顶部)
hq = []
heapq.heappush(hq, 5)      # 入堆, O(log n)
heapq.heappush(hq, 3)      # 入堆
heapq.heappush(hq, 7)      # 入堆
# 堆内容: [3, 5, 7], 堆顶=最小值

heapq.heappop(hq)           # 弹出最小值, O(log n), 返回3
# 堆内容: [5, 7]

hq[0]                       # 看堆顶(最小值), 不弹出, O(1)

# 从已有列表建堆
arr = [5, 3, 7, 1, 9]
heapq.heapify(arr)          # 原地建堆, O(n), arr变成[1,3,7,5,9]

# 大顶堆技巧: push/pop时取负数
hq_max = []
heapq.heappush(hq_max, -5)  # 存-5
heapq.heappush(hq_max, -3)  # 存-3
heapq.heappush(hq_max, -7)  # 存-7
# 堆内容: [-7,-5,-3], 堆顶=-7 → 原值7是最大值
result = -heapq.heappop(hq_max)  # 弹出-7, 取负=7(最大值)

# heapreplace: 先弹出堆顶再push新值(比pop+push快一步)
heapq.heapreplace(hq, 2)    # 弹出原堆顶, 插入2, 返回原堆顶值

# 哈夫曼编码(贪心+堆)
# LC参考: 无直接对应, 但hw8: 22161哈夫曼树, hw520: 04080
# 核心: 每次pop两个最小,合并后push回去
import heapq
def huffman_wpl(freqs):       # freqs={字符:频率}
    hq = list(freqs.items())   # [(频率,字符),...]
    heapq.heapify(hq)
    total = 0                  # WPL累加
    while len(hq) > 1:
        f1, a = heapq.heappop(hq)  # 最小频率
        f2, b = heapq.heappop(hq)  # 次小频率
        total += f1 + f2           # 合并的内部节点贡献f1+f2到WPL
        heapq.heappush(hq, (f1+f2, a+b))  # 合并节点push回堆
    return total
# 注意: WPL = 所有内部节点频率之和 = 上述total
# 另一种计算: 叶子频率×深度, 需要记录深度

# Top-K问题(维护大小为K的小顶堆)
# LC215 第K大元素, LC347 前K高频
def top_k_frequent(nums, k):
    from collections import Counter; import heapq
    freq = Counter(nums)       # {元素:频率}
    hq = []                    # 小顶堆,维护前K个最高频
    for num, f in freq.items():
        heapq.heappush(hq, (f, num))  # (频率,元素)
        if len(hq) > k:              # 堆超过K→弹出最低频
            heapq.heappop(hq)
    return [num for f, num in hq]    # 堆中就是前K高频

# 双堆找中位数(大顶堆存小半+小顶堆存大半)
# LC295 数据流中位数
class MedianFinder:
    def __init__(self):
        self.small = []   # 大顶堆(存较小半), 用负数模拟
        self.large = []   # 小顶堆(存较大半), 正常heapq
    def addNum(self, num):
        heapq.heappush(self.small, -num)     # 先进small(取负)
        heapq.heappush(self.large, -heapq.heappop(self.small))  # small最大→large
        if len(self.large) > len(self.small):  # 平衡:small多一个或相等
            heapq.heappush(self.small, -heapq.heappop(self.large))
    def findMedian(self):
        if len(self.small) > len(self.large):  # 奇数个→small堆顶
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2  # 偶数→两堆顶平均
```

**堆常见考法一览:**

| 题型 | 识别线索 | LC参考 |
|---|---|---|
| Top-K | 第K大/小、前K高频 | LC215, 347 |
| 哈夫曼 | 编码、WPL最小 | hw: 04080, 22161 |
| 数据流中位数 | 动态维护中位数 | LC295 |
| 优先队列 | 带优先级的调度 | 题库: 堆 |
| Kruskal辅助 | MST中排序边 | — |

---

## 三、队列 (Queue) — collections.deque

> **核心**: 双端队列, 两端都O(1)操作。比list.pop(0)快得多(list头部操作O(n))!
> **关键API**: append右端加, appendleft左端加, pop右端删, popleft左端删。

```python
from collections import deque

q = deque()              # 空队列
q.append(x)              # 右端入队(标准队列尾部), O(1)
q.appendleft(x)          # 左端入队(双端队列特有), O(1)
q.pop()                  # 右端出队, O(1), 返回右端元素
q.popleft()              # 左端出队(标准队列头部), O(1), 返回左端元素
q[0]                     # 看队头, 不弹出, O(1)
q[-1]                    # 看队尾, 不弹出, O(1)
len(q)                   # 队列大小
if not q: ...            # 判断队空

# 标准队列(FIFO): append入队, popleft出队
q = deque([1,2,3])
q.append(4)              # 入队: [1,2,3,4]
q.popleft()              # 出队返回1: [2,3,4]

# BFS标准模板(无权最短路)
# LC200 岛屿数量, LC994 腐烂橘子
from collections import deque
def bfs_grid(grid, start):
    R, C = len(grid), len(grid[0])
    visited = set()
    q = deque([(start[0], start[1], 0)])  # (行,列,步数)
    visited.add(start)
    while q:
        r, c, d = q.popleft()              # 出队(先进先出=逐层)
        if 目标条件: return d               # 首次到达=最短
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited:
                visited.add((nr,nc))        # 入队即标记!不是出队时
                q.append((nr,nc,d+1))
    return -1

# 多源BFS(所有起点同时入队)
# LC994 腐烂橘子: 所有烂橘子同时开始扩散
q = deque()
for i in range(R):
    for j in range(C):
        if grid[i][j]==2: q.append((i,j,0))  # 所有起点入队
# 然后正常BFS即可,逐层扩散=所有源同步扩展

# 单调队列(滑动窗口最大值)
# LC239 滑动窗口最大值
def sliding_max(nums, k):
    q = deque()              # 存索引, 维持值递减(队头最大)
    res = []
    for i in range(len(nums)):
        while q and nums[q[-1]] <= nums[i]:  # 队尾比当前小→删(没用了)
            q.pop()
        q.append(i)                            # 当前入队尾
        if q[0] <= i-k: q.popleft()             # 队头超出窗口→删
        if i >= k-1: res.append(nums[q[0]])      # 队头=窗口最大值
    return res
```

**队列常见考法一览:**

| 题型 | 识别线索 | LC参考 |
|---|---|---|
| BFS最短路 | 最少步数、最短距离、无权图 | LC200, 994 |
| 多源BFS | 多起点同时扩散 | LC994 |
| 单调队列 | 滑动窗口最大/最小值 | LC239 |
| 层序遍历 | 按层输出二叉树 | LC102 |
| 拓扑排序 | 入度0入队,Kahn算法 | LC207 |
| 约瑟夫问题 | 报数出圈 | — |

---

## 四、并查集 (DSU) — 手写class

> **核心**: 维护分组。find找根判同组, union合并两棵树。路径压缩+按大小合并→均摊O(1)。
> **关键**: 考试时直接抄这个class,然后只用find和union两个方法。

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n+1))  # 每个节点初始指向自己(是自己的根)
        self.size = [1]*(n+1)           # 树的大小,用于按大小合并

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩:跳一层
            x = self.parent[x]
        return x                         # 返回根节点

    def union(self, a, b):
        a, b = self.find(a), self.find(b)  # 先找各自的根
        if a == b: return                   # 已在同一组,不用合并
        if self.size[a] < self.size[b]:     # 小树挂大树(保持树浅)
            a, b = b, a
        self.parent[b] = a                  # b的根改为a
        self.size[a] += self.size[b]

# 使用示例1: 判断连通性
dsu = DSU(n)
for u, v in edges:
    dsu.union(u, v)
# 判断两点是否连通: dsu.find(a) == dsu.find(b)

# 使用示例2: Kruskal MST
# 详见Part5, 核心就是边排序+DSU判断两端是否已连通

# 使用示例3: 统计连通分量数
dsu = DSU(n)
for u, v in edges:
    dsu.union(u, v)
groups = sum(1 for i in range(1,n+1) if dsu.find(i)==i)  # 根的数量=组数
# LC参考: LC参考不多, 但hw B: 07734虫子的生活, hw D: 猫猫搭积木
```

---

## 五、defaultdict 和 Counter — 考试高频工具

```python
from collections import defaultdict, Counter

# defaultdict: 自动创建默认值,省去初始化麻烦
d = defaultdict(list)      # 不存在的key→自动创建[]
d['new'].append(1)          # 直接append, 不需要先d['new']=[]
d = defaultdict(int)        # 不存在的key→0, 做计数器: d[k]+=1
d = defaultdict(set)        # 不存在的key→空集合()

# 计数器Counter: 统计频率
cnt = Counter(['a','b','a','c','a','b'])  # Counter({'a':3,'b':2,'c':1})
cnt['a']                    # 3, a出现3次
cnt.most_common(2)          # [('a',3),('b',2)], 最常见的2个
cnt.update(['a','d'])        # 添加更多元素

# 实用: 用defaultdict(int)做频率计数
freq = defaultdict(int)
for x in arr:
    freq[x] += 1             # 不需要先判断x是否在字典中
```

---

## 六、TreeNode 定义 — 二叉树基础

```python
# 标准二叉树节点(考试通用定义)
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val          # 节点值
        self.left = left        # 左子树
        self.right = right      # 右子树

# 构建二叉树: 从列表层序构建(常见格式)
# 输入: [1,2,3,None,None,4,5] → 完整二叉树
def build_tree(vals):
    if not vals: return None
    root = TreeNode(vals[0])
    q = deque([root])
    i = 1
    while q and i < len(vals):
        node = q.popleft()
        if i < len(vals) and vals[i] is not None:
            node.left = TreeNode(vals[i])
            q.append(node.left)
        i += 1
        if i < len(vals) and vals[i] is not None:
            node.right = TreeNode(vals[i])
            q.append(node.right)
        i += 1
    return root

# 前序+中序建树(考试高频!)
# LC105, hw7: 22158
def build_from_pre_in(preorder, inorder):
    if not preorder: return None
    root_val = preorder[0]                  # 前序第一个=根
    root = TreeNode(root_val)
    idx = inorder.index(root_val)            # 中序中找根的位置
    root.left = build_from_pre_in(preorder[1:1+idx], inorder[:idx])    # idx个左子树元素
    root.right = build_from_pre_in(preorder[1+idx:], inorder[idx+1:])  # 其余右子树
    return root

# BST节点(左小右大)
class BSTNode:
    def __init__(self, val):
        self.val = val; self.left = None; self.right = None

def bst_insert(root, val):
    if not root: return BSTNode(val)        # 找到空位→插入
    if val < root.val:
        root.left = bst_insert(root.left, val)   # 小→左
    else:
        root.right = bst_insert(root.right, val) # 大或等→右
    return root
# 构建BST: 逐个插入
root = None
for val in vals:
    root = bst_insert(root, val)
# LC参考: LC230(BST第k小), hw D: 累加树(右根左遍历累加)
```

---

## 七、Trie (前缀树) — LC208

```python
class TrieNode:
    def __init__(self):
        self.children = {}       # 字符→子节点字典
        self.is_end = False      # 是否是单词结尾

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for ch in word:                          # 逐字符走
            if ch not in node.children:
                node.children[ch] = TrieNode()    # 新字符→建节点
            node = node.children[ch]              # 跳到子节点
        node.is_end = True                        # 标记单词结尾

    def search(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children: return False
            node = node.children[ch]
        return node.is_end                        # 必须是完整单词

    def startsWith(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children: return False
            node = node.children[ch]
        return True                               # 只要前缀匹配就行
# LC参考: LC208(实现Trie), hw8: 实现Trie
```
# DSA 机考 Cheatsheet — Part5: 题库类二+三 — 贪心/排序/二分(4题) + 图论(5题)

> 对应30题题库 **Category 2** (贪心/排序/二分4题) 和 **Category 3** (图论5题)

---

## Category 2: 贪心、排序与二分 (4题)

### 1. 排序 — 基础难度

**题目特征**: 排序后简单操作、合并区间、按特定规则排序

**模板题**: LC56 Merge Intervals, LC49 Group Anagrams, hwA E02724生日相同

**核心代码 — LC56 Merge Intervals (排序+合并)**:
```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])   # 按起点排序(贪心前提!)
    res = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= res[-1][1]:               # 当前起点<=上一个终点→重叠,合并
            res[-1][1] = max(res[-1][1], e)  # 更新终点为较大值
        else:
            res.append([s, e])            # 不重叠→新区间
    return res
```

**注意**:
- 排序是贪心/合并类题的第一步,不排序就没法贪心
- `sort(key=lambda x: (x[0], -x[1]))` 多关键字排序: 先按x[0]升序,x[0]相同再按x[1]降序

---

### 2. 贪心算法 — 中等难度

**题目特征**: 局部最优→全局最优、跳跃游戏、最优划分、买卖股票最佳时机

**模板题**: LC55 Jump Game, LC45 Jump Game II, LC121 Best Time to Buy and Sell Stock, LC763 Partition Labels

**核心代码 — LC55 Jump Game (贪心:最远可达)**:
```python
def canJump(nums):
    max_reach = 0                         # 当前能到达的最远位置
    for i in range(len(nums)):
        if i > max_reach: return False    # 当前位置超过了最远可达→到不了这里
        max_reach = max(max_reach, i+nums[i])  # 从当前位置能跳到i+nums[i]
    return True                            # 遍历完没被卡住→能到终点
```

**核心代码 — LC121 买卖股票最佳时机 (贪心:记录最低价)**:
```python
def maxProfit(prices):
    min_price = float('inf'); ans = 0
    for p in prices:
        min_price = min(min_price, p)      # 记录到目前为止最低价
        ans = max(ans, p - min_price)       # 当前价-最低价=最大利润
    return ans
```

**核心代码 — LC763 Partition Labels (贪心:记录最后出现位置)**:
```python
def partitionLabels(s):
    last = {ch: i for i, ch in enumerate(s)}  # 每个字符最后出现的位置
    start = 0; end = 0; res = []
    for i, ch in enumerate(s):
        end = max(end, last[ch])                # 当前分段的end=段内所有字符最后位置的最大值
        if i == end:                            # 到达end→分段结束
            res.append(i-start+1); start = i+1
    return res
```

**注意**:
- 贪心核心: 找到"局部最优就能推导全局最优"的证据(跳跃游戏=每步尽量远,买卖股票=尽量低价买)
- LC763的关键: 一个分段必须包含所有字符最后一次出现的位置,所以end要取max

---

### 3. 堆 — 困难难度

**题目特征**: 第K大/小元素、前K高频、数据流中位数、哈夫曼编码

**模板题**: LC215 Kth Largest, LC347 Top K Frequent, hw520: Huffman编码树

**核心代码 — LC215 第K大元素 (小顶堆维护K个最大)**:
```python
def findKthLargest(nums, k):
    import heapq
    heap = nums[:k]; heapq.heapify(heap)    # 堆化前k个(小顶堆,堆顶=第k大)
    for num in nums[k:]:
        if num > heap[0]:                   # 比堆顶大→替换堆顶(维护前k个最大的)
            heapq.heapreplace(heap, num)    # pop堆顶+push新值(一步完成)
    return heap[0]                          # 堆顶=第k大(比它大的有k-1个在堆里)
```

**核心代码 — 哈夫曼编码 (贪心+堆)**:
```python
import heapq
def huffman(freqs):                       # freqs={字符:频率}
    hq = [(f, i, ch) for i, (ch, f) in enumerate(freqs.items())]
    heapq.heapify(hq)                      # (频率, 序号防比较冲突, 字符/子树)
    while len(hq) > 1:
        f1, _, a = heapq.heappop(hq)       # 最小频率节点
        f2, _, b = heapq.heappop(hq)       # 次小频率节点
        heapq.heappush(hq, (f1+f2, i, (a,b)))  # 合并后push回堆
        i += 1                              # 序号递增,防止节点比较时报错
    return hq[0][2]                         # 返回树结构(用于计算WPL)

# WPL(带权路径长度)计算: 方法1=所有内部节点频率之和
def huffman_wpl(freqs):
    hq = [(f, ch) for ch, f in freqs.items()]
    heapq.heapify(hq)
    total = 0                               # WPL=所有内部节点权之和
    while len(hq) > 1:
        f1, a = heapq.heappop(hq)
        f2, b = heapq.heappop(hq)
        total += f1 + f2                    # 合并节点贡献f1+f2
        heapq.heappush(hq, (f1+f2, (a,b)))
    return total                             # total就是WPL!
# hw参考: hw520 E04080, hw8 T22161
```

**注意**:
- 堆存元组时如果第一个元素相同,Python会比较第二个→第二个元素必须可比较!
- 哈夫曼WPL两种算法: (1)内部节点权之和 (2)叶子频率×深度。方法1更简单
- 堆中加序号`i`防止字符/子树比较报错(TypeError)

---

### 4. 二分 — 困难难度

**题目特征**: 搜索满足条件的值、最小化/最大化答案、旋转数组搜索、找边界

**模板题**: LC33 Search in Rotated Sorted Array, LC34 Find First and Last, hw5 M02774木材加工

**核心代码 — LC33 旋转排序数组搜索 (判断有序段)**:
```python
def search(nums, target):
    lo, hi = 0, len(nums)-1
    while lo <= hi:
        mid = (lo+hi)//2
        if nums[mid] == target: return mid
        if nums[lo] <= nums[mid]:          # 左半段有序(lo到mid没有旋转断点)
            if nums[lo] <= target < nums[mid]:  # target在有序左半段
                hi = mid-1
            else:
                lo = mid+1                 # target在右半段
        else:                              # 右半段有序(mid到hi没有旋转断点)
            if nums[mid] < target <= nums[hi]:  # target在有序右半段
                lo = mid+1
            else:
                hi = mid-1                 # target在左半段
    return -1
```

**核心代码 — 二分答案 (考试最常用)**:
```python
# 最小化答案: 找满足check的最小值
def bsearch_min(low, high, check):
    lo, hi = low, high
    while lo < hi:
        mid = (lo+hi)//2
        if check(mid): hi = mid            # mid可行→试试更小
        else: lo = mid+1                   # mid不行→必须更大
    return lo

# 示例: 木材加工(hw5 M02774) — 切k段等长木料,最大化段长
def check(length):
    return sum(wood // length for wood in woods) >= k  # 能切出>=k段
ans = bsearch_min(1, max(woods), check)    # 找最小满足的length... 不对,这是最大化!
# 最大化要用bsearch_max:
def bsearch_max(low, high, check):
    lo, hi = low, high
    while lo < hi:
        mid = (lo+hi+1)//2                 # 向上取整!
        if check(mid): lo = mid            # mid可行→试试更大
        else: hi = mid-1
    return lo
ans = bsearch_max(1, max(woods), check)    # 最大可行段长
# hw参考: hw5 M02774, hwA M20746
```

**注意**:
- 二分答案核心: 确定check含义,最小化用`(lo+hi)//2`,最大化用`(lo+hi+1)//2`
- LC33关键: 判断mid左半段还是右半段有序,然后决定target在哪边

---

## Category 3: 图论与拓扑结构 (5题)

### 5. DSU并查集 — 中等难度

**题目特征**: 连通性判断、分组、合并集合、判断是否属于同一组、MST辅助

**模板题**: hwB 07734虫子生活, hwD 猫猫搭积木, hw4 01611嫌疑人, hwA 07734

**核心代码 — DSU完整class (直接抄)**:
```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n+1))     # 初始每个节点指向自己
        self.size = [1]*(n+1)              # 树的大小,按大小合并用

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩:跳一层
            x = self.parent[x]
        return x                            # 返回根节点(代表整个组)

    def union(self, a, b):
        a, b = self.find(a), self.find(b)   # 找各自的根
        if a == b: return                   # 已同组
        if self.size[a] < self.size[b]: a, b = b, a  # 小树挂大树
        self.parent[b] = a                  # b的根改为a
        self.size[a] += self.size[b]

    def count_groups(self, n):              # 统计连通分量数
        return sum(1 for i in range(1,n+1) if self.find(i)==i)

# 使用: 连通性判断
dsu = DSU(n)
for u, v in edges: dsu.union(u, v)
same_group = dsu.find(a) == dsu.find(b)    # True=同组/连通

# 使用: 统计组数
groups = dsu.count_groups(n)                # 根的数量=连通分量数
```

**注意**:
- DSU两大优化: 路径压缩(find时把中间节点直接连根) + 按大小合并(小挂大保持树浅)
- `self.parent[x] = self.parent[self.parent[x]]`是迭代版路径压缩,比递归版更稳定
- 判断连通: find(a)==find(b); 合并: union(a,b); 不需要其他操作

---

### 6. 树/DFS — 中等难度

**题目特征**: 树的遍历、建树、LCA、树直径/深度、路径和、子树统计

**模板题**: LC94/102/104/226/236/105/108, hw6全部, hw7全部

**核心代码 — 二叉树DFS模板 (前/中/后序)**:
```python
# 递归遍历 — 操作位置决定遍历类型
def dfs(root):
    if not root: return
    # 前序位置: 此时刚进入root,先处理root再处理子树
    dfs(root.left)
    # 中序位置: 左子树已处理完,现在处理root,再处理右子树(BST中序=升序!)
    dfs(root.right)
    # 后序位置: 两子树都处理完,最后处理root(适合需要子树结果的场景:深度/路径和/LCA)

# LC104 最大深度 (后序:需要子树深度)
def maxDepth(root):
    if not root: return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))  # 子树深度+1

# LC226 翻转二叉树 (前序或后序都行)
def invertTree(root):
    if not root: return None
    root.left, root.right = root.right, root.left  # 交换左右子树
    invertTree(root.left); invertTree(root.right)
    return root

# LC236 LCA最近公共祖先 (后序:需要知道子树里有没有p/q)
def lca(root, p, q):
    if not root or root==p or root==q: return root  # 空或找到目标→返回
    left = lca(root.left, p, q)     # 左子树找p/q
    right = lca(root.right, p, q)   # 右子树找p/q
    if left and right: return root  # 两边都找到→当前节点就是LCA
    return left or right            # 只一边找到→LCA在那边;都没→None

# 前序+中序建树 (LC105, hw7 22158)
def build(preorder, inorder):
    if not preorder: return None
    root = TreeNode(preorder[0])                # 前序第一个=根
    idx = inorder.index(preorder[0])            # 中序找根位置→分左右
    root.left = build(preorder[1:1+idx], inorder[:idx])    # 左子树
    root.right = build(preorder[1+idx:], inorder[idx+1:])  # 右子树
    return root
```

**核心代码 — BST操作**:
```python
# BST插入 (hwD: 累加树)
def bst_insert(root, val):
    if not root: return BSTNode(val)     # 空位→插入
    if val < root.val: root.left = bst_insert(root.left, val)   # 小→左
    else: root.right = bst_insert(root.right, val)              # 大→右
    return root

# BST累加 (右-根-左遍历, hwD: 累加树)
# 核心: BST右-根-左遍历=降序,从最大值开始累加
acc = 0
def bst_accumulate(root):
    global acc
    if not root: return
    bst_accumulate(root.right)           # 先右(最大值)
    acc += root.val; root.val = acc      # 累加并赋值
    bst_accumulate(root.left)            # 再左

# BST第k小 (LC230, 中序遍历数到第k个)
def kth_smallest(root, k):
    stack = []; node = root
    while stack or node:
        while node: stack.append(node); node = node.left  # 一路往左
        node = stack.pop(); k -= 1        # 弹出=中序访问(升序)
        if k == 0: return node.val
        node = node.right
```

**注意**:
- 二叉树万能公式: 递归+选择操作位置(前/中/后序)
- 后序最常用: 需要子树结果才能算当前节点(LCA/深度/路径和/树DP)
- BST核心性质: 中序=升序, 右-根-左=降序

---

### 7. 拓扑排序 — 中等难度

**题目特征**: 有前置依赖、能否完成所有课程、有向图判环、输出合法顺序

**模板题**: LC207 Course Schedule, hw9 sy382有向图判环, hw 09202舰队出击

**核心代码 — Kahn算法 (BFS版, 入度驱动)**:
```python
from collections import deque
def topo_sort(graph, n):
    indeg = [0]*(n+1)                         # 入度计数
    for u in range(1,n+1):
        for v in graph[u]: indeg[v] += 1      # v被u指向→v入度+1
    q = deque([u for u in range(1,n+1) if indeg[u]==0])  # 入度0的先出
    order = []
    while q:
        u = q.popleft(); order.append(u)      # "删掉"u(它没有前置了)
        for v in graph[u]:
            indeg[v] -= 1                     # u删了→v的前置少了1
            if indeg[v] == 0: q.append(v)     # v的前置全删完→v可以出
    if len(order) == n: return order           # 全删完=无环=成功
    return []                                  # 有环(有些节点入度永远不为0)

# LC207 判断能否完成课程
def canFinish(numCourses, prerequisites):
    indeg = [0]*numCourses; graph = [[] for _ in range(numCourses)]
    for u, v in prerequisites:
        graph[v].append(u); indeg[u] += 1     # v→u:要先学v才能学u
    q = deque([i for i in range(numCourses) if indeg[i]==0])
    cnt = 0
    while q:
        node = q.popleft(); cnt += 1
        for nei in graph[node]:
            indeg[nei] -= 1
            if indeg[nei] == 0: q.append(nei)
    return cnt == numCourses                   # 全删完=无环=可以完成
```

**注意**:
- 拓扑排序两种用途: (1)输出合法顺序 (2)判断有没有环
- Kahn算法核心: 入度0的节点没有依赖→可以先处理;处理后减少邻居入度
- `len(order)==n`判无环, `<n`判有环

---

### 8. SCC强连通分量 (Tarjan) — 困难难度

**题目特征**: 缩点、判断节点是否可达所有节点、最少广播站数、间谍网络

**模板题**: hwB 02186 Popular Cows, hwB 01236 Network of Schools, hw 0516/0517 SCC题

**核心代码 — Tarjan SCC (单次DFS+栈)**:
```python
def tarjan(graph, n):
    dfn = [0]*(n+1)   # 发现时间戳(DFS访问顺序)
    low = [0]*(n+1)   # 能回溯到的最早祖先时间戳
    on_stack = [False]*(n+1); stack = []; sccs = []; timer = 0

    def dfs(u):
        nonlocal timer; timer += 1
        dfn[u] = low[u] = timer             # 初始化: low=自己的时间戳
        stack.append(u); on_stack[u] = True  # 入栈追踪当前搜索路径
        for v in graph[u]:
            if dfn[v] == 0:                  # v未访问→递归深入
                dfs(v)
                low[u] = min(low[u], low[v]) # 子节点能回溯→我也能回溯
            elif on_stack[v]:                # v在栈中→是祖先(或同SCC),可回溯
                low[u] = min(low[u], dfn[v]) # 注意取dfn而非low!
        if low[u] == dfn[u]:                 # low==dfn→u是SCC根
            comp = []
            while True:
                w = stack.pop(); on_stack[w] = False; comp.append(w)
                if w == u: break              # 弹到u为止=一个完整SCC
            sccs.append(comp)

    for u in range(1,n+1):
        if dfn[u] == 0: dfs(u)
    return sccs                              # sccs=[每SCC的节点列表]
```

**核心代码 — Kosaraju SCC (两次DFS, 更简单)**:
```python
def kosaraju(graph, rg, n):                # graph=原图, rg=反图
    visited = set(); order = []
    def dfs1(u):                           # 第1次DFS: 原图,记录后序
        visited.add(u)
        for v in graph[u]:
            if v not in visited: dfs1(v)
        order.append(u)                    # 后序:递归返回后才记录
    for u in range(1,n+1):
        if u not in visited: dfs1(u)

    visited2 = set(); sccs = []
    def dfs2(u, comp):                     # 第2次DFS: 反图,按后序逆序出发
        visited2.add(u); comp.append(u)
        for v in rg[u]:
            if v not in visited2: dfs2(v, comp)
    for u in reversed(order):              # 后序逆序=反图SCC入口
        if u not in visited2:
            comp = []; dfs2(u, comp); sccs.append(comp)
    return sccs

# 反图构建: 对每条边u→v, 同时建rg[v].append(u)
rg = [[] for _ in range(n+1)]
for u in range(1,n+1):
    for v in graph[u]: rg[v].append(u)
```

**SCC应用 — 缩点+DAG**:
```python
# 缩点后形成DAG(无环有向图),可以在DAG上做拓扑排序/DP
sccs = tarjan(graph, n)
scc_id = {}                                # 原节点→所属SCC编号
for i, comp in enumerate(sccs):
    for node in comp: scc_id[node] = i

# 构建缩点后的DAG
dag = [[] for _ in range(len(sccs))]
for u in range(1,n+1):
    for v in graph[u]:
        if scc_id[u] != scc_id[v]:         # 不同SCC→DAG中有边
            dag[scc_id[u]].append(scc_id[v])
# hw参考: hwB 02186, 01236, 题库SCC+拓扑+DP题
```

**注意**:
- Tarjan核心: `low[u]==dfn[u]`时u是SCC根,栈中u之后弹出的都是同一SCC
- Kosaraju更易理解: 原图后序逆序→反图DFS入口,每次DFS挖一个SCC
- `elif on_stack[v]`用`dfn[v]`而非`low[v]`: Tarjan原始论文定义,用low在某些图上有bug
- SCC缩点后变DAG,可在DAG上拓扑排序→解决依赖问题

---

### 9. SCC+拓扑排序+DP — 困难难度

**题目特征**: 缩点后在DAG上做最长路/最短路/计数、间谍网络、软件安装

**模板题**: 题库SCC+拓扑+DP题, hw 0524 P2515软件安装(SCC+树DP)

**解题流程**:
```
原图 → Tarjan/Kosaraju找SCC → 缩点成DAG → 拓扑排序 → 在DAG上DP
```

**核心代码 — SCC缩点+DAG上DP**:
```python
# 步骤1: 找SCC
sccs = tarjan(graph, n)

# 步骤2: 缩点→DAG (见上方缩点代码)

# 步骤3: DAG拓扑排序
order = topo_sort(dag, len(sccs))          # 在缩点DAG上拓扑排序

# 步骤4: DAG上DP (最长路/最短路/最值)
dp = [初始值]*len(sccs)
for u in order:                            # 按拓扑序遍历DAG节点
    for v in dag[u]:                        # u的所有后继SCC
        dp[v] = 更新(dp[v], dp[u]+...)      # 状态转移
```

**注意**:
- SCC+拓扑+DP是图论最难的组合,但步骤是固定的: 缩点→拓扑→DP
- 缩点后DAG保证无环,拓扑排序合法,DP顺序正确
- 如果缩点后DAG只有一条链→其实就是拓扑序上的线性DP
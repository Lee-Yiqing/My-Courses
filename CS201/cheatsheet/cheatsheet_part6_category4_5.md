# DSA 机考 Cheatsheet — Part6: 题库类四+五 — DP+状态压缩(5题) + 高级数据结构(5题)

> 对应30题题库 **Category 4** (DP+状态压缩5题) 和 **Category 5** (高级数据结构5题)

---

## Category 4: 动态规划与状态压缩 (5题)

### 1. 线性DP — 基础难度

**题目特征**: 最长递增子序列(LIS)、爬楼梯、打家劫舍、最大子数组、背包问题

**模板题**: LC70 Climbing Stairs, LC198 House Robber, LC53 Max Subarray, LC300 LIS, LC322 Coin Change

**核心代码 — LC198 打家劫舍 (线性DP, 选或不选)**:
```python
def rob(nums):
    a = b = 0                    # a=dp[i-2](隔两个的最大), b=dp[i-1](隔一个的最大)
    for x in nums:
        a, b = b, max(b, a+x)   # 选当前: a+x(隔一个+当前); 不选: b(维持上一个)
    return b                      # b=最终最大值
# 状态转移: dp[i]=max(dp[i-1], dp[i-2]+nums[i]), 即选或不选当前
```

**核心代码 — LC53 最大子数组 (Kadane算法)**:
```python
def maxSubArray(nums):
    cur = best = nums[0]          # cur=当前连续子数组和, best=全局最大
    for num in nums[1:]:
        cur = max(num, cur+num)    # 要么从num重新开始,要么延续cur+num
        best = max(best, cur)      # 更新全局最大
    return best
# 核心: cur<0时,延续只会让后面的更小→不如从当前重新开始
```

**核心代码 — LC300 最长递增子序列 (DP+二分优化)**:
```python
# O(n^2) DP版: dp[i]=以nums[i]结尾的LIS长度
def lengthOfLIS(nums):
    n = len(nums); dp = [1]*n      # dp[i]至少=1(自己)
    for i in range(n):
        for j in range(i):         # 遍历i之前所有元素
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j]+1)  # nums[j]能接到nums[i]前→长度+1
    return max(dp)

# O(nlogn) 二分优化版: 维持tails数组(贪心+二分)
def lengthOfLIS_fast(nums):
    tails = []                      # tails[k]=长度为k+1的LIS的最小末尾元素
    for num in nums:
        idx = bisect.bisect_left(tails, num)  # 在tails中找num的位置
        if idx == len(tails): tails.append(num)  # num比所有tails大→LIS长度增加
        else: tails[idx] = num      # 替换:让tails尽量小(贪心,末尾小更有潜力)
    return len(tails)
```

**核心代码 — LC322 Coin Change (完全背包DP)**:
```python
def coinChange(coins, amount):
    dp = [float('inf')]*(amount+1)   # dp[i]=凑出金额i的最少硬币数
    dp[0] = 0                         # 金额0→0个硬币
    for i in range(1, amount+1):
        for c in coins:
            if i >= c:                # 当前金额>=硬币面值→可以尝试用这枚硬币
                dp[i] = min(dp[i], dp[i-c]+1)  # 用这枚硬币:dp[i-c]+1
    return dp[amount] if dp[amount]!=float('inf') else -1  # 不可能凑出→-1
```

**核心代码 — LC416 Partition Equal Subset Sum (0-1背包)**:
```python
def canPartition(nums):
    total = sum(nums)
    if total % 2: return False        # 奇数总和→不可能分成两等份
    target = total // 2
    dp = [False]*(target+1); dp[0] = True   # dp[j]=能否凑出j
    for num in nums:
        for j in range(target, num-1, -1):   # 逆序遍历!0-1背包每物品只用一次
            dp[j] = dp[j] or dp[j-num]        # 不选num(dp[j])或选num(dp[j-num])
    return dp[target]
# 注意: 0-1背包逆序遍历j(每物品最多用1次); 完全背包正序(可重复用)
```

**注意**:
- DP核心: 定义状态→写转移方程→确定遍历顺序→初始化
- 爬楼梯/打家劫舍: 滚动变量优化(只用前两个值,不需要整个数组)
- 0-1背包j逆序(每物品用一次); 完全背包j正序(物品可重复); LC322是完全背包
- LIS贪心版: tails数组越小越好(末尾小→后面更容易接上更长的)

---

### 2. DP+滑动窗口 — 中等难度

**题目特征**: DP状态转移需要从窗口内取最优值(用单调队列优化O(n)→O(1)取窗口max)

**模板题**: LC1871 Jump Game VII (dp+sliding window), 题库DP+滑动窗口题

**核心代码 — DP+单调队列优化**:
```python
# 朴素DP: dp[i]=能否到达i, 对每个i检查窗口dp[i-min..i-max]中是否有True
# O(n*窗口大小) → 用滑动窗口维护窗口内True的数量 → O(n)

from collections import deque
def dp_sliding_window(nums, min_jump, max_jump):
    n = len(nums)
    dp = [False]*n; dp[0] = True         # 起点可达
    pre = [0]*(n+1)                       # pre[i]=dp[0..i-1]中True的数量(前缀和优化)
    pre[1] = 1
    for i in range(1, n):
        if nums[i] == '1': dp[i] = False  # 不能跳到障碍
        else:
            l = i - max_jump; r = i - min_jump  # 窗口范围[l..r]
            l = max(l, 0)
            if pre[r+1] - pre[l] > 0:     # 窗口内有True→i可达
                dp[i] = True
        pre[i+1] = pre[i] + (1 if dp[i] else 0)
    return dp[n-1]
# LC参考: LC1871(跳跃游戏VII, dp+前缀和优化窗口查询)
```

**注意**:
- DP+滑动窗口优化: 朴素转移O(n×window), 用前缀和/单调队列降到O(n)
- 前缀和`pre[r+1]-pre[l]`快速判断窗口内是否有True(>0就有)
- 单调队列版本: 适合需要取窗口max/min而不是只判断有无的情况

---

### 3. DFS序/树背包DP — 中等难度

**题目特征**: 树上选节点(选/不选约束)、树上背包(选K个节点使总价值最大)、没有上司的舞会

**模板题**: hw7 宝藏二叉树, hw 0413 P1352没有上司的舞会, 题库树背包DP

**核心代码 — 树DP (选或不选)**:
```python
# 没有上司的舞会: 选了u→子节点不能选; 不选u→子节点随意
# LC参考: 无直接对应, hw: 24637宝藏二叉树, P1352没有上司的舞会
def tree_dp(root, children, values):
    dp = {}                              # dp[u]=[不选u的最大值, 选u的最大值]
    def dfs(u):
        dp[u] = [0, values[u]]           # 初始化: 不选=0, 选=自身价值
        for v in children[u]:
            dfs(v)                        # 先算子树(后序遍历!)
            dp[u][0] += max(dp[v][0], dp[v][1])  # u不选→v选不选都行取max
            dp[u][1] += dp[v][0]                  # u选→v必须不选
    dfs(root)
    return max(dp[root])
```

**核心代码 — 树背包DP (DFS序+背包)**:
```python
# 在树上选K个节点使总价值最大,父子有约束
# 方法: DFS序把树变线性,然后在DFS序上做背包
# 核心: dfs(u)返回u的子树大小sz[u], DFS序中u的子树是连续的sz[u]个元素
# 在DFS序上做0-1背包: 每个节点选或不选,选了要跳过它的子树(因为子树不能和它同时选或不选看约束)

# 简化版: 树背包=DFS序上0-1背包,注意选节点时要考虑子树的约束
# hw参考: 题库DFS序/树上背包DP题, hw 24637
```

**注意**:
- 树DP核心: 后序遍历(先算子树再算当前), dp[u]依赖dp[子节点]
- 选/不选模式: dp[u][0]=不选u, dp[u][1]=选u, 约束体现在转移方程中
- 树背包: DFS序把树"线性化",然后做背包,注意子树在DFS序中是连续区间

---

### 4. 构造/DP — 实用难度

**题目特征**: 构造满足条件的方案、最优调度、方案数DP

**模板题**: 题库构造/DP题

**核心代码 — 方案数DP**:
```python
# 统计满足某条件的方案数(而非最值)
# dp[i]表示到达状态i的方案数, 转移时累加而非取max/min
MOD = 10**9 + 7
dp = [0]*(n+1); dp[0] = 1               # 初始:1种方案(什么都不做)
for i in range(1, n+1):
    for 转移条件:
        dp[i] += dp[前驱状态]              # 方案数累加!
        dp[i] %= MOD                       # 大数取模
```

**注意**:
- 最值DP取max/min; 方案数DP累加(+)并取模
- 构造题: 先DP找到最优值,再回溯构造方案(记录选择路径)

---

### 5. 带状态BFS/状态压缩DP — 实用难度

**题目特征**: 旅行售货商(TSP)、棋盘覆盖、位掩码表示"哪些已访问"、最少操作步数(状态空间搜索)

**模板题**: hw2 T30201旅行售货商( bitmask DP), 题库带状态BFS/状态压缩DP

**核心代码 — TSP Bitmask DP (旅行售货商)**:
```python
def tsp(n, dist):                          # dist[i][j]=i→j的距离矩阵
    INF = float('inf')
    dp = [[INF]*n for _ in range(1<<n)]    # dp[mask][u]: 已访问mask中的节点,最后在u,的最小代价
    dp[1][0] = 0                            # 只访问节点0(0号出发),停在0,代价0
    for mask in range(1<<n):               # 枚举所有访问状态(2^n种)
        for u in range(n):
            if dp[mask][u] == INF: continue  # 此状态不可达→跳过
            for v in range(n):
                if mask & (1<<v): continue   # v已在mask中(已访问)→跳过
                new_mask = mask | (1<<v)     # 访问v后的新状态
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u]+dist[u][v])  # 松弛
    # 全访问后回到起点0
    full = (1<<n)-1                          # 所有节点都访问了
    return min(dp[full][u]+dist[u][0] for u in range(n))
# hw2参考: T30201旅行售货商
# 核心思想: mask的二进制位表示"哪些节点已访问", 每次从u去没访问过的v, mask多一位
```

**核心代码 — 带状态BFS (在状态空间中搜索最短路)**:
```python
# 当普通BFS不够→节点本身有状态(如棋盘局面),BFS在状态空间中搜索
# 状态编码: 用整数/元组表示当前局面, BFS扩展所有可能的下一局面
from collections import deque
def state_bfs(initial_state, target_state, get_neighbors):
    visited = {initial_state}
    q = deque([(initial_state, 0)])        # (当前状态, 步数)
    while q:
        state, steps = q.popleft()
        if state == target_state: return steps
        for next_state in get_neighbors(state):
            if next_state not in visited:
                visited.add(next_state)
                q.append((next_state, steps+1))
    return -1
```

**注意**:
- Bitmask DP: mask的每一位=一个选择(0=没选, 1=选了), 状态转移=把一位从0变1
- TSP关键: `mask|(1<<v)`是新状态, 从u到v的代价累加
- 带状态BFS: 把"局面"当节点, BFS在局面空间找最短操作步数
- n≤20时Bitmask DP可行(2^20≈10^6), n更大→TLE,要换思路

---

## Category 5: 高级数据结构与数学搜索 (5题)

### 6. 滑动窗口/单调队列 — 中等难度

**题目特征**: 滑动窗口最大值/最小值、固定窗口查询最值、DP需要取窗口max

**模板题**: LC239 Sliding Window Maximum, hw4 P2698花盆(单调队列)

**核心代码 — LC239 滑动窗口最大值 (单调双端队列)**:
```python
from collections import deque
def maxSlidingWindow(nums, k):
    q = deque()                # 存索引, 队头到队尾值递减(队头=窗口最大)
    res = []
    for i in range(len(nums)):
        while q and nums[q[-1]] <= nums[i]:  # 队尾比当前小→删(当前更大,队尾没用了)
            q.pop()
        q.append(i)                              # 当前入队尾
        if q[0] <= i-k: q.popleft()               # 队头超出窗口范围→过期删除
        if i >= k-1: res.append(nums[q[0]])         # 窗口形成,队头=最大值
    return res
```

**注意**:
- 单调队列 vs 单调栈: 队列两端都能操作(左边过期删除,右边无用删除);栈只能右端操作
- 存索引而非值→方便判断是否过期(`q[0]<=i-k`)
- DP优化时: 朴素转移O(n×k),单调队列O(n)(窗口内max/min取O(1))

---

### 7. 剪枝搜索/约数分解 — 中等难度

**题目特征**: 分解质因数、约数枚举、数论+搜索剪枝

**模板题**: 题库剪枝搜索题

**核心代码 — 质因数分解**:
```python
def prime_factors(n):
    factors = []; i = 2
    while i*i <= n:                    # 只需试到sqrt(n)
        while n % i == 0:              # i是因子→反复除
            factors.append(i); n //= i
        i += 1
    if n > 1: factors.append(n)        # 剩余>1→本身就是大质因子
    return factors

# 约数枚举
def divisors(n):
    divs = []
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            divs.append(i)
            if i != n//i: divs.append(n//i)  # 配对约数
    return sorted(divs)
```

**注意**:
- 质因数分解到sqrt(n)就够了,剩的n>1就是最后一个质因子
- 约数枚举也是到sqrt(n), i和n//i是一对约数

---

### 8. 数论(最小质因数)/BFS — 困难难度

**题目特征**: 通过质数传送(质数相邻→BFS)、质数筛法、数论+最短路

**模板题**: LC3629 通过质数传送到达终点(质数+BFS), 题库数论+BFS题

**核心代码 — 质数筛法(埃拉托斯特尼筛)**:
```python
def sieve(n):                          # 返回<=n的所有质数
    is_prime = [True]*(n+1); is_prime[0]=is_prime[1]=False
    for i in range(2, int(n**0.5)+1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):  # i的倍数都不是质数
                is_prime[j] = False
    return [i for i in range(2,n+1) if is_prime[i]]
# 筛法O(n log log n), 比逐个判断O(n sqrt(n))快很多
# 预处理一次,后续O(1)判断质数: is_prime[x]
```

**核心代码 — 数论+BFS (质数邻接图上的最短路)**:
```python
# LC3629: 从a到b,每步只能换一位数字且结果必须是质数→质数间BFS
# 步骤1: 筛出范围内所有质数
# 步骤2: 建质数邻接图(只差一位数字的质数互连)
# 步骤3: BFS找最短路

def prime_bfs(start, end, limit):
    primes = sieve(limit)
    prime_set = set(primes)
    # 建邻接图: 两个质数只差一位→相连
    graph = defaultdict(list)
    for p in primes:
        s = str(p)
        for i in range(len(s)):
            for d in '0123456789':
                if i==0 and d=='0': continue  # 不能有前导零
                if d == s[i]: continue
                new = int(s[:i]+d+s[i+1:])
                if new in prime_set:
                    graph[p].append(new)
    # BFS
    visited = {start}; q = deque([(start,0)])
    while q:
        cur, dist = q.popleft()
        if cur == end: return dist
        for nxt in graph[cur]:
            if nxt not in visited:
                visited.add(nxt); q.append((nxt,dist+1))
    return -1
```

**注意**:
- 数论题+BFS: 先筛质数预处理,然后建特殊邻接图(质数间的连通关系),最后BFS
- 篮法是预处理,一次算好所有质数,后续O(1)查询

---

### 9. 离散化/BIT(树状数组)/懒删除堆 — 实用难度

**题目特征**: 大范围值域压缩到小范围(离散化)、区间查询(BIT, 超纲但可简化)、堆中删除指定元素

**模板题**: hwA 27093排队又来了(线段树/离散化), 题库BIT/懒删除堆题

**核心代码 — 离散化 (大值域压缩)**:
```python
# 当值域很大(如10^9)但实际只用了少量值(如10^5个)→离散化压缩到1..10^5
def discretize(arr):
    sorted_unique = sorted(set(arr))      # 去重排序
    # 映射: 原值→压缩后的排名(1-based)
    rank = {v: i+1 for i, v in enumerate(sorted_unique)}
    return [rank[x] for x in arr], rank    # 返回压缩后的数组和映射字典

# 示例: arr=[1000000000, 5, 1000000000, 3]
# sorted_unique=[3,5,1000000000], rank={3:1, 5:2, 1000000000:3}
# 压缩后=[3,2,3,1] → 值域从10^9压缩到3, 可以用数组做索引了!
```

**核心代码 — 懒删除堆 (堆中标记删除而非真正删除)**:
```python
# Python的heapq不支持删除指定元素→用"懒删除":标记为无效, pop时跳过
import heapq
lazy_heap = []          # 堆中存(值, id)
deleted = set()         # 标记删除的id集合

def push(val, id):
    heapq.heappush(lazy_heap, (val, id))

def pop():
    while lazy_heap:                          # 不断pop直到遇到未删除的
        val, id = heapq.heappop(lazy_heap)
        if id not in deleted: return val, id  # 有效→返回
    return None                               # 堆空

def delete(id):
    deleted.add(id)           # 只是标记,不真正从堆中删除(等pop时跳过)
# 适用场景: 需要删除堆中指定元素,但heapq没有remove操作
```

**注意**:
- 离散化: 先去重排序→建立原值到排名的映射→后续操作用排名代替原值
- BIT(树状数组)超纲!但如果数据范围小(离散化后),可以用前缀和+二分代替
- 懒删除堆: mark删除而不是真正删除,pop时跳过已标记的(有效元素堆顶才是真的)

---

### 10. 线段树/懒惰传播 — 实用难度 (超纲! 但可简化)

**题目特征**: 区间修改+区间查询(线段树超纲,但可以用其他方法简化)

**模板题**: hw8 M307区域和检索(线段树), hw520 T30878力场叠加(线段树懒传播), hwA 27093

**超纲提醒**: 线段树和树状数组都不考! 但有些题可以不用线段树也能做:
- 区间求和→前缀和O(1)查询(但不能修改)
- 区间最值→暴力O(n)(数据小时可行)
- 单点修改+区间查询→前缀和每次重新计算O(n),数据小时可行

**核心代码 — 前缀和替代线段树(仅适用于不修改或修改少)**:
```python
# 不修改的前缀和: Part3已有, O(1)区间查询
# 需要修改: 每次修改后重新计算前缀和O(n), 适合修改次数少的场景

# 单点修改+区间查询(暴力版,数据≤10^5且修改少时可行)
arr = [0]*n
for 操作:
    if 修改: arr[i] = v; 重新算prefix(如果需要)
    if 查询: 暴力sum(arr[l:r+1])或用prefix
```

**核心代码 — 线段树(如果你有余力想学,参考用)**:
```python
# 注意: 线段树超纲! 以下仅供有余力的同学参考
class SegTree:
    def __init__(self, n):
        self.n = n; self.tree = [0]*(4*n)  # 线段树需要4n空间

    def update(self, idx, val, node=1, l=0, r=None):
        if r is None: r = self.n-1
        if l == r: self.tree[node] = val; return   # 叶子节点→直接赋值
        mid = (l+r)//2
        if idx <= mid: self.update(idx,val,node*2,l,mid)      # 在左半
        else: self.update(idx,val,node*2+1,mid+1,r)           # 在右半
        self.tree[node] = self.tree[node*2] + self.tree[node*2+1]  # 合并左右

    def query(self, ql, qr, node=1, l=0, r=None):
        if r is None: r = self.n-1
        if ql > r or qr < l: return 0               # 范围不重叠→0
        if ql <= l and r <= qr: return self.tree[node]  # 完全包含→直接返回
        mid = (l+r)//2
        return self.query(ql,qr,node*2,l,mid) + self.query(ql,qr,node*2+1,mid+1,r)
# hw参考: hw8 M307(线段树), 但超纲!考试如果遇到区间查询,先用前缀和或暴力
```

**注意**:
- 线段树超纲! 题目可能有不用线段树的解法(前缀和/暴力/离散化)
- 懒传播(Lazy Propagation)也超纲,更复杂,考试不需要
- 如果考试真的遇到区间修改+查询,先试前缀和+暴力,数据小可能够用

---

## 附: 图论完整模板补充 (Dijkstra/Floyd/Kruskal/AOE)

> 这些在Part5只简略提到,这里给出完整代码(考试抄用)

### Dijkstra (单源最短路, 非负权)
```python
import heapq
def dijkstra(graph, start, n):
    dist = [float('inf')]*(n+1); dist[start] = 0
    pq = [(0, start)]                    # 小顶堆:(距离,节点)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue         # 旧条目,跳过(已有更短路径到u)
        for v, w in graph[u]:            # 遍历u的邻居
            if dist[u]+w < dist[v]:      # 经过u到v更短→松弛
                dist[v] = dist[u]+w
                heapq.heappush(pq, (dist[v], v))
    return dist
# hw参考: hw520 05443兔子与樱花, hwD 30910邮递员(正反向Dijkstra)
# 核心: 贪心+堆,每次弹出距起点最近的节点确认其最短路,然后用它松弛邻居
```

### Floyd-Warshall (全源最短路)
```python
def floyd(n, dist):
    for k in range(1,n+1):              # k在外层:逐步允许更多中间节点
        for i in range(1,n+1):
            for j in range(1,n+1):
                if dist[i][k]+dist[k][j] < dist[i][j]:  # 经过k更短→更新
                    dist[i][j] = dist[i][k]+dist[k][j]
# 初始化: dist[i][j]=0(if i==j), inf(否则), 有边填权值
# hw参考: hw520 05443(也可用Floyd做), hw9 sy382(Floyd判环)
# 核心: DP, dist[i][j]=只经过≤k的中间节点的最短路, k递增
```

### Kruskal MST
```python
def kruskal(edges, n):
    edges.sort(key=lambda e: e[2])      # 按权值升序
    dsu = DSU(n); total = 0; cnt = 0
    for u, v, w in edges:
        if dsu.find(u) != dsu.find(v):  # 两端不连通→不会成环→加入
            dsu.union(u, v); total += w; cnt += 1
            if cnt == n-1: break        # MST有n-1条边
    return total                         # cnt<n-1→图不连通
# hw参考: hw5 05442兔子与星空, hw520 01258, hwD 27351
# 核心: 贪心+DSU, 最小边优先,两端不连通就加入
```

### AOE关键路径
```python
def critical_path(graph, rg, n):        # graph=正向邻接, rg=反向邻接
    order = topo_sort(graph, n)          # 先拓扑排序
    if not order: return -1              # 有环
    ve = [0]*(n+1)                       # 最早发生时间(正向推)
    for u in order:
        for v, w in graph[u]: ve[v] = max(ve[v], ve[u]+w)  # 取max:所有前置完成
    vl = [ve[order[-1]]]*(n+1)           # 最晚发生时间(反向推),终点vl=ve[终点]
    for v in reversed(order):
        for u, w in rg[v]: vl[u] = min(vl[u], vl[v]-w)  # 取min:不延误任何后续
    critical = []
    for u in range(1,n+1):
        for v, w in graph[u]:
            if vl[v]-ve[u] == w: critical.append((u,v))  # 松弛=0→关键活动
    return ve, vl, critical
# hw参考: hwD 30899火星大工程
# 核心: 正向推ve(前置完成后才能开始),反向推vl(不延误终点前提下最晚开始),松弛=0是关键
```

### BFS网格最短路 (完整版)
```python
from collections import deque
def bfs_grid(grid, start, end):
    R, C = len(grid), len(grid[0])
    visited = set(); q = deque([(start[0],start[1],0)])
    visited.add(start)
    while q:
        r, c, d = q.popleft()
        if (r,c) == end: return d        # 首次到达=最短
        for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr,nc = r+dr,c+dc
            if 0<=nr<R and 0<=nc<C and (nr,nc) not in visited and grid[nr][nc]!='#':
                visited.add((nr,nc)); q.append((nr,nc,d+1))
    return -1
# hw参考: hw520 20741两座孤岛(多源BFS变体), hw9 28046词梯
# 核心: 入队即标记visited, BFS逐层扩展=无权最短路
```

### 前缀和 (1D/2D)
```python
# 1D: prefix[i+1]=prefix[i]+arr[i], 查询[l,r]: prefix[r+1]-prefix[l]
prefix = [0]*(n+1)
for i in range(n): prefix[i+1] = prefix[i]+arr[i]

# 2D: 容斥原理, 查询(r1,c1)-(r2,c2):
# ps[r2+1][c2+1]-ps[r1][c2+1]-ps[r2+1][c1]+ps[r1][c1]
ps = [[0]*(C+1) for _ in range(R+1)]
for i in range(R):
    for j in range(C):
        ps[i+1][j+1] = grid[i][j]+ps[i][j+1]+ps[i+1][j]-ps[i][j]
# LC参考: LC304(2D前缀和), LC560(1D前缀和+哈希)
# hw2参考: M304二维区域和检索
```

### 归并排序逆序对
```python
def merge_sort_count(nums):
    if len(nums) <= 1: return nums, 0
    mid = len(nums)//2
    left, cl = merge_sort_count(nums[:mid])
    right, cr = merge_sort_count(nums[mid:])
    merged = []; inv = cl+cr; i=j=0
    while i<len(left) and j<len(right):
        if left[i] <= right[j]: merged.append(left[i]); i+=1
        else: merged.append(right[j]); j+=1; inv += len(left)-i  # left[i..]全比right[j]大
    merged += left[i:]; merged += right[j:]
    return merged, inv
# hw参考: hw3 M02299 Ultra-QuickSort
# 核心: 合并时left[i]>right[j]→left[i..mid]全部是right[j]的逆序,一口气计数len(left)-i
```
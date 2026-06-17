# DSA 机考 Cheatsheet — Part4: 题库类一 — 基础数据结构+双指针(8题)

> 对应30题题库 **Category 1**: 基础数据结构与双指针
> 每题格式: 【题目特征】→【模板题】→【核心代码】→【注意】

---

## P1. 哈希表 (set) — 基础难度

**题目特征**: 去重、查找是否存在、两数之和、频率统计

**模板题**: LC1 Two Sum, LC128 Longest Consecutive Sequence, hw1: E1356

**核心代码 — LC1 Two Sum (哈希表查找)**:
```python
def twoSum(nums, target):
    seen = {}                          # 哈希表: {数值:下标}
    for i, num in enumerate(nums):
        complement = target - num      # 需要找的配对数
        if complement in seen:         # 配对数已在表中→找到!
            return [seen[complement], i]
        seen[num] = i                  # 当前数记入表,等后续配对
```

**核心代码 — LC128 Longest Consecutive (集合找连续)**:
```python
def longestConsecutive(nums):
    s, longest = set(nums), 0          # set去重,O(1)查找
    for num in s:
        if num-1 not in s:             # num-1不在→num是连续序列起点(只从起点开始)
            cur, length = num, 1
            while cur+1 in s:          # 从起点往后连续数
                cur += 1; length += 1
            longest = max(longest, length)
    return longest
```

**注意**:
- `set`查找O(1), `list`查找O(n) — 查多必用set
- `defaultdict(int)`做计数比手动初始化方便
- LC128只从起点(`num-1 not in s`)开始遍历,避免重复,O(n)而非O(n^2)

---

## 2. 双指针/单次遍历 — 基础难度

**题目特征**: 两数之和(有序数组)、移动零、删除元素、判断子序列

**模板题**: LC283 Move Zeroes, LC11 Container With Most Water, LC15 3Sum

**核心代码 — LC283 Move Zeroes (快慢指针原地操作)**:
```python
def moveZeroes(nums):
    pos = 0                            # pos: 下一个非零应该放的位置
    for i in range(len(nums)):
        if nums[i] != 0:               # 当前是非零→搬到前面
            nums[pos] = nums[i]; pos += 1
    for i in range(pos, len(nums)):    # 剩余位置填零
        nums[i] = 0
```

**核心代码 — LC11 Container With Most Water (左右指针收缩)**:
```python
def maxArea(height):
    l, r, ans = 0, len(height)-1, 0    # 左右两端开始
    while l < r:
        ans = max(ans, min(height[l],height[r])*(r-l))  # 当前面积=短板×宽度
        if height[l] < height[r]:      # 移动矮的那边(短板决定容量)
            l += 1
        else:
            r -= 1
    return ans
```

**核心代码 — LC15 3Sum (排序+固定+双指针)**:
```python
def threeSum(nums):
    nums.sort(); res = []              # 排序后才能用双指针
    for i in range(len(nums)-2):
        if i>0 and nums[i]==nums[i-1]: continue  # 跳过重复(去重)
        l, r = i+1, len(nums)-1        # 双指针在i之后范围
        while l < r:
            s = nums[i]+nums[l]+nums[r]
            if s < 0: l += 1           # 和太小→左指针右移(增大)
            elif s > 0: r -= 1         # 和太大→右指针左移(减小)
            else:
                res.append([nums[i],nums[l],nums[r]])
                while l<r and nums[l]==nums[l+1]: l += 1  # 跳重复左
                while l<r and nums[r]==nums[r-1]: r -= 1  # 跳重复右
                l += 1; r -= 1
    return res
```

**注意**:
- 双指针前提: **数组必须有序**(排序后或本身就是有序的)
- LC11移动矮边的直觉: 如果移动高边,面积只会更小(宽度减小且高度不变或更低)
- LC15去重: 固定i跳相同,移动l/r也要跳相同,否则会出重复三元组

---

## 3. 队列 (deque) — 基础难度

**题目特征**: 约瑟夫问题、打印机模拟、BFS队列、滑动窗口

**模板题**: LC102 Level Order, LC207 Course Schedule

**核心代码 — BFS层序遍历 (LC102)**:
```python
from collections import deque
def levelOrder(root):
    if not root: return []
    q = deque([root]); res = []
    while q:
        level = []
        for _ in range(len(q)):          # 一次pop完一层(len(q)锁定当前层大小)
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level)
    return res
```

**注意**:
- `len(q)`在循环开始时记录,循环中append新节点不影响(因为for范围已固定)
- 约瑟夫问题: deque报数旋转, `q.rotate(-k)`左转k位然后pop

---

## 4. 哈希表/二分查找 — 基础难度

**题目特征**: 查询距离最近的相等元素、频率查询+二分定位

**模板题**: LC3488 Closest Equal Element Queries

**核心代码 — 二分查找基础模板**:
```python
import bisect

# bisect_left(arr, x): 第一个>=x的位置(插入位置)
# bisect_right(arr, x): 第一个>x的位置
# arr必须有序!

arr = [1,3,5,7,9]
bisect.bisect_left(arr, 5)    # 2, 5在位置2
bisect.bisect_right(arr, 5)   # 3, 5之后的位置3
bisect.bisect_left(arr, 6)    # 3, 6不存在,插入位置3(在5和7之间)

# 手写二分(考试更灵活)
def binary_search(arr, target):
    lo, hi = 0, len(arr)-1
    while lo <= hi:
        mid = (lo+hi)//2
        if arr[mid] == target: return mid
        elif arr[mid] < target: lo = mid+1
        else: hi = mid-1
    return -1                              # 没找到

# 二分答案(搜索满足条件的最小/最大值) — 考试高频!
def bsearch_min(low, high, check):         # check(mid): mid可行?
    lo, hi = low, high
    while lo < hi:
        mid = (lo+hi)//2
        if check(mid): hi = mid            # mid可行→试试更小
        else: lo = mid+1                   # mid不行→必须更大
    return lo                              # 最小可行值

def bsearch_max(low, high, check):
    lo, hi = low, high
    while lo < hi:
        mid = (lo+hi+1)//2                 # 向上取整!防死循环
        if check(mid): lo = mid            # mid可行→试试更大
        else: hi = mid-1                   # mid不行→必须更小
    return lo                              # 最大可行值

# LC参考: LC35(搜索插入), LC33(旋转数组搜索), LC34(找边界)
# hw参考: hw5 M02774木材加工(二分答案), hwA M20746(二分答案)
```

**注意**:
- 二分答案的关键: 先确定check函数的含义,再决定是bsearch_min还是bsearch_max
- 最大化答案时mid向上取整`(lo+hi+1)//2`,否则lo可能不动→死循环
- bisect模块比手写方便,但手写更灵活(可以加自定义条件)

---

## 5. 滑动窗口/双指针 — 中等难度

**题目特征**: 固定长度窗口统计、最长不含重复子串、最小覆盖子串、频率匹配

**模板题**: LC3 Longest Substring Without Repeating, LC76 Minimum Window Substring, LC438 Find All Anagrams

**核心代码 — LC3 无重复最长子串 (滑动窗口+set)**:
```python
def lengthOfLongestSubstring(s):
    left = 0; seen = set(); ans = 0      # left=窗口左边界
    for right in range(len(s)):           # right=窗口右边界(逐个右移)
        while s[right] in seen:           # 新字符已在窗口内→收缩左边界
            seen.remove(s[left]); left += 1
        seen.add(s[right])                # 新字符入窗口
        ans = max(ans, right-left+1)      # 更新最大长度
    return ans
```

**核心代码 — LC438 找所有异位词 (固定窗口+频率对比)**:
```python
def findAnagrams(s, p):
    from collections import Counter
    need = Counter(p)                    # 目标频率
    window = Counter()                   # 当前窗口频率
    res = []; left = 0; k = len(p)       # 窗口大小=目标串长度
    for right in range(len(s)):
        window[s[right]] += 1            # 右边入窗口
        if right-left+1 > k:             # 窗口超长→左边出窗口
            window[s[left]] -= 1
            if window[s[left]] == 0: del window[s[left]]
            left += 1
        if right-left+1 == k and window == need:  # 窗口大小等于k且频率匹配
            res.append(left)
    return res
```

**注意**:
- 滑动窗口核心: 维持窗口满足某条件,右端扩张,左端收缩
- Counter可以直接比较: `window == need` 检查频率完全相同
- 固定窗口(right-left+1==k)和可变窗口(right-left+1动态变化)是两种模式

---

## 6. 单调栈 — 中等难度

**题目特征**: 下一个更大/更小元素、删数字使最小/最大、柱状图最大矩形、每日温度

**模板题**: LC739 Daily Temperatures, LC84 Largest Rectangle, hw5 T30102, hwA E04137

**核心代码 — LC739 每日温度 (找下一个更高温度)**:
```python
def dailyTemperatures(T):
    n = len(T); ans = [0]*n; stack = []  # stack存索引,维持温度递减
    for i in range(n):
        while stack and T[stack[-1]] < T[i]:  # 栈顶温度比当前低→找到了更高温度
            prev = stack.pop()                 # 弹出栈顶索引
            ans[prev] = i - prev               # 距离=当前-之前
        stack.append(i)                        # 当前入栈等待
    return ans                                  # 残留栈中ans=0(后面没有更高了)
```

**核心代码 — LC84 柱状图最大矩形 (单调递增栈)**:
```python
def largestRectangleArea(heights):
    stack = []; ans = 0
    heights = [0] + heights + [0]         # 首尾加0(哨兵),简化边界处理
    for i in range(len(heights)):
        while stack and heights[stack[-1]] > heights[i]:  # 当前比栈顶矮→栈顶找到右边界
            h = heights[stack.pop()]                      # 弹出的高度=矩形高度
            w = i - stack[-1] - 1                          # 宽度=右边界i - 新栈顶(左边界) - 1
            ans = max(ans, h*w)                            # 更新最大面积
        stack.append(i)
    return ans
```

**核心代码 — 删k位使数字最小 (贪心+单调栈)**:
```python
def delete_k_digits(s, k):
    stack = []
    for ch in s:
        while k > 0 and stack and stack[-1] > ch:  # 前面有更大的数字→删掉更优
            stack.pop(); k -= 1
        stack.append(ch)
    while k > 0: stack.pop(); k -= 1                # 还没删够→从末尾删
    return ''.join(stack).lstrip('0') or '0'        # 去前导零,全零返回'0'
# hwA参考: E04137最小新整数
```

**注意**:
- 单调栈存**索引**而非值,方便计算距离/宽度
- LC84首尾加0是关键技巧(哨兵):首0保栈不空,尾0强制弹出所有剩余
- 删数字变体: 核心是"遇到更小的就删前面更大的",贪心保证前导尽量小

---

## 7. 栈/字符串处理 — 困难难度

**题目特征**: 字符串解码`3[a2[bc]]`、复杂括号嵌套、最长有效括号

**模板题**: LC394 Decode String, LC32 Longest Valid Parentheses, hw5 M30637合法出栈序列

**核心代码 — LC394 Decode String (双栈:数字栈+字符串栈)**:
```python
def decodeString(s):
    num_stack = []; str_stack = []       # 数字栈存倍数,字符串栈存前缀
    cur = ""; k = 0                      # cur=当前正在构建的字符串, k=当前倍数
    for ch in s:
        if ch.isdigit():
            k = k*10 + int(ch)           # 数字可能多位: "12"→k从1变到12
        elif ch == '[':
            num_stack.append(k)          # 倍数入栈,准备开始新的内层
            str_stack.append(cur)        # 前缀入栈,内层结束后要拼回来
            cur = ""; k = 0              # 重置,开始构建内层字符串
        elif ch == ']':
            repeat = num_stack.pop()     # 弹出内层倍数
            cur = str_stack.pop() + cur * repeat  # 前缀 + 内层重复repeat次
        else:
            cur += ch                    # 字符直接追加到当前字符串
    return cur
```

**核心代码 — LC32 Longest Valid Parentheses (栈存索引)**:
```python
def longestValidParentheses(s):
    stack = [-1]; ans = 0               # 初始栈:-1(虚拟左边界)
    for i, ch in enumerate(s):
        if ch == '(':
            stack.append(i)              # 左括号位置入栈
        else:
            stack.pop()                  # 弹出匹配的左括号(或虚拟边界)
            if not stack:                # 栈空→多余的右括号,成为新虚拟边界
                stack.append(i)
            else:
                ans = max(ans, i-stack[-1])  # 当前右括号与栈顶左括号的距离=有效长度
    return ans
```

**核心代码 — 合法出栈序列判断**:
```python
def is_valid_pop_seq(push_seq, pop_seq):
    stack = []; i = 0                    # i=push_seq当前入栈位置
    for target in pop_seq:               # 逐个检查pop序列的每个目标
        while (not stack or stack[-1] != target) and i < len(push_seq):
            stack.append(push_seq[i]); i += 1  # 不断入栈直到栈顶=target
        if stack[-1] == target:
            stack.pop()                  # 匹配,弹出
        else:
            return False                 # 无法匹配→不合法
    return True
# hw5参考: M30637合法出栈序列
```

**注意**:
- LC394双栈是核心: 遇到`[`时把当前状态存入栈(相当于递归的"保存现场"),遇到`]`时弹出恢复
- LC32初始栈`[-1]`是关键: 如果没有虚拟边界,栈空时无法计算有效长度
- 合法出栈序列: 入栈直到栈顶等于目标,如果入完了还不等于→不合法

---

## 8. 栈/状态历史树 — 实用难度

**题目特征**: 浏览器前进后退、撤销/重做、状态回溯

**模板题**: LC1472 Browser History (类似概念)

**核心代码 — 双栈实现前进后退**:
```python
class BrowserHistory:
    def __init__(self, homepage):
        self.back_stack = [homepage]     # 后退栈:当前页面在栈顶
        self.forward_stack = []          # 前进栈

    def visit(self, url):
        self.back_stack.append(url)      # 访问新页面→加入后退栈
        self.forward_stack = []           # 访问新页面清空前进栈(不能再前进)

    def back(self, steps):
        while steps > 0 and len(self.back_stack) > 1:
            self.forward_stack.append(self.back_stack.pop())  # 当前页→前进栈
            steps -= 1
        return self.back_stack[-1]        # 返回当前页面

    def forward(self, steps):
        while steps > 0 and self.forward_stack:
            self.back_stack.append(self.forward_stack.pop())  # 前进栈→后退栈
            steps -= 1
        return self.back_stack[-1]
```

**注意**:
- 状态历史本质: 两个栈,后退把当前推入前进栈,前进把当前推入后退栈
- visit新页面时清空前进栈(新访问会"断掉"前进历史)
# DSA Assignment #8: 🌲（3/3）

*Updated 2026-04-21 19:09 GMT+8*
 *Compiled by <mark>李逸青 元培学院</mark> (2026 Spring)*



>**说明：**
>
>1. **解题与记录：**
>
>     对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
>2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的本人头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
> 
>3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。  
>
>请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### M1722.执行交换操作后的最小汉明距离

dsu, https://leetcode.cn/problems/minimize-hamming-distance-after-swap-operations/


思路：用并查集把可交换位置合并成组，组内分别统计 source 和 target 的元素频次，无法匹配的即为汉明距离。

代码：

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py
class Solution:
    def minimumHammingDistance(
        self,
        source: List[int],
        target: List[int],
        allowedSwaps: List[List[int]]
    ) -> int:
        n = len(source)
        uf = UnionFind(n)
        for a, b in allowedSwaps:
            uf.union(a, b)
        groups = defaultdict(lambda: [Counter(), Counter()])
        for i in range(n):
            root = uf.find(i)
            groups[root][0][source[i]] += 1
            groups[root][1][target[i]] += 1
        ans = 0
        for src_cnt, tgt_cnt in groups.values():
            total = sum(src_cnt.values())
            match = sum((src_cnt & tgt_cnt).values())
            ans += total - match
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20260505224720586](C:\Users\Katherine Lee\AppData\Roaming\Typora\typora-user-images\image-20260505224720586.png)



### T22161: 哈夫曼编码树

greedy, http://cs101.openjudge.cn/practice/22161/



代码：

```python
import heapq

class HuffmanNode:
    def __init__(self, chars, weight):
        self.chars = chars 
        self.weight = weight
        self.left = None
        self.right = None
    def __lt__(self, other):
        if self.weight != other.weight:
            return self.weight < other.weight
        return min(self.chars) < min(other.chars)
def build_huffman_tree(char_freq):
    heap = []
    for ch, w in char_freq.items():
        heapq.heappush(heap, HuffmanNode({ch}, w))
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(
            left.chars | right.chars,
            left.weight + right.weight
        )
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0] if heap else None
def generate_codes(root):
    codes = {}
    def dfs(node, code):
        if not node:
            return
        if len(node.chars) == 1:
            codes[next(iter(node.chars))] = code
            return
        dfs(node.left, code + "0")
        dfs(node.right, code + "1")
    dfs(root, "")
    return codes
def encode(s, codes):
    return "".join(codes[ch] for ch in s)
def decode(bits, root):
    result = []
    node = root
    for b in bits:
        node = node.left if b == "0" else node.right
        if len(node.chars) == 1:
            result.append(next(iter(node.chars)))
            node = root
    return "".join(result)
if __name__ == "__main__":
    n = int(input().strip())
    char_freq = {}
    for _ in range(n):
        line = input().split()
        ch = line[0]
        w = int(line[1])
        char_freq[ch] = w
    root = build_huffman_tree(char_freq)
    codes = generate_codes(root)
    import sys
    inputs = sys.stdin.read().strip().split("\n")
    for line in inputs:
        line = line.strip()
        if not line:
            continue
        if all(c in "01" for c in line):
            print(decode(line, root))
        else:
            print(encode(line, codes))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20260505225018660](C:\Users\Katherine Lee\AppData\Roaming\Typora\typora-user-images\image-20260505225018660.png)



### M晴问9.5: 平衡二叉树的建立

手搓AVL, https://sunnywhy.com/sfbj/9/5/359

代码：

```python
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1 
class AVLTree:
    def __init__(self):
        self.root = None
    def get_height(self, node):
        if not node:
            return 0
        return node.height
    def get_balance(self, node):
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)
    def right_rotate(self, z):
        y = z.left
        T2 = y.right
        y.right = z
        z.left = T2
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y
    def left_rotate(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self.get_height(z.left), self.get_height(z.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))        
        return y    
    def insert(self, root, key):
        if not root:
            return AVLNode(key)
        elif key < root.key:
            root.left = self.insert(root.left, key)
        else:
            root.right = self.insert(root.right, key)
        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))
        balance = self.get_balance(root)
        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)
        return root
    def preorder(self, root):
        result = []
        if root:
            result.append(str(root.key))
            result.extend(self.preorder(root.left))
            result.extend(self.preorder(root.right))
        return result
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    avl_tree = AVLTree()
    for num in arr:
        avl_tree.root = avl_tree.insert(avl_tree.root, num)
    preorder_result = avl_tree.preorder(avl_tree.root)
    print(' '.join(preorder_result))
if __name__ == "__main__":
    main()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20260505225344226](C:\Users\Katherine Lee\AppData\Roaming\Typora\typora-user-images\image-20260505225344226.png)



### M208.实现Trie（前缀树）

trie, https://leetcode.cn/problems/implement-trie-prefix-tree/

代码

```python
class Trie:
    def __init__(self):
        self.root = {}
        self.is_end = "#"
    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node:
                node[ch] = {}
            node = node[ch]
        node[self.is_end] = True
    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node:
                return False
            node = node[ch]
        return self.is_end in node
    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if ch not in node:
                return False
            node = node[ch]
        return True
```



<mark>（至少包含有"Accepted"）</mark>

![image-20260505224244283](C:\Users\Katherine Lee\AppData\Roaming\Typora\typora-user-images\image-20260505224244283.png)





## 2. 学习总结和个人收获

这次的作业还没有来得及好好消化，做完了相对好上手的week9作业才回来做，感觉有一些地方还是理解的不到位emm，5月份还是要花一些精力回顾树的讲义和经典例题。




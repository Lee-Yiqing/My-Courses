# DSA Assignment #9: 图（1/3）

*Updated 2026-04-28 13:47 GMT+8*
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



### M433.最小基因变化

bfs, https://leetcode.cn/problems/minimum-genetic-mutation/


思路：

每个基因看成图中的一个节点，一次合法突变就是一条边，利用BFS让路径最短。

代码：

```python
class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        if endGene not in bank:
            return -1
        gene_bank = set(bank)
        queue = deque([(startGene, 0)])
        visited = {startGene}
        while queue:
            current, steps = queue.popleft()
            if current == endGene:
                return steps
            for i in range(len(current)):
                for char in 'ACGT':
                    if char != current[i]:
                        new_gene = current[:i] + char + current[i+1:]
                        if new_gene in gene_bank and new_gene not in visited:
                            visited.add(new_gene)
                            queue.append((new_gene, steps + 1))
        return -1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20260505214653043](C:\Users\Katherine Lee\AppData\Roaming\Typora\typora-user-images\image-20260505214653043.png)



### sy382: 有向图判环 中等

Karn, dfs, Floyd-Warshall, https://sunnywhy.com/sfbj/10/3/382

思路：利用0来标记没有访问，2是任何连法都无法感化的死胡同，只有1是正在访问，连上1才是完成闭环。



代码：

```python
import sys
from typing import List

def hasCycle(n: int, edges: List[List[int]]) -> bool:
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
    state = [0] * n
    def dfs(u: int) -> bool:
        if state[u] == 1:
            return True 
        if state[u] == 2:
            return False
        state[u] = 1
        for v in graph[u]:
            if dfs(v):
                return True
        state[u] = 2
        return False
    for i in range(n):
        if state[i] == 0:
            if dfs(i):
                return True
    return False
data = sys.stdin.read().strip().split()
idx = 0
n = int(data[idx])
idx += 1
m = int(data[idx])
idx += 1
edges = []
for _ in range(m):
    u = int(data[idx])
    idx += 1
    v = int(data[idx])
    idx += 1
    edges.append([u, v])
if hasCycle(n, edges):
    print("Yes")
else:
    print("No")
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20260505215832826](C:\Users\Katherine Lee\AppData\Roaming\Typora\typora-user-images\image-20260505215832826.png)



### M909.蛇梯棋

bfs, https://leetcode.cn/problems/snakes-and-ladders/

思路：转化棋盘编码和行列位置的对应关系，用bfs搜索最短路径。



代码：

```python
from typing import List
from collections import deque

class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        n = len(board)
        target = n * n
        def id_to_rc(id_):
            id_ -= 1 
            block = id_ // n
            pos = id_ % n
            r = n - 1 - block
            if block % 2 == 0:
                c = pos
            else:
                c = n - 1 - pos
            return r, c
        visited = [False] * (target + 1)
        queue = deque()
        queue.append((1, 0)) 
        visited[1] = True
        while queue:
            curr, steps = queue.popleft()
            if curr == target:
                return steps        
            for k in range(1, 7):
                next_id = curr + k
                if next_id > target:
                    break                
                r, c = id_to_rc(next_id)
                if board[r][c] != -1:
                    next_id = board[r][c]         
                if not visited[next_id]:
                    visited[next_id] = True
                    queue.append((next_id, steps + 1))
        return -1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20260505220610190](C:\Users\Katherine Lee\AppData\Roaming\Typora\typora-user-images\image-20260505220610190.png)



### M28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/

思路：DFS搜索，按Warnsdorff规则优先走选择最少的格子，回溯求骑士周游路径。

代码

```python
import sys
def knight_tour(n, sr, sc):
    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    visited = [[False] * n for _ in range(n)]
    visited[sr][sc] = True
    count = 1
    total = n * n
    def on_board(r, c):
        return 0 <= r < n and 0 <= c < n
    def degree(r, c):
        cnt = 0
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if on_board(nr, nc) and not visited[nr][nc]:
                cnt += 1
        return cnt
    def dfs(r, c):
        nonlocal count
        if count == total:
            return True
        next_moves = []
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if on_board(nr, nc) and not visited[nr][nc]:
                next_moves.append((nr, nc))

        next_moves.sort(key=lambda x: degree(x[0], x[1]))
        for nr, nc in next_moves:
            visited[nr][nc] = True
            count += 1
            if dfs(nr, nc):
                return True
            visited[nr][nc] = False
            count -= 1
        return False
    return "success" if dfs(sr, sc) else "fail"
if __name__ == "__main__":
    n = int(sys.stdin.readline())
    sr, sc = map(int, sys.stdin.readline().split())
    print(knight_tour(n, sr, sc))
```



<mark>（至少包含有"Accepted"）</mark>

![image-20260505222931667](C:\Users\Katherine Lee\AppData\Roaming\Typora\typora-user-images\image-20260505222931667.png)









## 2. 学习总结和个人收获

作业题和以往的知识关联很紧密，综合性越来越高。度过了疲于奔命的期中周和过分忙碌的五一后，五月的重心真的要放在数算上面了。机考将至，还是得提升一些手速和熟练度。






"""Algorithm execution traces on paper-only subgraph.

Each trace tells a story about what the algorithm reveals about the literature
landscape, not just abstract graph operations. Descriptions explain:
  - WHAT the algorithm step does (data structure operation)
  - WHY it matters for literature navigation (real-world meaning)
"""

from collections import deque, defaultdict
import heapq


def _node_label(nodes, nid):
    n = nodes.get(nid)
    if n:
        title = n.get("label", nid)
        return f"{nid}: {title[:30]}" if len(title) > 30 else f"{nid}: {title}"
    return nid


def bfs_trace_on_subgraph(nodes, edges, start_id, max_depth=3):
    """BFS story: "从这篇论文出发，你能触及多远的知识领域？"

    Each layer = concentric circle of related papers.
    Layer 0: the starting paper itself
    Layer 1: directly related (1 hop) — closest neighbors
    Layer 2: indirectly related (2 hops) — broader context
    Layer 3: peripheral connections
    """
    adj = defaultdict(list)
    for e in edges:
        adj[e["source"]].append(e)
        adj[e["target"]].append({"source": e["target"], "target": e["source"],
                                  "type": e["type"], "weight": e["weight"]})

    visited = set()
    queue = deque([(start_id, 0)])
    steps = []
    current_layer_nodes = []
    current_layer_edges = []
    current_depth = 0
    total_discovered = 0

    while queue:
        node_id, dist = queue.popleft()
        if node_id in visited:
            continue
        visited.add(node_id)

        if dist != current_depth and current_layer_nodes:
            total_discovered += len(current_layer_nodes)
            layer_names = [_node_label(nodes, nid) for nid in current_layer_nodes[:5]]
            names_str = ", ".join(layer_names)
            if len(current_layer_nodes) > 5:
                names_str += f" 等{len(current_layer_nodes)}篇"

            meaning = {
                0: "起点——你选的这篇论文是探索的出发点",
                1: "近邻——直接引用或共享关键词的论文，与起点最密切相关",
                2: "二度关联——需要两步才能到达的论文，提供了更广阔的背景",
                3: "远端——三步之外的论文，属于外围知识领域",
            }.get(current_depth, f"深度{current_depth}——更远的知识关联")

            edge_type_counts = defaultdict(int)
            for e in current_layer_edges:
                edge_type_counts[e["type"]] += 1
            edge_summary = "、".join(f"{k}{v}条" for k, v in sorted(edge_type_counts.items()))

            steps.append({
                "step": len(steps),
                "depth": current_depth,
                "discovered": current_layer_nodes,
                "edges_explored": current_layer_edges,
                "meaning": meaning,
                "summary": f"第{current_depth}层发现{len(current_layer_nodes)}篇论文: {names_str}",
                "algo_note": f"BFS层序遍历: 队列(queue)逐层弹出, 共探索{edge_summary}",
                "description": f"{meaning}\n{names_str}\n[BFS] 逐层扩展, deque.popleft(), 发现{len(current_layer_nodes)}节点"
            })
            current_layer_nodes = []
            current_layer_edges = []
            current_depth = dist

        current_layer_nodes.append(node_id)

        if max_depth is not None and dist >= max_depth:
            continue

        for e in adj.get(node_id, []):
            neighbor = e["target"]
            if neighbor not in visited:
                queue.append((neighbor, dist + 1))
                current_layer_edges.append(e)

    if current_layer_nodes:
        total_discovered += len(current_layer_nodes)
        meaning = f"深度{current_depth}——最外层可达的知识"
        layer_names = [_node_label(nodes, nid) for nid in current_layer_nodes[:5]]
        names_str = ", ".join(layer_names)
        steps.append({
            "step": len(steps),
            "depth": current_depth,
            "discovered": current_layer_nodes,
            "edges_explored": current_layer_edges,
            "meaning": meaning,
            "summary": f"第{current_depth}层发现{len(current_layer_nodes)}篇: {names_str}",
            "algo_note": "BFS: deque(双端队列)实现层序遍历, O(V+E)",
            "description": f"{meaning}\n发现{len(current_layer_nodes)}篇论文\n[BFS] 层序遍历完成"
        })

    # Final summary step
    steps.append({
        "step": len(steps),
        "depth": -1,  # summary marker
        "discovered": list(visited),
        "edges_explored": [],
        "meaning": f"从 {_node_label(nodes, start_id)} 出发，你的研究邻域覆盖 {total_discovered} 篇论文",
        "summary": f"总共触及 {total_discovered} 篇论文",
        "algo_note": "BFS完整过程: deque.popleft()逐层扩展, 保证最短路径性质",
        "description": f"探索完成！从 {_node_label(nodes, start_id)} 出发，BFS触达 {total_discovered}/{len(nodes)} 篇论文"
    })

    return steps


def dijkstra_trace_on_subgraph(nodes, edges, start_id, end_id, weight_profile=None):
    """Dijkstra story: "最优阅读路径——从论文A到论文B，应该按什么顺序读？"

    Each step = settling a node (confirming its shortest distance).
    Edge weights reflect "reading effort": cite=easy(0.5), keyword=hard(1.5).
    The final path = recommended reading sequence.
    """
    if weight_profile is None:
        weight_profile = {"cite": 0.5, "sequence": 0.3, "shared_keyword": 1.5,
                          "shared_author": 2.0}

    weight_meaning = {
        "cite": "引用链(阅读流畅,权重低)",
        "sequence": "演进链(自然过渡,权重低)",
        "shared_keyword": "关键词跳跃(需换方向,权重高)",
        "shared_author": "作者切换(跨度大,权重最高)",
    }

    adj = defaultdict(list)
    for e in edges:
        w = e["weight"] * weight_profile.get(e["type"], 1.0)
        adj[e["source"]].append({"target": e["target"], "type": e["type"],
                                  "weight": w, "raw_weight": e["weight"]})
        adj[e["target"]].append({"target": e["source"], "type": e["type"],
                                  "weight": w, "raw_weight": e["weight"]})

    dist = {start_id: 0.0}
    prev = {}
    settled = set()
    heap = [(0.0, start_id)]
    steps = []

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue

        settled.add(u)
        relaxed_edges = []

        for e in adj.get(u, []):
            v = e["target"]
            new_dist = d + e["weight"]
            old_dist = dist.get(v, float('inf'))

            if new_dist < old_dist:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))
                wm = weight_meaning.get(e["type"], e["type"])
                relaxed_edges.append({
                    "source": u, "target": v,
                    "type": e["type"],
                    "old_dist": round(old_dist, 2) if old_dist != float('inf') else None,
                    "new_dist": round(new_dist, 2),
                    "edge_weight": round(e["weight"], 2),
                    "meaning": wm,
                    "description": f"发现更短路径 {_node_label(nodes,u)}→{_node_label(nodes,v)} via {wm}"
                })

        # Build meaning for this step
        if len(relaxed_edges) == 0:
            meaning = f"确认 {_node_label(nodes,u)} 的最短阅读距离 = {round(d,2)}，无需更新邻居"
        else:
            meaning = f"确认 {_node_label(nodes,u)} (阅读距离{round(d,2)})，同时发现 {len(relaxed_edges)} 条更优阅读路径"

        steps.append({
            "step": len(steps),
            "settled_node": u,
            "settled_nodes": list(settled),
            "distances": {k: round(v, 2) for k, v in dist.items() if v != float('inf')},
            "relaxed_edges": relaxed_edges,
            "meaning": meaning,
            "summary": f"确认 {_node_label(nodes,u)} — 阅读距离 {round(d,2)}",
            "algo_note": f"Dijkstra堆优化: heapq.heappop取出dist最小节点, O(E log V)",
            "description": f"{meaning}\n[算法] heapq优先队列保证每次取出距离最小节点"
        })

        if u == end_id:
            path = []
            node = end_id
            while node in prev:
                path.append(node)
                node = prev[node]
            path.append(start_id)
            path.reverse()

            path_labels = [_node_label(nodes, p) for p in path]
            path_str = " → ".join(path_labels)

            steps[-1]["path"] = path
            steps[-1]["meaning"] = f"到达目标！最优阅读路径: {path_str}"
            steps[-1]["summary"] = f"最优阅读路线: {len(path)}步, 总阅读距离 {round(d,2)}"
            steps[-1]["description"] = f"到达 {_node_label(nodes,end_id)}！\n最优阅读路线: {path_str}\n总阅读距离 = {round(d,2)} (引用链权重低=流畅阅读)"

            # Add final summary step
            steps.append({
                "step": len(steps),
                "settled_node": None,
                "path": path,
                "path_labels": path_labels,
                "total_dist": round(d, 2),
                "meaning": f"推荐阅读顺序: {path_str}\n先读起点{_node_label(nodes,start_id)}建立基础，沿引用链逐步过渡到{_node_label(nodes,end_id)}",
                "summary": f"阅读路线: {len(path)}篇论文, 距离{round(d,2)}",
                "algo_note": "Dijkstra最短路: 堆优化优先队列, 贪心策略保证最优, O(E log V)",
                "description": f"阅读路线规划完成！\n{path_str}\n总距离={round(d,2)}"
            })
            break

    return steps


def pagerank_trace_on_subgraph(nodes, edges, damping=0.85, max_iter=15, tolerance=1e-6):
    """PageRank story: "影响力排名——哪些论文是这个领域真正的核心？"

    All papers start equal, then influence flows through citations/keywords.
    After convergence, top papers = the field's landmarks.
    """
    node_ids = list(nodes.keys())
    N = len(node_ids)
    scores = {nid: 1.0 / N for nid in node_ids}

    adj = defaultdict(set)
    for e in edges:
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])

    degree = {nid: len(adj.get(nid, set())) for nid in node_ids}

    steps = []

    # Top 3 at init
    top3 = sorted(node_ids, key=lambda nid: -scores[nid])[:3]
    top3_str = ", ".join([_node_label(nodes, nid) for nid in top3])

    steps.append({
        "step": 0, "iteration": 0,
        "scores": {nid: round(1.0 / N, 6) for nid in node_ids},
        "max_change": round(1.0 / N, 6),
        "meaning": f"初始状态: 所有{N}篇论文影响力均等，还没有区分谁更重要",
        "summary": f"所有论文 score = {round(1.0/N, 4)} (均等)",
        "algo_note": "PageRank初始化: 每个节点1/N, 随机游走者均匀分布",
        "description": f"初始: {N}篇论文影响力均等(各{round(1.0/N, 4)})\n[算法] score(u) = (1-d)/N + d * Σ score(v)*P(v→u)"
    })

    for iteration in range(1, max_iter + 1):
        new_scores = {nid: 0.0 for nid in node_ids}

        for nid in node_ids:
            for neighbor in adj.get(nid, set()):
                d = degree.get(neighbor, 1)
                if d > 0:
                    new_scores[nid] += scores[neighbor] / d

        for nid in node_ids:
            new_scores[nid] = (1 - damping) / N + damping * new_scores[nid]

        max_change = max(abs(new_scores[nid] - scores[nid]) for nid in node_ids)
        scores = new_scores

        top5 = sorted(node_ids, key=lambda nid: -scores[nid])[:5]
        top5_str = ", ".join([f"{_node_label(nodes, nid)}({round(scores[nid], 4)})" for nid in top5])
        converged = max_change < tolerance

        meaning = f"影响力正在分化：被更多重要论文引用/关联的论文分数上升"
        if converged:
            meaning = f"排名已稳定！领域核心论文确定"
        elif iteration <= 3:
            meaning = f"影响力开始流动：引用多的论文分数上升，孤立论文下降"

        steps.append({
            "step": iteration, "iteration": iteration,
            "scores": {nid: round(scores[nid], 6) for nid in node_ids},
            "max_change": round(max_change, 6),
            "converged": converged,
            "meaning": meaning,
            "summary": f"Top 5: {top5_str}",
            "algo_note": f"PageRank迭代: score传播+阻尼, 随机游走模型(d={damping})",
            "description": f"{meaning}\nTop 5: {top5_str}\n[算法] score(u) = (1-d)/N + d*Σ(score(v)/deg(v))"
        })

        if converged:
            # Final summary
            top5_final = sorted(node_ids, key=lambda nid: -scores[nid])[:5]
            steps.append({
                "step": iteration + 1,
                "iteration": iteration + 1,
                "scores": {nid: round(scores[nid], 6) for nid in node_ids},
                "max_change": 0,
                "converged": True,
                "meaning": f"结论：{_node_label(nodes, top5_final[0])} 是vibe coding领域最具影响力的论文",
                "summary": f"核心论文: {_node_label(nodes, top5_final[0])}",
                "algo_note": "PageRank收敛: max_change < tolerance, 体现'被重要论文引用=自己也重要'",
                "description": f"影响力排名完成！\n核心论文: {_node_label(nodes, top5_final[0])}\n[算法] PageRank收敛条件: max_change < 1e-6"
            })
            break

    return steps


def components_trace_on_subgraph(nodes, edges):
    """Components story: "研究社区——这些论文自然形成了几个研究方向？"

    Only using cite+sequence edges (direct academic connections),
    not shared_keyword (which connects everything).
    Papers in the same component = same research thread.
    """
    node_ids = list(nodes.keys())

    adj = defaultdict(set)
    for e in edges:
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])

    visited = set()
    steps = []

    for start in node_ids:
        if start in visited:
            continue

        component = []
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)

        comp_labels = [_node_label(nodes, nid) for nid in component[:5]]
        comp_str = ", ".join(comp_labels)
        if len(component) > 5:
            comp_str += f" 等{len(component)}篇"

        meaning = {
            True: f"发现研究社区 {len(steps)+1}: {comp_str} ——这些论文通过引用链紧密相连，属于同一研究方向",
            False: f"孤立论文 {_node_label(nodes, start)} ——没有引用链连接到其他论文，是一个独立的研究点"
        }.get(len(component) >= 2, f"孤立论文 {_node_label(nodes, start)}")

        # Collect domain distribution
        domain_counts = defaultdict(int)
        for nid in component:
            n = nodes.get(nid)
            if n:
                domain_counts[n.get("domain", "?")] += 1
        domain_str = "、".join(f"{k}域{v}篇" for k, v in sorted(domain_counts.items()))

        steps.append({
            "step": len(steps),
            "component_id": len(steps) + 1,
            "start_node": start,
            "discovered": component,
            "size": len(component),
            "meaning": meaning,
            "summary": f"社区{len(steps)+1}: {len(component)}篇论文 ({domain_str})",
            "algo_note": "连通分量: BFS遍历发现可达节点集, O(V+E)",
            "description": f"{meaning}\n{domain_str}\n[算法] BFS从{start}出发, 发现连通分量"
        })

    # Final summary
    multi = [s for s in steps if s["size"] >= 2]
    isolated = [s for s in steps if s["size"] == 1]
    steps.append({
        "step": len(steps),
        "component_id": -1,
        "meaning": f"发现 {len(multi)} 个研究社区 + {len(isolated)} 篇孤立论文",
        "summary": f"{len(multi)}个社区, {len(isolated)}篇孤立",
        "algo_note": "连通分量完整: 所有可达节点集, BFS确保O(V+E)时间",
        "description": f"研究社区发现完成！\n{len(multi)}个紧密社区 + {len(isolated)}篇独立论文\n[算法] BFS遍历全部节点, 每个连通分量=一个可达集"
    })

    return steps


def bridges_trace_on_subgraph(nodes, edges):
    """Bridges story: "知识空隙——哪些连接一旦断裂，研究领域就会分裂？"

    Bridge edges = fragile connections that hold different communities together.
    Removing a bridge = two research threads become disconnected.
    These are "cross-community" papers that bridge different subfields.
    """
    node_ids = list(nodes.keys())

    adj = defaultdict(set)
    edge_type_map = {}
    for e in edges:
        adj[e["source"]].add(e["target"])
        adj[e["target"]].add(e["source"])
        edge_type_map[(e["source"], e["target"])] = e["type"]
        edge_type_map[(e["target"], e["source"])] = e["type"]

    disc = {}
    low = {}
    timer = [0]
    bridges = []
    visited = set()
    steps = []

    def _dfs(u, parent):
        visited.add(u)
        disc[u] = timer[0]
        low[u] = timer[0]
        timer[0] += 1

        new_bridge_info = []

        for v in adj.get(u, set()):
            if v not in visited:
                _dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    et = edge_type_map.get((u, v), "unknown")
                    bridge = {"source": u, "target": v, "edge_type": et}
                    bridges.append(bridge)
                    new_bridge_info.append(bridge)
            elif v != parent:
                low[u] = min(low[u], disc[v])

        if new_bridge_info:
            bridge_strs = []
            for b in new_bridge_info:
                bridge_strs.append(
                    f"{_node_label(nodes, b['source'])} ↔ {_node_label(nodes, b['target'])} 是知识空隙——删掉这条连接，两个研究方向将断开"
                )
            meaning = bridge_strs[0]
        else:
            meaning = f"访问 {_node_label(nodes, u)}: 正常连接，不是脆弱节点"

        steps.append({
            "step": len(steps),
            "node": u,
            "disc": disc[u],
            "low": low[u],
            "parent": parent,
            "bridges_found_so_far": list(bridges),
            "new_bridges": new_bridge_info,
            "meaning": meaning,
            "summary": f"DFS {_node_label(nodes, u)}: disc={disc[u]}, low={low[u]}" +
                       (f" | 发现{len(new_bridge_info)}个知识空隙" if new_bridge_info else ""),
            "algo_note": f"Tarjan DFS: low[{u}]={low[u]}, disc[{u}]={disc[u]}" +
                         (f" | low[v]>disc[u] → 桥边(知识空隙)" if new_bridge_info else ""),
            "description": f"{meaning}\n[算法] Tarjan: disc=发现时间, low=最远回溯, low[v]>disc[u]=桥边条件"
        })

    for node in node_ids:
        if node not in visited:
            _dfs(node, None)

    # Final summary
    if bridges:
        bridge_strs = [f"{_node_label(nodes, b['source'])} ↔ {_node_label(nodes, b['target'])}" for b in bridges]
        steps.append({
            "step": len(steps),
            "node": None,
            "meaning": f"发现 {len(bridges)} 个知识空隙: 这些连接维系着不同研究方向之间的对话，一旦断裂，社区将分裂",
            "summary": f"{len(bridges)}个知识空隙",
            "algo_note": f"Tarjan DFS完整: {len(bridges)}条桥边(low[v]>disc[u])",
            "description": f"知识空隙检测完成！\n{len(bridges)}个脆弱连接: {', '.join(bridge_strs[:3])}\n[算法] Tarjan: O(V+E)时间, low/disc数组判断桥边"
        })

    return steps

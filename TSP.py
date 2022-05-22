import math
from global_var import init_position
import collections


def tsp(graph, node_list):
    def get_distance(node_list):
        total_distance = 0
        empty_distance = 0
        task_distance = 0

        for node in node_list:
            total_distance += graph.distance_node(0, init_position, node[0][0])
            total_prev = node[0][0]

            empty_distance += graph.distance_node(0, init_position, node[0][0])
            empty_prev = node[0][1]
            empty_distance -= graph.distance_node(0, node[0][1], node[0][0])
            for start, end in node:
                total_distance += graph.distance_node(0, total_prev, start)
                total_distance += graph.distance_node(0, start, end)

                empty_distance += graph.distance_node(0, empty_prev, start)

                task_distance += graph.distance_node(0, start, end)

                total_prev = end
                empty_prev = end
            total_distance += graph.distance_node(0, total_prev, init_position)
            empty_distance += graph.distance_node(0, empty_prev, init_position)
        return total_distance, empty_distance, task_distance

    def dp(cur, visit, route, ans_candidate):
        if visit == (1 << N) - 2:
            ans_candidate.append(route)
            return graph.distance_node(0, dic[cur][1], init_position)

        if D[cur][visit]:
            return D[cur][visit]

        Min = 0xffffff
        for next in range(0, N):
            if visit & (1 << next):
                continue
            if cur == next:
                continue
            ret = dp(next, visit | (1 << next), route + [next], ans_candidate) + graph.distance_node(0, dic[cur][1], dic[next][0])
            Min = min(Min, ret)

        D[cur][visit] = Min
        return D[cur][visit]

    dic = collections.defaultdict(list)
    min_dist = math.inf
    for j in range(len(node_list)):
        dic[j] = node_list[j]
        temp = graph.distance_node(0, 0, node_list[j][0])
        if min_dist > temp:
            min_dist = temp
            dic[j], dic[0] = dic[0], dic[j]
    N = len(dic)

    D = [[0] * (1 << N) for _ in range(N)]
    ans_candidate = []
    dp(0, 0, [], ans_candidate)

    min_val = math.inf
    tsp_route = []
    for a in ans_candidate:
        temp = [dic[x] for x in a]
        _, ret, _ = get_distance([temp])
        if min_val > ret:
            min_val = ret
            tsp_route = temp
    tsp_route.insert(0, dic[0])

    origin_total_distance, _, _ = get_distance([node_list])
    tsp_total_distance, _, _ = get_distance([tsp_route])

    ret = [origin_total_distance, tsp_total_distance, tsp_route]
    return ret

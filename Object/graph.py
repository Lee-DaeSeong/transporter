import sys
import numpy as np

from Object.stock import Stockyard
from Object.intersection import Intersection
from Object.road import Road


class Graph:
    INF = sys.maxsize

    def __init__(self, s_data, i_data, r_data):
        self.stock_list = []        # 적치장 리스트
        self.inter_list = []        # 교차로 리스트
        self.road_list = []         # 도로 리스트
        self.graph = []             # 맵 거리
        self.width_list = []        # 도로 폭 리스트
        self.min_path_list = []     # 노드 간 최소 거리 리스트 [0: 거리, 1:이전 노드]
        self.total_ob = None        # 총 노드 수
        self.create_object(s_data, i_data, r_data)
        for road in self.road_list:
            self.width_list.append(road.w)

        # 도로 넓이 개수
        self.width_list = list(set(self.width_list))
        
        # 도로 넓이에 따라 최소 도착 지점 구하기
        self.min_path()
    
    # 맵 구성요소들 생성
    def create_object(self, s_data, i_data, r_data):
        for no, x, y in s_data:
            self.stock_list.append(Stockyard(no, x, y))

        for no, x, y in i_data:
            self.inter_list.append(Intersection(no, x, y))

        # start, end, width
        for s, e, w in r_data:
            s = self.node(s)
            e = self.node(e)
            self.road_list.append(Road(s, e, w))
        
        # 그래프 거리 적용하기
        self.total_ob = len(self.stock_list) + len(self.inter_list)
        self.graph = np.array([[self.INF for _ in range(self.total_ob)] for _ in range(self.total_ob)])
        self.graph[range(self.total_ob), range(self.total_ob)] = 0

        # road.s.no ?
        # d -> distance
        for road in self.road_list:
            i = road.s.no
            j = road.e.no
            self.graph[i][j] = road.d
            self.graph[j][i] = road.d

    # 도로 폭에 따른 최소 거리 구하기
    def min_path(self):
        for width in self.width_list:
            graph = self.graph
            for road in self.road_list:
                if road.w < width:          # 도로 폭이 현재 폭보다 작으면 이용 불가
                    s = road.s              # 시작 노드
                    e = road.e              # 끝 노드
                    graph[s][e] = self.INF  # 도로 사용 불가
                    graph[e][s] = self.INF  # 도로 사용 불가

            self.min_path_list.append(self.node_distance(graph=graph))
    
    # 모든 노드들 사이에서 최소 거리 구하기
    def node_distance(self, graph):
        min_path = []
        for i in range(self.total_ob):
            d = self.dijkstra(i, self.total_ob, graph)
            min_path.append(d)
        return min_path

    # 다익스트라 알고리즘
    def dijkstra(self, K, V, graph):
        # s는 해당 노드를 방문 했는지 여부를 저장하는 변수이다
        s = [False for _ in range(V)]
        # d는 memoization을 위한 array이다. d[i]는 정점 K에서 i까지 가는 최소한의 거리가 저장되어 있다.
        d = [[self.INF, None] for _ in range(V)]
        d[K] = [0, K]

        while True:
            m = self.INF
            N = -1

            # 방문하지 않은 노드 중 d값이 가장 작은 값을 선택해 그 노드의 번호를 N에 저장한다.
            # 즉, 방문하지 않은 노드 중 K 정점과 가장 가까운 노드를 선택한다.
            for j in range(V):
                if not s[j] and m > d[j][0]:
                    m = d[j][0]
                    N = j

            # 방문하지 않은 노드 중 현재 K 정점과 가장 가까운 노드와의 거리가 INF 라는 뜻은
            # 방문하지 않은 남아있는 모든 노드가 A에서 도달할 수 없는 노드라는 의미이므로 반복문을 빠져나간다.
            if m == self.INF:
                break

            # N번 노드를 '방문'한다.
            # '방문'한다는 의미는 모든 노드를 탐색하며 N번 노드를 통해서 가면 더 빨리 갈 수 있는 노드가 있는지 확인하고,
            # 더 빨리 갈 수 있다면 해당 노드(노드의 번호 j라고 하자)의 d[j]를 업데이트 해준다.
            s[N] = True

            for j in range(V):
                if s[j]:
                    continue
                if graph[N][j] == self.INF:
                    continue

                via = d[N][0] + graph[N][j]

                if d[j][0] > via:
                    d[j][0] = via
                    d[j][1] = N

        return d

    # 노드 반환
    def node(self, index):
        if index < len(self.stock_list):
            return self.stock_list[index]
        else:
            index -= len(self.stock_list)
            return self.inter_list[index]

    # 이동 경로에 있는 노드 반환
    def return_node_list(self, size, d, a, start_in_flag=False):
        total_dis = self.min_path_list[size][d][a][0]
        node_list = [a]
        pre_node = a
        while True:
            pre_node = self.min_path_list[size][d][pre_node][1]
            if pre_node == d:
                break
            node_list.insert(0, pre_node)

        # 출발지 포함
        if start_in_flag:
            node_list.insert(0, d)

        return total_dis, node_list

    # 노드 간 총 거리 반환
    def distance_node(self, w, s, e):
        # 모든 도로 이용가능
        return self.min_path_list[w][s][e][0]

    # 넓이 인덱스 반환
    def width_index(self, size):
        for index, width in enumerate(self.width_list):
            if width >= size:
                return index
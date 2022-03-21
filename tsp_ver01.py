import collections
import itertools
import math
import time
import copy
import time
import numpy as np
import random

# 트랜스 포터 작업 개수,
# 염색체 세대 수, 트랜스 포터 작업 개수
from global_var import *

from Data_create.call_data import object_data
from Data_create.create_trans import transporter_data
from Object.graph import Graph
from Object.Transporter import Trans_manager  # , child_Trans_manager
from Task.Task import Task_schedule, task_classification
from Simulation import perform_graph
from Simulation.transporter_view import transporter_schedule_view
from Generation.Population import Population
from Generation.Gen import GA
from Heuristic.Heuristic_condition import H1, H2, H3

np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # inf = infinity

######################## 변수 로드 #########################
# 데이터 불러오기
stock_data, inter_data, road_data = object_data()

# 그래프 생성
graph = Graph(stock_data, inter_data, road_data)
# print("Map Complete!!")

# 트랜스포터 목록 들고 오기
trans_manager = Trans_manager()
transporter_data(transporter_num, trans_manager, graph)
# print("Transporter Complete!!")

##########################################################

time_perform = []
trans_perform, dis_perform, c = [], [], 0
prev = time.time()

base_s = None
task_work_time = None
task_empty_time = None
for w_index, work_num in enumerate(task_num):
    ######################## 데이터 생성 함수 #########################
    # 작업목록 스케줄 생성 및 작업 관계 생성
    task_manager = Task_schedule(graph, work_num)
    ################################################################

    # 작업 시간, 작업 간 이동 시간, 공차 시간
    task_work_time, task_empty_time, empty_speed = task_classification(task_manager.task_list, trans_manager.t_list,
                                                                       graph=graph)

    ######################## 휴리스틱 알고리즘 #########################
    # 휴리스틱 1번
    # 트랜스포터의 대수 최소화
    # 작업량이 적은 트랜스포터 (a)에 있는 작업을 작업량이 많은 트랜스포터 (b)에 할당
    # 그 후, 트랜스포터 (b)는 새로운 작업의 공차거리를 최소화할 수 있는 순서에 작업 배치
    h1 = H1(task_work_time, task_empty_time, task_manager.task_list, graph=graph)
    # 새로운 스케줄이 생성 -> 휴리스틱이 돌아가는 것 -> 최종 결과

    # 초기 세대 생성
    pop = Population(task_work_time, task_empty_time, trans_manager, task_manager, initialise=True)
    ga_base = GA(task_manager, task_work_time, task_empty_time, h1=h1)

    # 결과 저장할 변수
    b_trans, bw_t, be_t, btotal_time, be_d = 0, 0, 0, 0, 0

    # 최적해 탐색                           #100
    for g_index, generation in enumerate(n_generations):
        pop_base = copy.deepcopy(pop)

        base_f, base_s = 0, None

        for i in range(generation):
            ########### 휴리스틱 알고리즘 돌아가는 부분 ###########
            pop_base = ga_base.evolvePopulation(pop_base, trans_manager, base_flag=True)
            ##################################################
            print("{}/{}/{}".format(w_index, g_index, i / generation))
            ################## 평가함수 ####################

            base_pop = pop_base.getfittest()
            temp_f = base_pop.getfitness(work_time=task_work_time, empty_time=task_empty_time)
            if base_f < temp_f:
                base_s = copy.deepcopy(base_pop)
                base_f = temp_f
        b_trans, bw_t, be_t, btotal_time, be_d = base_s.gettrans_num_time(work_time=task_work_time,
                                                                          empty_time=task_empty_time)  # scheduling.py -> Transporter.py

        # 대수,  작업 시간, 공차 시간, 작업 + 공차 시간, 총 이동거리
        print("work_num: ", work_num)
        print("base    : ", format(int(b_trans), ','), format(int(bw_t), ','), format(int(be_t), ','),
              format(int(btotal_time), ','), format(int(be_d), ','))

# print(time.time() - prev)


b_trans, bw_t, be_t, btotal_time, be_d, origin_node_list = base_s.gettrans_num_time_base(work_time=task_work_time,
                                                                                         empty_time=task_empty_time)


# 트랜스포터 번호 (트랜스포터 가용 무게) : task 번호(무게) (시작, 끝)

# 거리, 이전 노드
# print(graph.min_path())

# 총 이동 거리 == 0 -> (시작 -> 끝) -> (시작 -> 끝) ... (끝 -> 0)


# node_list 거리 합계
def get_distance(node_list):
    total_distance = 0
    empty_distance = 0
    task_distance = 0
    total_prev = 0
    end = 0
    empty_prev = 0
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
    return empty_distance


print(get_distance(origin_node_list))


def tsp(cur, visit, root):
    if visit == (1 << N)-2:
        global ans
        ans.append(root)
        return graph.distance_node(0, dic[cur][1], init_position)

    if D[cur][visit]:
        return D[cur][visit]

    Min = 0xffffff
    for next in range(0, N):
        if visit & (1 << next):
            continue
        if cur == next:
            continue
        ret = tsp(next, visit | (1 << next), root + [next]) + graph.distance_node(0, dic[cur][1], dic[next][0])
        Min = min(Min, ret)

    D[cur][visit] = Min
    return D[cur][visit]

ans=[]
for i in origin_node_list:
    ans.append(len(i))
print(ans)

origin_node_list.append([[5, 14], [8, 10], [12, 16]])
origin_node_list.append([[1, 1], [8, 10], [12, 16], [13, 9], [1, 2], [3, 4]])
prev = time.time()
temp=[]
for i in range(1, 18):
    temp.append([i, i])
origin_node_list.append(temp)
print(len(temp))
for i in origin_node_list:
    dic=collections.defaultdict(list)
    min_dist=math.inf
    for j in range(len(i)):
        dic[j]=i[j]
        temp=graph.distance_node(0, 0, i[j][0])
        if min_dist > temp:
            min_dist=temp
            dic[j], dic[0] = dic[0], dic[j]
    N = len(dic)

    D = [[0] * (1 << N) for _ in range(N)]
    print('origin :', get_distance([i]))
    ans=[]
    tsp(0, 0, [])

    min_val=math.inf
    root=[]
    for a in ans:
        temp=[dic[x] for x in a]
        ret = get_distance([temp])
        if min_val > ret:
            min_val=ret
            root=temp
    root.insert(0, dic[0])
    print('tsp :', get_distance([root]), root)

print(time.time() - prev)
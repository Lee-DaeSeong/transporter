import time
import copy
import time
import numpy as np

# 트랜스 포터 작업 개수, 염색체 세대 수, 트랜스 포터 작업 개수
from global_var import transporter_num, n_generations, task_num

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
# 100개만
# 100, 200, 300
for w_index, work_num in enumerate(task_num):
    ######################## 데이터 생성 함수 #########################
    # 작업목록 스케줄 생성 및 작업 관계 생성
    task_manager = Task_schedule(graph, work_num)
    ################################################################

    # 작업 시간, 작업 간 이동 시간, 공차 시간
    task_work_time, task_empty_time, empty_speed = task_classification(task_manager.task_list, trans_manager.transporter_list,
                                                                       graph=graph)

    ######################## 휴리스틱 알고리즘 #########################
    # 휴리스틱 1번
    # 트랜스포터의 대수 최소화
    # 작업량이 적은 트랜스포터 (a)에 있는 작업을 작업량이 많은 트랜스포터 (b)에 할당
    # 그 후, 트랜스포터 (b)는 새로운 작업의 공차거리를 최소화할 수 있는 순서에 작업 배치
    h1 = H1(task_work_time, task_empty_time, task_manager.task_list, graph=graph)
    # 새로운 스케줄이 생성 -> 휴리스틱이 돌아가는 것 -> 최종 결과

    # 휴리스틱 2번
    # 총 이동거리 최소화
    # 재배치할 작업의 선택 -> 공차거리가 긴 작업을 비교 탐색하여 선택 (c) or 임의의 작업 선택 (d, 효율)
    # 재배치 방법 -> 공차시간을 비교하여 가장 짧은 공차시간을 가질 수 있는 위치 배치 (x)
    # or 작업들 간의 우선순위 기반 현재 위치에서 공차 시간보다 짧은 공차시간 가지는 위치에 배치 (y, 안정적)
    h2 = H2(task_work_time, task_empty_time, task_manager.task_list, empty_speed)

    # 초기 세대 생성
    pop = Population(task_work_time, task_empty_time, trans_manager, task_manager, initialise=True)
    ga_random = GA(task_manager, task_work_time, task_empty_time)
    ga_base = GA(task_manager, task_work_time, task_empty_time, h1=h1)
    ga_distance = GA(task_manager, task_work_time, task_empty_time, h2=h2)

    # 결과 저장할 변수
    r_trans, rw_t, re_t, rtotal_time, re_d = 0, 0, 0, 0, 0
    b_trans, bw_t, be_t, btotal_time, be_d = 0, 0, 0, 0, 0
    d_trans, dw_t, de_t, dtotal_time, de_d = 0, 0, 0, 0, 0

    # 최적해 탐색                           #100
    for g_index, generation in enumerate(n_generations):
        pop_base = copy.deepcopy(pop)
        pop_random = copy.deepcopy(pop)
        pop_distance = copy.deepcopy(pop)

        random_f, random_s = 0, None
        base_f, base_s = 0, None
        distance_f, distance_s = 0, None

        for i in range(generation):
            ########### 휴리스틱 알고리즘 돌아가는 부분 ###########
            pop_random = ga_random.evolvePopulation(pop_random, trans_manager, random_flag=True)
            pop_base = ga_base.evolvePopulation(pop_base, trans_manager, base_flag=True)
            pop_distance = ga_distance.evolvePopulation(pop_distance, trans_manager, distance_flag=True)
            ##################################################

            print("{}/{}/{}".format(w_index, g_index, i / generation))

            ################## 평가함수 ####################
            random_pop = pop_random.getfittest()
            temp_f = random_pop.getfitness(work_time=task_work_time, empty_time=task_empty_time)
            if random_f < temp_f:
                random_s = copy.deepcopy(random_pop)
                random_f = temp_f

            base_pop = pop_base.getfittest()
            temp_f = base_pop.getfitness(work_time=task_work_time, empty_time=task_empty_time)
            if base_f < temp_f:
                base_s = copy.deepcopy(base_pop)
                base_f = temp_f

            distance_pop = pop_distance.getfittest()
            temp_f = distance_pop.getfitness(work_time=task_work_time, empty_time=task_empty_time, distance_flag=True)
            if distance_f < temp_f:
                distance_s = copy.deepcopy(distance_pop)
                distance_f = temp_f

        r_trans, rw_t, re_t, rtotal_time, re_d = random_s.gettrans_num_time(work_time=task_work_time,
                                                                            empty_time=task_empty_time)
        b_trans, bw_t, be_t, btotal_time, be_d = base_s.gettrans_num_time(work_time=task_work_time,
                                                                          empty_time=task_empty_time)
        d_trans, dw_t, de_t, dtotal_time, de_d = distance_s.gettrans_num_time(work_time=task_work_time,
                                                                              empty_time=task_empty_time)

        # 대수,  작업 시간, 공차 시간, 작업 + 공차, 공차 거리
        print("work_num: ", work_num)
        print("random  : ", format(int(r_trans), ','), format(int(rw_t), ','), format(int(re_t), ','), format(int(rtotal_time), ','), format(int(re_d), ','))
        print("base    : ", format(int(b_trans), ','), format(int(bw_t), ','), format(int(be_t), ','), format(int(btotal_time), ','), format(int(be_d), ','))
        print("distance: ", format(int(d_trans), ','), format(int(dw_t), ','), format(int(de_t), ','), format(int(dtotal_time), ','), format(int(de_d), ','))
print(time.time() - prev)
import time
import copy
import numpy as np

# 트랜스 포터 작업 개수, 염색체 세대 수, 트랜스 포터 작업 개수
from global_var import transporter_num, n_generations, task_num

from Data_create.call_data import object_data
from Data_create.create_trans import transporter_data
from Object.graph import Graph
from Object.Transporter import Trans_manager # , child_Trans_manager
from Task.Task import Task_schedule,task_classification
from Simulation import perform_graph
from Simulation.transporter_view import transporter_schedule_view
from Generation.Population import Population
from Generation.Gen import GA
from Heuristic.Heuristic_condition import H1, H2, H3

np.set_printoptions(threshold=np.inf, linewidth=np.inf) #inf = infinity

######################## 변수 로드 #########################
# 데이터 불러오기
stock_data, inter_data, road_data = object_data()

# 그래프 생성
graph = Graph(stock_data, inter_data, road_data)
print("Map Complete!!")

# 트랜스포터 목록 들고 오기
trans_manager = Trans_manager()
transporter_data(transporter_num, trans_manager, graph)
print("Transporter Complete!!")

##########################################################

time_perform = []
trans_perform, dis_perform, c = [], [], 0
                                  #100, 200, 300
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
    h1 = H1(task_work_time, task_empty_time, task_manager.task_list, graph=graph)
    # 휴리스틱 2번
    h2 = H2(task_work_time, task_empty_time, task_manager.task_list, empty_speed)
    # 휴리스틱 3번
    h3 = H3(task_work_time, task_empty_time, task_manager.task_list)
    #################################################################

    # 초기 세대 생성
    pop = Population(task_work_time, task_empty_time, trans_manager, task_manager, initialise=True)
    ga_random = GA(task_manager, task_work_time, task_empty_time)
    ga_base = GA(task_manager, task_work_time, task_empty_time, h1=h1)
    ga_distance = GA(task_manager, task_work_time, task_empty_time, h2=h2)
    ga_gen = GA(task_manager, task_work_time, task_empty_time, h3=h3)
    # ga_bd = GA(task_manager, task_work_time, task_empty_time, h1=h1, h2=h2)

    # 결과 저장할 변수
    r_trans, rw_t, re_t, rtotal_time, re_d = 0, 0, 0, 0, 0
    b_trans, bw_t, be_t, btotal_time, be_d = 0, 0, 0, 0, 0
    d_trans, dw_t, de_t, dtotal_time, de_d = 0, 0, 0, 0, 0
    g_trans, gw_t, ge_t, gtotal_time, ge_d = 0, 0, 0, 0, 0
    # bd_trans, bdw_t, bde_t, bdtotal_time, bde_d = 0, 0, 0, 0, 0

    # 최적해 탐색
    for g_index, generation in enumerate(n_generations):
        pop_base = copy.deepcopy(pop)
        pop_random = copy.deepcopy(pop)
        pop_distance = copy.deepcopy(pop)
        pop_gen = copy.deepcopy(pop)
        # pop_base_distance = copy.deepcopy(pop)

        random_f, random_s = 0, None
        base_f, base_s = 0, None
        distance_f, distance_s = 0, None
        gen_f, gen_s = 0, None
        # bd_f, bd_s = 0, None

        for i in range(generation):
            ########### 휴리스틱 알고리즘 돌아가는 부분 ###########
            # pop_random = ga_random.evolvePopulation(pop_random, trans_manager, random_flag=True)
            # pop_base = ga_base.evolvePopulation(pop_base, trans_manager, base_flag=True)
            # pop_distance = ga_distance.evolvePopulation(pop_distance, trans_manager, distance_flag=True)
            pop_gen = ga_gen.evolvePopulation(pop_gen, trans_manager, gen_flag=True)
            # pop_base_distance = ga_bd.evolvePopulation(pop_base_distance, trans_manager, bd_flag=True)
            ##################################################

            print("하는중: {}/{}/{}".format(w_index, g_index, i / generation))
            
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

            # base_distance_pop = pop_base_distance.getfittest()
            # temp_f = base_distance_pop.getfitness(work_time=task_work_time, empty_time=task_empty_time,
            #                                       distance_flag=True)

            ################## 평가함수 ####################

            # if bd_f < temp_f:
            #     bd_s = copy.deepcopy(base_distance_pop)
            #     bd_f = temp_f
            # d_trans, dw_t, de_t, dtotal_time = distance_s.gettrans_num_time(work_time=task_work_time,
            #                                                                 empty_time=task_empty_time)
        # r_time = round(float(np.mean(r_time)),4)
        # b_time = round(float(np.mean(b_time)),4)
        r_trans, rw_t, re_t, rtotal_time, re_d = random_s.gettrans_num_time(work_time=task_work_time, empty_time=task_empty_time)
        b_trans, bw_t, be_t, btotal_time, be_d = base_s.gettrans_num_time(work_time=task_work_time, empty_time=task_empty_time)
        d_trans, dw_t, de_t, dtotal_time, de_d = distance_s.gettrans_num_time(work_time=task_work_time, empty_time=task_empty_time)
        # bd_trans, bdw_t, bde_t, bdtotal_time = bd_s.gettrans_num_time(work_time=task_work_time, empty_time=task_empty_time)

        # time_perform.append([r_time, b_time])
        # trans_perform.append([r_trans, b_trans])
        # dis_perform.append([r_dis, b_dis])

    print("work_num: ", work_num)
    print("random  : ", r_trans, rw_t, re_t, rtotal_time, re_d)
    print("base    : ", b_trans, bw_t, be_t, btotal_time, be_d)
    print("distance: ", d_trans, dw_t, de_t, dtotal_time, de_d)
    # print("b_d     : ", bd_trans, bdw_t, bde_t, bdtotal_time)
    print("")
# 결과지
# perform_graph.perform_graph(time_perform, trans_perform, dis_perform)


# 그래프
# transporter_schedule_view(base_s.trans_m, 'base')

# random_f, random_s = 0, None
# h1_f, h1_s = 0, None
# for i in range(n_generations):
#     pop_random = ga_random.evolvePopulation(pop_random, trans_manager, random_flag=True)
#     pop_h1 = ga_h1.evolvePopulation(pop_h1, trans_manager, nv_flag=True)
#     print("하는중: {}".format(i/n_generations))
#
#     random_pop = pop_random.getfittest()
#     if random_f < random_pop.getfitness():
#         random_s = copy.deepcopy(random_pop)
#         random_f = random_pop.getfitness()
#
#     h1_pop = pop_h1.getfittest()
#     if h1_f < h1_pop.getfitness():
#         h1_s = copy.deepcopy(h1_pop)
#         h1_f = h1_pop.getfitness()
#
#     print("random    :", random_pop.gettrans_num_distance())
#     print("h1        :", h1_pop.gettrans_num_distance())
#     print("pop_random:", random_s.gettrans_num_distance())
#     print("pop_h1    :", h1_s.gettrans_num_distance())

# 그래프
# transporter_schedule_view(random_s.trans_m, 'random')
# transporter_schedule_view(h1_s.trans_m, 'h1')


# # 영상
# # map_view(graph, result_manager)
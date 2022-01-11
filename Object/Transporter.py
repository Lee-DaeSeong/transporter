import time, datetime
import copy
import numpy as np
from global_var import init_position, work_stime, work_etime, load_time, unload_time


class Transporter:
    def __init__(self, no, size, availble_weight, w_speed, e_speed, t_speed, graph, color, task=None, total_data=None, total_time=None, total_distance=None):
        self.no = no
        self.size = size                    # 트랜스포터 크기
        self.a_w = availble_weight    # 트랜스포터 가용 무게
        self.w_s = w_speed                  # 작업 속도
        self.e_s = e_speed                  # 공차 속도
        self.t_s = t_speed                  # 회전 속도
        self.task = []                      # 할당된 작업 목록
        self.total_time = 0
        self.total_distance = 0
        self.total_data = []  # 각 작업별 정보
        self.graph = graph
        self.color = color

        if task is not None:
            self.task = task
        if total_data is not None:
            self.total_data = total_data
        if total_distance is not None:
            self.total_distance = total_distance

        # state = 0: 그냥 대기 상태 / 1:공차 상태 / 2:상차 상태 / 3: 작업 상태 / 4: 하차 상태
        # self.total_data의 길이는 작업 수만큼이고 각 작업 리스트 별 (출발 시각, 도착 시각, 출발 노드, 도착 노드, 사용 도로, 속도, 상태)의 원소가 있음

        self.simul_task = 0  # 시뮬레이션 용 작업 인덱스
        self.simul_i = 0     # 시뮬레이션 용

    def __deepcopy__(self, memodict={}):
        task, total_data = self.task.copy(), self.total_data.copy()
        trans = Transporter(self.no, self.size, self.a_w, self.w_s, self.e_s, self.t_s, self.graph, self.color, task=task, total_time= self.total_time, total_distance=self.total_distance)
        return trans

    def undertake_task(self, empty_time, work_time, task=None, position=None, random_flag=False, base_flag=False):
        if random_flag:
            no = task.task_no()
            empty_time = empty_time[self.e_s]
            w_t, e_t = None, None
            if len(self.task) == 0:
                e_t = empty_time[-1][no] + empty_time[no][no]
            else:
                e_t = empty_time[self.task[-1].task_no()][no] + empty_time[no][no]
            w_t = work_time[self.w_s][no]

            # 작업 추가
            self.task.append(task)
            self.total_time = self.total_time + w_t + e_t

        elif base_flag:
            self.total_time = 0
            self.task.insert(position, task)
            no, w_t, e_t = 0, 0, 0
            for i, t in enumerate(self.task):
                no = t.task_no()
                w_t += work_time[self.w_s][no]

                if i == 0:
                    e_t += empty_time[self.e_s][-1][no]
                else:
                    e_t += empty_time[self.e_s][self.task[i-1].task_no()][no]

            e_t += empty_time[self.e_s][no][no]

            self.total_time = self.total_time + w_t + e_t

    # 특정 작업의 공차(+작업) 시간
    def task_empty_time(self, empty_time, task, work_flag=False, work_time=None):
        no = task.task_no()
        task_index = self.task.index(task)
        pre, next, work = 0, 0, 0

        if task_index == 0:
            pre = empty_time[self.e_s][-1][no]
        else:
            pre = empty_time[self.e_s][self.task[task_index-1].task_no()][no]
        if task_index == len(self.task) - 1:
            next = empty_time[self.e_s][no][no]
        else:
            next = empty_time[self.e_s][no][self.task[task_index+1].task_no()]

        if work_flag:
            work = work_time[self.w_s][no]

        result = pre + next + work
        return result
    
    # 특정 작업의 앞뒤 작업 간 공차 거리
    def task_other_time(self, empty_time, task):
        no = task.task_no()
        task_index = self.task.index(task)

        if task_index == 0:
            pre = -1
        else:
            pre = self.task[task_index-1].task_no()

        if task_index == len(self.task) - 1:
            next = no
        else:
            next = self.task[task_index+1].task_no()

        result = 0
        # 작업 X인 경우
        if pre == -1 and next == no:
            return result
        # 앞의 작업 뒤에 복귀하는 경우
        elif next == no:
            result = empty_time[self.e_s][pre][pre]
        else:
            result = empty_time[self.e_s][pre][next]

        return result

    # 특정 작업 삭제 시
    def remove_task(self, task, empty_time, work_time):
        self.task.remove(task)

        no, w_t, e_t = 0, 0, 0
        for i, t in enumerate(self.task):
            no = t.task_no()
            w_t += work_time[self.w_s][no]

            if i == 0:
                e_t += empty_time[self.e_s][-1][no]
            else:
                e_t += empty_time[self.e_s][self.task[i-1].task_no()][no]

        e_t += empty_time[self.e_s][no][no]
        self.total_time = w_t + e_t

    # 현재 작업들 걸리는 시간
    def cal_time(self, work_time, empty_time):
        if len(self.task) == 0:
            self.total_time = 0
            return 0, 0

        w_t, e_t = 0, 0

        work_t = work_time[self.w_s]
        empty_t = empty_time[self.e_s]

        pre_no = None
        for i, task in enumerate(self.task):
            no = task.task_no()

            if i == 0:
                e_t = e_t + empty_t[-1][no]
            else:
                e_t = e_t + empty_t[pre_no][no]

            w_t = w_t + work_t[no]
            pre_no = no

        e_t = e_t + empty_t[pre_no][pre_no]
        self.total_time = w_t + e_t

        return w_t, e_t

    # 작업들 거리
    def cal_distance(self):
        if len(self.task) == 0:
            return

        self.total_distance = 0
        pre, e_d = None, None

        for i, task in enumerate(self.task):
            if i == 0:
                e_d = self.graph.distance_node(0, init_position, task.start_p())
            else:
                e_d = self.graph.distance_node(0, pre, task.start_p())
            w_d = self.graph.distance_node(0, task.start_p(), task.dest_p())
            pre = task.start_p()
            self.total_distance = self.total_distance + w_d + e_d

        e_d = self.graph.distance_node(0, pre, init_position)
        self.total_distance = self.total_distance + e_d

    # 무게 조건
    def weight_condition(self, task_weight):
        if self.a_w >= task_weight:
            return True
        return False

    # 시간 조건
    def time_condition(self, empty_time, work_time, task, i=None, chromo=False):
        work_t = work_time[self.w_s]
        empty_t = empty_time[self.e_s]

        task_no = task.task_no()
        w_t = work_t[task_no]
        e_t, a_t = 0, 0

        # 현재는 H2에서 쓰임
        if i is not None:
            if i == 0:
                e_t += empty_t[-1][task_no]
                a_t += empty_t[-1][self.task[i].task_no()]
            else:
                e_t += empty_t[self.task[i-1].task_no()][task_no]
                a_t += empty_t[self.task[i-1].task_no()][self.task[i].task_no()]

            if i == len(self.task) - 1:
                e_t += empty_t[task_no][task_no]
            else:
                e_t += empty_t[task_no][self.task[i].task_no()]

        # 나머지
        else:
            if len(self.task) == 0:
                e_t += empty_t[-1][task_no]
                e_t += empty_t[task_no][task_no]
            else:
                e_t += empty_t[self.task[-1].task_no()][task_no]
                e_t += empty_t[task_no][task_no]

        if work_stime + self.total_time + w_t + e_t - a_t <= work_etime:
            return True
        return False

    # 작업 초기화
    def reset_data(self):
        self.task = []
        self.total_data = []
        self.total_time = 0
        self.total_distance = 0

    # 특정 작업의 스케줄 반환
    def get_data(self, task):
        index = self.task.index(task)
        return self.total_data[index]

    # 작업을 시작한 시각 / 노드 리스트 / 작업 출발지 / 작업 도착지 (작업 할당 함수)
    # def setTask(self, start_time, node_list, task, h_flag=False):
    #     if not h_flag:
    #         self.task.append(task)
    #
    #     task_d = task.start_p()
    #     task_data = []
    #
    #     state = 1
    #     speed = self.e_s
    #     s_t, e_t = start_time, start_time
    #
    #     road_list = self.search_road(node_list)
    #
    #     for index, node in enumerate(node_list):
    #         road, s_n, e_n = None, None, None
    #
    #         # 하차 노드 전임
    #         if index != len(node_list) - 1:
    #             road = road_list[index]
    #             s_n, e_n = node, node_list[index + 1]
    #
    #             # 출발 노드가 상차지에서라면..
    #             if node == task_d:
    #                 data = (s_t, s_t + load_time, node, node, None, 0, 2)
    #                 task_data.append(data)
    #                 s_t += load_time
    #                 speed = self.w_s
    #                 state = 3
    #
    #             # 시간 구하는 거하고
    #             e_t = self.cal_time(s_t, road, speed)
    #             data = (s_t, e_t, s_n, e_n, road, speed, state)
    #             self.total_distance = self.total_distance + road.d
    #
    #         # 하차 노드임
    #         else:
    #             data = (s_t, s_t + unload_time, node, node, None, 0, 4)
    #
    #         task_data.append(data)
    #         s_t = e_t
    #
    #     self.total_data.append(task_data)
    #
    # # 사용 도로 탐색 함수 ( 도로 리스트 반환)
    # def search_road(self, node_list):
    #     road_list = [None for _ in range(len(node_list) - 1)]
    #
    #     f1 = False
    #     f2 = False
    #     for road in self.graph.road_list:
    #         for index, node in enumerate(node_list):
    #             index -= 1
    #             if road.s.no == node:
    #                 f1 = True
    #                 if f2 is True:
    #                     road_list[index] = road
    #
    #             elif road.e.no == node:
    #                 f2 = True
    #                 if f1 is True:
    #                     road_list[index] = road
    #
    #             else:
    #                 f1 = False
    #                 f2 = False
    #     if None in road_list:
    #         print("node_list: ", node_list)
    #     return road_list

    # 도로 출발 시각과 끝 시각 구하는 함수 ( 출발 시각, 도착 시각 반환)
    # def cal_time(self, start_time, road, speed):
    #     range_time = road.d // speed
    #     end_time = start_time + range_time
    #     return end_time

    # 현재 위치 반환
    def cur_p(self, simul_flag=False):
        if not simul_flag:
            if len(self.task) == 0:
                return init_position
            else:
                return self.task[-1].dest_p()
                # return self.total_data[-1][-1][3]

    # 작업 가능한 시각 반환
    def a_time(self):
        if len(self.task) == 0:
            return work_stime
        else:
            return self.total_time
            # return self.total_data[-1][-1][1]

    # 작업당 공차 and 작업 시간 구간 반환 --> 현재 trans_view에서만 사용중
    def t_time(self, index):
        est = self.total_data[index][0][0]
        eet = None
        wst = None
        wet = self.total_data[index][-1][1]
        for i, data in enumerate(self.total_data[index]):
            if self.total_data[index][i + 1][6] != 1:
                eet = data[1]
                wst = self.total_data[index][i + 1][0]
                break
        wst += 1
        est += 1

        # 데이터 타입으로 변환기 --> trans_view에서 그래프 x축 데이터 변환용
        def convert_datetime(est, eet, wst, wet):
            est = datetime.timedelta(seconds=int(est))
            est = datetime.datetime.strptime(str(est), "%H:%M:%S")
            eet = datetime.timedelta(seconds=int(eet))
            eet = datetime.datetime.strptime(str(eet), "%H:%M:%S")
            wst = datetime.timedelta(seconds=int(wst))
            wst = datetime.datetime.strptime(str(wst), "%H:%M:%S")
            wet = datetime.timedelta(seconds=int(wet))
            wet = datetime.datetime.strptime(str(wet), "%H:%M:%S")

            return est, eet, wst, wet

        est, eet, wst, wet = convert_datetime(est, eet, wst, wet)

        empty = [est, eet]
        work = [wst, wet]
        return empty, work


class Trans_manager:

    def __init__(self, t_list=None):
        if t_list is None:
            self.t_list = []
        else:
            self.t_list = t_list

    def __repr__(self):
        genestr = "-------------------------- Trans --------------------------\n"
        for t in self.t_list:
            genestr += "{}({})\t:".format(t.no, t.a_w)
            task_list = t.task
            if len(task_list) == 0:
                genestr += " None\n"
            else:
                for i, task in enumerate(task_list):
                    if i != len(task_list) - 1:
                        genestr += " {}({}) ->".format(task.task_no(), task.weight())
                genestr += " {}({})\n".format(task_list[-1].task_no(), task_list[-1].weight())

        genestr += "-------------------------- END --------------------------\n"

        return genestr

    def __deepcopy__(self, memodict={}):
        t_list = []
        for t in self.t_list:
            t = copy.deepcopy(t)    # t.__deepcopy__()도 가능
            t_list.append(t)

        trans_manager = Trans_manager(t_list=t_list)
        return trans_manager

    def add_trans(self, trans):
        self.t_list.append(trans)

    # 모든 트랜스포터 작업 초기화
    def pop_trans_task(self):
        for trans in self.t_list:
            trans.reset_data()

    def numoftrans(self):
        return len(self.t_list)

    # 트랜스포터 얻기
    def get_trans(self, trans):
        for t in self.t_list:
            if trans.no == t.no:
                return t
        return None

    # 트랜스포터 사용 대수, 총 이동거리 반환
    def total_trans_data(self, work_time, empty_time):
        num = 0
        empty_distance = 0
        work, empty, total_time = 0, 0, 0
        count = 0
        for trans in self.t_list:
            count+=1
            w_t, e_t = trans.cal_time(work_time, empty_time)
            trans.cal_distance()
            empty_distance += trans.total_distance
            if trans.total_time != 0:
                num = num + 1
                work += w_t
                empty += e_t
                total_time += trans.total_time

        return num, work, empty, total_time, empty_distance

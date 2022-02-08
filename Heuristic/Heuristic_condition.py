import numpy as np
import random, copy, traceback
import inspect
from global_var import work_stime, work_etime, cross_num, mutation


# 첫 번째 규칙 클래스
class H1:
    def __init__(self, task_wtime, task_etime, task_list, graph):
        self.task_empty_time = task_etime    # 작업 간 공차시간
        self.task_work_time = task_wtime   # 작업 시간
        self.task_list = task_list      # 전체 작업 목록
        self.graph = graph
    
    # 작업 인덱스 추출
    def task_index(self, task):
        return self.task_list.index(task)
    
    # 특정 트랜스포터 내에 task에서 현재 변경하려는 task를 넣을 수 있는 곳 추출
    def trans_list_task(self, trans, task):
        task_list = trans.task
        if len(task_list) == 0:
            return 0, None

        no = task.task_no()

        p_list = []
        rulat_list = []

        for i in range(len(task_list)+1):

            # 임시 할당
            trans.undertake_task(self.task_empty_time, self.task_work_time, task=task, position=i, base_flag=True)
            # print("2")
            # 시간 조건 X이면 패스
            if trans.total_time + work_stime > work_etime:
                trans.remove_task(task, self.task_empty_time, self.task_work_time)
                continue
            
            # 작업 공차 시간
            e_t = trans.task_empty_time(self.task_empty_time, task)

            p_list.append(i)
            if e_t == 0:
                rulat_list.append(1)
            else:
                rulat_list.append(1/e_t)
            
            # 임시로 할당했던 작업 삭제
            trans.remove_task(task, self.task_empty_time, self.task_work_time)

        return p_list, rulat_list


# 두 번째 규칙 클래스
class H2:
    def __init__(self, task_wtime, task_etime, task_list, empty_speed):
        self.task_work_time = task_wtime
        self.task_empty_time = task_etime
        self.task_list = task_list

        self.priority_empty_list = self.priority(empty_speed)

    def priority(self, empty_speed):
        temp_empty_time = []

        for i, empty in enumerate(self.task_empty_time[empty_speed[0]]):
            # empty = 일차원 리스트임 = 작업 i번쨰의 공차 거리 리스트
            set_empty = sorted(set(empty))
            np_empty = np.array(empty)

            dict_list = {}
            for j in set_empty:
                index_list = np.where(np_empty == j)[0]
                dict_list[int(j)] = list(index_list)

            temp_empty_time.append(dict_list)
        return temp_empty_time

    def change_taskposition(self, trans_list):
        count = 0
        change_count = 0
        while True:
            base_time = 0
            random_task_index = random.randint(0, len(self.task_list) - 1)
            random_task = self.task_list[random_task_index]

            transporter = None

            for trans in trans_list:
                if random_task in trans.task:
                    transporter = trans
                    break
                    
            # 기본 시간
            base_time = transporter.task_empty_time(self.task_empty_time, random_task, work_flag=True, work_time=self.task_work_time)
            # ---------------------------------------------------------> 이전 값 추출 완료

            # 트랜스포터에 작업 삭제
            a_time = transporter.task_other_time(self.task_empty_time, random_task)     # 랜덤 작업을 제거했을 때, 앞 뒤 작업 간 공차시간
            transporter.remove_task(random_task, self.task_empty_time, self.task_work_time)

            # 무게 제약사항에 맞는 트랜스포터
            can_trans_list = []
            for trans in trans_list:
                w_flag = trans.weight_condition(random_task.weight())
                if w_flag:
                    can_trans_list.append(trans)
                
            # 이전보다 나은 자리 추출
            trans_q = {}
            for trans in can_trans_list:
                new_time = 0

                # 기존에 작업이 없는 트랜스포터
                if len(trans.task) == 0:
                    trans.undertake_task(self.task_empty_time, self.task_work_time, random_task
                                         , position=0, base_flag=True)
                    new_time = trans.task_empty_time(self.task_empty_time, random_task, work_flag=True, work_time=self.task_work_time)
                    new_time += a_time

                    if self.good_result(new_time, base_time, trans):
                        if new_time not in trans_q:
                            trans_q[new_time] = [[trans, 0]]
                        else:
                            trans_q[new_time].append([trans, 0])
                    trans.remove_task(random_task, self.task_empty_time, self.task_work_time)

                else:
                    for i in range(0, len(trans.task)+1):
                        trans.undertake_task(self.task_empty_time, self.task_work_time, random_task
                                             , i, base_flag=True)
                        b_time = trans.task_other_time(self.task_empty_time, random_task)
                        new_time = trans.task_empty_time(self.task_empty_time, random_task, work_flag=True,
                                                         work_time=self.task_work_time)
                        new_time += a_time
                        base_time += b_time

                        if self.good_result(new_time, base_time, trans):
                            if new_time not in trans_q:
                                trans_q[new_time] = [[trans, i]]
                            else:
                                trans_q[new_time].append([trans, i])

                        base_time -= b_time
                        trans.remove_task(random_task, self.task_empty_time, self.task_work_time)

            if len(trans_q) == 0:
                count += 1
                if count == 10:
                    break
                continue

            transporter_list = trans_q[min(trans_q)]
            num = len(transporter_list[0][0].task)
            tp = transporter_list[0]
            for t in transporter_list:
                if num < len(t[0].task):
                    num = len(t[0].task)
                    tp = t

            transporter_, position_ = tp
            transporter_.undertake_task(self.task_empty_time, self.task_work_time, random_task
                                 , position_, base_flag=True)

            change_count += 1
            if change_count == 5:
                break

    def good_result(self, value, base, transporter):
        if value <= base:
            w_t, e_t = transporter.cal_time(self.task_work_time, self.task_empty_time)
            if work_stime + w_t + e_t <= work_etime:
                return True
        return False

    def find_values(self, i1, i2):
        for key, values in self.priority_empty_list[i1].items():
            if i2 in values:
                return key


# 세 번째 유전 클래스
class H3:
    def __init__(self, task_wtime, task_etime, task_list):
        self.task_work_time = task_wtime
        self.task_empty_time = task_etime
        self.task_list = task_list
        self.pass_task = []

    # dir(pop)
    # ['__class__', '__deepcopy__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
    #  '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__',
    #  '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__',
    #  '__subclasshook__', '__weakref__', 'compatition', 'empty_time', 'getfittest', 'getschedule', 'getuse_trans',
    #  'populationsize', 'reset', 'saveschedule', 'schedules', 'work_time']


    def create_gen(self, pop):
        pop.schedules = pop.compatition(distance_flag=True)
        new_pop = copy.deepcopy(pop)
        new_pop.reset()

        for i in range(cross_num):
            parent_1 = pop[i]           # 오류

            sub_iter = cross_num - i - 1
            for j in range(sub_iter):
                parent_2 = pop[j+i+1]

                self.calcul_gen(parent_1, parent_2)

    # 계산
    def calcul_gen(self, p1, p2):
        p1_trans = p1.trans_m.t_list
        p2_trans = p2.trans_m.t_list

        p1_task_et = {}
        p2_task_et = {}

        ###### 공차시간 기준으로 정렬
        for task in self.task_list:
            for t in p1_trans:
                if task in t.task:
                    empty_time = t.task_other_time(self.task_empty_time, task)
                    if empty_time in p1_task_et:
                        p1_task_et[empty_time].append(task)
                    else:
                        p1_task_et[empty_time] = [task]

            for t in p2_trans:
                if task in t.task:
                    empty_time = t.task_other_time(self.task_empty_time, task)
                    if empty_time in p2_task_et:
                        p2_task_et[empty_time].append(task)
                    else:
                        p2_task_et[empty_time] = [task]

        p1_task_et = dict(sorted(p1_task_et.items()))
        p2_task_et = dict(sorted(p2_task_et.items()))

        ###### 공차시간 기준으로 정렬 끝

        # 자식 생성 #
        while True:
            key = random.random()

            # 교차
            if key > mutation:
                self.cross(p1_trans, p2_trans, p1_task_et, p2_task_et)

            # 변이
            else:
                self.mutation(p1_trans, p2_trans, p1_task_et, p2_task_et)



    # 교차 연산
    def cross(self, p1_trans, p2_trans, p1_t, p2_t):
        t1 = list(p1_t.keys())[0]
        t2 = list(p2_t.keys())[0]
        task, trans = None, None
        # p1의 공차시간에 있는 작업 선택
        if t1 <= t2:
            task = p1_t[t1][0]
        else:
            task = p2_t[t2][0]

        pass

    # 변이 연산
    def mutation(self, p1, p2, p1_t, p2_t):
        pass

    # 할당
    def allocation(self, task, trans):

        for t in trans:
            if task in t.task:
                # 할당할 트랜스포터 스펙 뽑기
                t.e_s




    # def select_transporter(self, child, task, parent1, parent2, mode=None):
    #     trans1 = self.find_trans(task, parent1.trans_m.t_list)
    #     trans2 = self.find_trans(task, parent2.trans_m.t_list)
    #
    #     r1 = trans1.task_empty_time(self.task_empty_time, task, work_flag=True, work_time=self.task_work_time)
    #     r2 = trans2.task_empty_time(self.task_empty_time, task, work_flag=True, work_time=self.task_work_time)
    #
    #     select_trans = None
    #     if r1 <= r2:
    #         select_trans = self.child_trans(trans1, child)
    #     else:
    #         select_trans = self.child_trans(trans2, child)
    #
    #     pass_trans = []
    #
    #     # 연관된 작업들 추출
    #     task_set = [task]
    #     task_index = select_trans.task.index(task)
    #     if task_index > 0:
    #         task_set.insert(0, select_trans.task[task_index-1])
    #     if task_index < len(select_trans.task) - 1:
    #         task_set.insert(-1, select_trans.task[task_index+1])
    #
    #     # 할당하기
    #     assignment_flag = True
    #     # 시간제약사항 X인 경우 해당 트랜스포터는 X
    #     if not self.cal_time(task_set, select_trans):
    #         pass_trans.append(select_trans)
    #         # select_trans = self.another_child_trans(child, pass_trans)
    #
    #     for task in task_set:
    #         # 제약사항
    #         if not (select_trans.weight_condition(task.weight())):
    #
    #         # 성공
    #         else:
    #             self.pass_task.append(task)
    #
    # def cal_time(self, task_set, trans):
    #     w_t = self.task_work_time[trans.w_s]
    #     e_t = self.task_empty_time[trans.e_s]
    #
    #     total_time = 0
    #     case1, case2 = 0, 0
    #     if len(trans.task)==0:
    #         total_time = e_t[-1][task_set[0].task_no()]
    #         total_time += e_t[task_set[-1].task_no()][task_set[-1]]
    #     else:
    #         case1 = e_t[-1][task_set[0].task_no()]
    #         case1 += e_t[task_set[-1].task_no()][trans.task[0].task_no()]
    #
    #         case2 = e_t[trans.task[-1].task_no()][task_set[0].task_no()]
    #         case2 += e_t[task_set[-1].task_no()][task_set[-1].task_no()]
    #
    #         total_time = case1
    #         if case1 > case2:
    #             total_time = case2
    #
    #     for i, task in enumerate(task_set):
    #         total_time += w_t[task.task_no()]
    #
    #         if i != 0:
    #             total_time += e_t[task_set[i-1].task_no()][task_set[i].task_no()]
    #
    #     if work_stime + trans.total_time + total_time <= work_etime:
    #         return True
    #     return False
    #
    # # def another_child_trans(self, child, pass_trans):
    # #     another_trans_list = []
    # #     for trans in child.trans_m:
    # #         if trans.a_w == pass_trans[0].a_w and trans not in pass_trans:
    # #             another_trans_list.append(trans)
    # #
    # #     another_trans = another_trans_list[0]
    # #     for trans in another_trans_list:
    # #         if len(another_trans.task) < len(trans.task):
    # #             another_trans = trans
    # #
    # #     return another_trans
    #
    #
    # def child_trans(self, trans, child):
    #     for child_trans in child.trans_m:
    #         if trans.no == child_trans.no:
    #             return child_trans
    #
    # def find_trans(self, task, trans_list):
    #     for trans in trans_list:
    #         if task in trans.task:
    #             return trans_list_task
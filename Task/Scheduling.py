import random, copy


class Schedule:
    def __init__(self, trans_m=None, task_m=None):
        self.trans_m = trans_m
        self.task_m = task_m

    def __deepcopy__(self, memodict={}):
        trans_m = copy.deepcopy(self.trans_m)
        schedule = Schedule(trans_m, self.task_m)

        return schedule

    # 작업 할당_init
    def allocation_task(self, work_time, empty_time):
        while True:
            flag = True
            for task in self.task_m.task_list:
                t_list = []
                # 무게 & 시간 제약 사항
                for trans in self.trans_m.t_list:
                    w_flag = trans.weight_condition(task.weight())
                    t_flag = trans.time_condition(empty_time, work_time, task)

                    if w_flag and t_flag:
                        t_list.append(trans)

                if len(t_list) == 0:
                    flag = False
                    self.trans_m.pop_trans_task()
                    break

                i = random.randint(0, len(t_list)-1)
                trans = t_list[i]   # 실제 작업 할당될 트랜스포터

                # 트랜스포터에 작업 할당
                trans.undertake_task(empty_time, work_time, task=task, random_flag=True)

            if flag:
                break
    
    # 평가함수
    def getfitness(self, work_time, empty_time, distance_flag=False):
        result = None
        # 사용 트랜스포터, 총 작업시간, 총 공차시간, 총 시간
        trans_num, w_t, e_t, total_time, empty_distance = self.trans_m.total_trans_data(work_time=work_time, empty_time=empty_time)

        if distance_flag:
            result = 1 / (e_t + trans_num)
        else:
            result = 1 / (10000000 * trans_num + total_time)

        return result

    # 해당 스케줄에서 쓰이는 트랜스포터 대수와 총 시간
    def gettrans_num_time(self, work_time, empty_time):
        trans_num, w_t, e_t, total_time, empty_distance = self.trans_m.total_trans_data(work_time=work_time, empty_time=empty_time)

        return trans_num, w_t, e_t, total_time, empty_distance

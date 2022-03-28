from global_var import populationSize
from Generation.Population import Population
from Task.Scheduling import Schedule
from Heuristic.rulat import trans_rulat, task_rulat

import time, random, traceback
from global_var import mutation


class GA:
    def __init__(self, task_manager, work_time, empty_time, h1=None, h2=None, h3=None, mutationrate=0.05, tournamentsize=5, e=False):
        self.task_m = task_manager
        self.work_time = work_time
        self.empty_time = empty_time

        self.h1 = h1
        self.h2 = h2
        self.h3 = h3

        self.mutationRate = mutationrate
        self.tournamentSize = tournamentsize
        self.elitism = e

    def evolvePopulation(self, pop, trans_manager, base_flag=False, random_flag=False, distance_flag=False, gen_flag=False, bd_flag=False, nv_flag=False):
        newPopulation = None

        # 베이스라인
        if base_flag:
            newPopulation = self.base_(pop)

        # 랜덤
        if random_flag:
            newPopulation = Population(work_time=self.work_time, empty_time=self.empty_time, trans_manager=trans_manager, task_manager=self.task_m, initialise=True)

        # 이동거리 최소화 모델
        if distance_flag:
            newPopulation = self.distance_(pop)

        # 베이스 + 이동거리
        if bd_flag:
            newPopulation = self.base_(pop)
            newPopulation = self.distance_(newPopulation)

        # 유전
        if gen_flag:
            newPopulation = self.gen_(pop)

        # 이웃 탐색으로
        if nv_flag:
            newPopulation = self.hu_v1(pop)


        # # 가장 좋은거는 다시 유전
        # if self.elitism:
        #     if great_schdule.getfitness() > newPopulation.schedules[0].getfitness():
        #         newPopulation.schedules.saveschedule(0, great_schdule)

        return newPopulation

    def base_(self, pop):
        for s in pop.schedules:
            try:
                for task in self.task_m.task_list:
                    # 원래 위치에서 삭제
                    self.pretask_remove(task, s.trans_m.transporter_list)
                    trans_con, rulat_trans = trans_rulat(s.trans_m.transporter_list, task.b.w)
                    while True:
                        # 트랜스포터 할당 (룰렛휠)
                        index = rulat_trans.index(max(rulat_trans))
                        trans_d = trans_con[index]

                        task_index_con, rulat_task = self.h1.trans_list_task(trans_d, task)

                        # 기존의 작업 존재 X
                        if task_index_con == 0:
                            trans_d.undertake_task(self.empty_time, self.work_time, task=task, position=0,
                                                   base_flag=True)
                            break

                        elif len(task_index_con) != 0:
                            index = rulat_task.index(max(rulat_task))
                            task_index_d = task_index_con[index]
                            trans_d.undertake_task(self.empty_time, self.work_time, task=task, position=task_index_d, base_flag=True)
                            break

                        else:
                            trans_con.remove(trans_d)
                            rulat_trans.remove(max(rulat_trans))

            except:
                print(s.trans_m)
                traceback.print_stack()
                traceback.print_exc()
                exit()

        return pop

    def distance_(self, pop):
        for s in pop.schedules:
            try:
                self.h2.change_taskposition(s.trans_m.transporter_list)

            except:
                traceback.print_stack()
                traceback.print_exc()
                exit()

        return pop

    def gen_(self, pop):
        try:
            self.h3.create_gen(pop)
        except:
            traceback.print_stack()
            traceback.print_exc()
            exit()

        return pop

    def print_pop(self, pop):
        for s in pop.schedules:
            for t in s.trans_m.transporter_list:
                print(t.task)
                print(t.total_data)
            break

    # 원래 위치하던 곳에서 삭제
    def pretask_remove(self, task, trans_list):
        for t in trans_list:
            if task in t.task:
                t.remove_task(task, self.empty_time, self.work_time)
                break


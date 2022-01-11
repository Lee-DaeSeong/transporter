import copy
from global_var import populationSize, cross_num
from Task.Scheduling import Schedule
from Object.Transporter import Trans_manager


class Population:
    def __init__(self, work_time=None, empty_time=None, trans_manager=None, task_manager=None, schedules=None, initialise=False):
        self.schedules = [None for _ in range(populationSize)]
        self.work_time = work_time
        self.empty_time = empty_time

        if initialise:
            for i in range(0, populationSize):
                t_m = copy.deepcopy(trans_manager)
                newSchedule = Schedule(t_m, task_manager)
                newSchedule.allocation_task(work_time, empty_time)
                self.saveschedule(i, newSchedule)
        else:
            self.schedules = schedules

    def __deepcopy__(self, memodict={}):
        schedules = []
        for i in range(0, self.populationsize()):
            schedules.append(copy.deepcopy(self.schedules[i]))

        population = Population(work_time=self.work_time, empty_time=self.empty_time, schedules=schedules)

        return population

    def reset(self):
        for s in self.schedules:
            s.trans_m.pop_trans_task()

    def saveschedule(self, index, schdule):
        self.schedules[index] = schdule

    def getschedule(self, index):
        return self.schedules[index]

    def getuse_trans(self, i):
        return self.schedules[i].trans_m

    def getfittest(self, distance_flag=False):
        fittest = self.schedules[0]
        for i in range(1, self.populationsize()):
            if fittest.getfitness(self.work_time, self.empty_time, distance_flag=distance_flag) \
                    <= self.getschedule(i).getfitness(self.work_time, self.empty_time, distance_flag=distance_flag):
                fittest = self.getschedule(i)

        return fittest

    def compatition(self, distance_flag=False):
        parent_dict = {}

        for i in range(0, self.populationsize()):
            score = self.getschedule(i).getfitness(self.work_time, self.empty_time, distance_flag=distance_flag)
            parent_dict[self.getschedule(i)] = score

        p = sorted(parent_dict.items(), key=lambda x: x[1], reverse=True)
        pop = []
        for s in p:
            pop.append(s[0])
            if len(pop) == cross_num:
                break

        return pop

    def populationsize(self):
        return len(self.schedules)
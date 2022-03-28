from pathlib import Path
import random
import datetime
import pandas as pd
import numpy as np

from global_var import init_position, load_time, unload_time
from Object.block import Block


class Task:
    def __init__(self, block, s_node, e_node):
        self.b = block      # 블록 클래스
        self.s_n = s_node   # 출발 적치 장 클래스
        self.e_n = e_node   # 도착 적치 장 클래스

    def return_inform(self):
        return self.b, self.s_n, self.e_n

    # 블록 번호(작업 번호)
    def task_no(self):
        return self.b.no

    # 작업 무게
    def weight(self):
        return self.b.w

    # 작업 사이즈
    def size(self):
        return self.b.b_s()

    # 시작 지점
    def start_p(self):
        return self.s_n.no

    # 도착 지점
    def dest_p(self):
        return self.e_n.no


class Task_schedule:
    def __init__(self, graph, task_num):
        self.task_list = []
        self.task_num = task_num
        self.fileName = r".\Data\task_data_{}.xlsx".format(task_num)
        self.graph = graph
        fileObj = Path(self.fileName)

        if not fileObj.is_file():
            self.generate_task()

        else:
            trans_df = pd.read_excel(self.fileName)
            for i in range(len(trans_df)):
                no, x, y, w, s_no, e_no = trans_df[['block_no', 'b_x', 'b_y', 'b_w', 's_n', 'e_n']].loc[i]
                b = Block(no, x, y, w)
                # d = graph.distance_node(0, s_no, e_no)
                # print(s_no, e_no, d)
                s_no = self.graph.node(s_no)    # 시작 노드 클래스
                e_no = self.graph.node(e_no)    # 끝 노드 클래스
                self.task_list.append(Task(b, s_no, e_no))
        self.task_list.sort(key=lambda x: x.weight(), reverse=True)


    def __repr__(self):
        genestr = "-------------------------- Task --------------------------\n"
        genestr += "Block_no\t\tDeparture\tArrive\n"
        for t in self.task_list:
            b, s_n, e_n = t.return_inform()
            # s = datetime.timedelta(seconds=s)
            # e = datetime.timedelta(seconds=e)

            genestr += "\t{}({})\t\t\t{}\t\t  {}\n".format(b.no, t.weight(), s_n.no, e_n.no)

        genestr += "-------------------------- END --------------------------\n"

        return genestr

    # 작업 생성기
    def generate_task(self):
        for no in range(self.task_num):
            # 블록 생성
            b = self.generate_block(no)
            # 출발지 도착지 설정
            s_no, e_no = self.generate_node()
            self.task_list.append(Task(b, s_no, e_no))
        # self.task_list.sort(key=lambda x: x.s_t)
        self.generate_file()

    # 블록 생성기
    def generate_block(self, no):
        x = random.randint(1,50)
        y = random.randint(1,50)
        w = random.random()
        if w <= 0.07:
            w = random.randint(1, 50)
        elif w <= 0.18:
            w = random.randint(50, 150)
        elif w <= 0.29:
            w = random.randint(150, 250)
        elif w <= 0.47:
            w = random.randint(250, 350)
        elif w <= 0.66:
            w = random.randint(350, 450)
        elif w <= 0.81:
            w = random.randint(450, 500)
        elif w <= 0.92:
            w = random.randint(500, 600)
        elif w <= 1:
            w = random.randint(600, 700)
        b = Block(no, x, y, w)
        return b

    # 작업 출발지와 도착지 생성기
    def generate_node(self):
        # start = random.randint(0, len(self.graph.stock_list)-1)
        # end = random.randint(0, len(self.graph.stock_list)-1)
        start = random.randint(0, 20)
        end = random.randint(0, 20)
        # 출발지와 도착지가 같으면 안됨
        while True:
            if end == start:
                end = random.randint(0, len(self.graph.stock_list)-1)
            else:
                break

        start = self.graph.node(start)
        end = self.graph.node(end)

        return start, end

    def generate_file(self):
        task_execl = pd.DataFrame(
            {
                'block_no': [],
                'b_x': [],
                'b_y': [],
                'b_w': [],
                's_n': [],
                'e_n': []
            })

        block_no_list = []
        b_x_list = []
        b_y_list = []
        b_w_list = []
        s_n_list = []
        e_n_list = []

        for task in self.task_list:
            block_no_list.append(task.b.no)
            b_x_list.append(task.b.s_x)
            b_y_list.append(task.b.s_y)
            b_w_list.append(task.b.w)
            s_n_list.append(task.s_n.no)
            e_n_list.append(task.e_n.no)

        task_execl['block_no'] = block_no_list
        task_execl['b_x'] = b_x_list
        task_execl['b_y'] = b_y_list
        task_execl['b_w'] = b_w_list
        task_execl['s_n'] = s_n_list
        task_execl['e_n'] = e_n_list

        task_execl.to_excel(self.fileName)


def task_classification(task_list, trans_list, graph):

    # task_list -> task -> b(block), s_n, e_n
    # 출발, 도착 적치 장 클래스??

    empty_speed = []
    work_speed = []

    # 트랜스포터들의 공차 속도에 따른 작업 관계를 위해 추출
    # 적재 o, 적재 x 속도 리스트

    for i in trans_list:
        if i.e_s not in empty_speed:
            empty_speed.append(i.e_s)
        if i.w_s not in work_speed:
            work_speed.append(i.w_s)
    empty_speed = sorted(empty_speed)
    work_speed = sorted(work_speed)

    task_work_time, task_empty_time = task_matrix(task_list, empty_speed, work_speed, graph)

    return task_work_time, task_empty_time, empty_speed


def task_matrix(t_list, empty_speed, work_speed, graph):
    # 작업 속도별 전체 작업 시간 메트릭스
    task_work = {}

    # 작업 시간
    for speed in work_speed:
        work_time = np.zeros(len(t_list))
        for i, node in enumerate(t_list):
            s_node = node.start_p()
            d_node = node.dest_p()
            d = graph.distance_node(0, s_node, d_node)  # 거리
            # print(s_node, d_node, d)
            # speed로 i번째 작업 걸리는 시간
            work_time[i] = d//speed + load_time + unload_time
        # 속도들에 대해 (0~i번째) 작업 시간
        task_work[speed] = work_time


    # 작업간 이동 시간
    task_distance_matrix = {}
    for speed in empty_speed:
        empty_time = np.zeros((len(t_list)+1, len(t_list)))
        for i, s_node in enumerate(t_list):
            s_node = s_node.dest_p()  # 작업 도착지점
            for j, d_node in enumerate(t_list[:i]):
                d_node = d_node.start_p()  # 작업 시작지점
                d = graph.distance_node(0, s_node, d_node)  # 거리
                empty_time[i, j] = d // speed

            for j, d_node in enumerate(t_list[i + 1:]):
                d_node = d_node.start_p()  # 작업 시작지점
                d = graph.distance_node(0, s_node, d_node)  # 거리
                empty_time[i, j+i+1] = d // speed
            
            d = graph.distance_node(0, s_node, init_position)   # 거리
            empty_time[i, i] = d // speed
            d = graph.distance_node(0, init_position, s_node)  # 거리
            empty_time[-1, i] = d // speed

        task_distance_matrix[speed] = empty_time

    return task_work, task_distance_matrix

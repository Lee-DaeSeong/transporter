from pathlib import Path
import pandas as pd
import random
from Object.Transporter import Transporter

def transporter_data(trans_num, trans_manager, graph):
    fileName = r".\Data\transporter_data.xlsx"
    fileObj = Path(fileName)

    if not fileObj.is_file():

        trans_df = pd.read_excel(r"transporter_list.xlsx")

        trans_data = zip(trans_df['size'], trans_df['availble_weight'], trans_df['work_speed'], trans_df['empty_speed'], trans_df['turn_speed'])
        trans_execl = pd.DataFrame(
            {
                'no': [],
                'size': [],
                'available_weight': [],
                'work_speed': [],
                'empty_speed': [],
                'turn_speed': []
            })
        no_list = []
        size_list = []
        available_weight_list = []
        work_speed_list = []
        empty_speed_list = []
        turn_speed_list = []
        for no in range(trans_num):
            p = random.random()

            if p < 0.07:
                p = 0
            elif p < 0.18:
                p = 1
            elif p < 0.29:
                p = 2
            elif p < 0.47:
                p = 3
            elif p < 0.66:
                p = 4
            elif p < 0.81:
                p = 5
            elif p < 0.92:
                p = 6
            else:
                p = 7
            model, size, a_w, w_s, e_s, t_s = trans_df.loc[p]
            # print(size, type(size), type(w_s), type(a_w))
            w_s, e_s, t_s = trans_mpers(w_s, e_s, t_s)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            trans_manager.add_trans(Transporter(no, size, a_w, w_s, e_s, t_s, graph, color))

            no_list.append(no)
            size_list.append(size)
            available_weight_list.append(a_w)
            work_speed_list.append(w_s)
            empty_speed_list.append(e_s)
            turn_speed_list.append(t_s)

        trans_execl['no'] = no_list
        trans_execl['size'] = size_list
        trans_execl['available_weight'] = available_weight_list
        trans_execl['work_speed'] = work_speed_list
        trans_execl['empty_speed'] = empty_speed_list
        trans_execl['turn_speed'] = turn_speed_list

        trans_execl.to_excel(fileName)

    else:
        # else문 실행

        trans_df = pd.read_excel(r".\Data\transporter_data.xlsx")

        for i in range(len(trans_df)):

            # print("trans_df.loc[]: ", trans_df.loc[i])
            no, size, a_w, w_s, e_s, t_s = trans_df[['no', 'size', 'available_weight', 'work_speed', 'empty_speed', 'turn_speed']].loc[i]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            trans_manager.add_trans(Transporter(no, size, a_w, w_s, e_s, t_s, graph, color))

# 미터 퍼 세크으로 변환
def trans_mpers(w_s, e_s, t_s):
    w = w_s * 1000 // 3600
    e = e_s * 1000 // 3600
    t = t_s * 1000 // 3600
    return w, e, t

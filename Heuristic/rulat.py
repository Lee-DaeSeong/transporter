import numpy as np

# 트랜스포터 룰렛판 생성
def trans_rulat(transporter_list, weight):
    trans_con = [i for i in transporter_list if i.a_w >= weight]
    rulat = np.array([len(i.task) for i in trans_con])
    rulat = rulat + 1  # 할당 안 된 트랜스포터도 기회 주려고..
    rulat = rulat**2
    rulat = rulat / np.sum(rulat)
    rulat = list(rulat)

    return trans_con, rulat

# 작업 룰렛판 생성
def task_rulat(rulat_list):
    rulat_list = np.array(rulat_list)
    rulat_list = np.insert(rulat_list, 0, np.average(rulat_list))   # 맨 처음에 들어가는 것도 기회 줄려고..
    rulat = np.sum(rulat_list) / rulat_list
    rulat = rulat / np.sum(rulat)

    return rulat
from global_var import work_stime, work_etime, load_time, unload_time

# 무게 제약 사항
def weight_condition(task, trans_list):
    w = task.weight()
    pass_trans = []

    for trans in trans_list:
        if trans.a_w >= w:
            pass_trans.append(trans)

    return pass_trans

# 시간 제약사항       최소 거리 행렬, 현재 작업, 트랜스포터 리스트
def time_condition(task, trans_list, work_time, empty_time):
    pass_trans = []
    d = task.start_p()
    a = task.dest_p()

    for t in trans_list:
        p = t.cur_p()       # 트랜스포터 현재 위치
        a_t = t.a_time()    # 트랜스포터 작업 가능 시각

        # size = real_size(task, t)           # 작업 폭
        #
        # w_i = graph.width_index(size)       # 작업 폭 인덱스
        # e_i = graph.width_index(t.size)     # 공차 폭 인덱스

        # 총 시각
        # e_d = graph.distance_node(e_i, p, d) // t.e_s
        # w_d = graph.distance_node(w_i, d, a) // t.w_s
        no = task.task_no()
        e_t = empty_time[t.e_s][p][no]
        w_t = work_time[t.w_s][no]
        total_t = e_t + load_time + w_t + unload_time

        # 최대 마무리 시각보다 일찍 끝난다면..
        if a_t + total_t <= work_etime:
            pass_trans.append(t)

    return pass_trans

# 실제 넓이
def real_size(task, trans):
    b_s = task.size()
    if b_s < trans.size:
        return trans.size
    else:
        return b_s

# def modified(schedule):
#     t_ = {}
#     r_ = schedule.graph.road_list
#
#     for trans in schedule.trans_m:
#         for data in trans.total_data:
#             if data[4] is not None:
#                 if data[0] not in t_:
#                     # 시작 시각, 끝 시각, 사용 도로
#                     t_[data[0]] = [(data[1], data[4])]
#                 else:
#                     t_[data[0]] = t_[data[0]] + [(data[1], data[4])]

def modified(schedule):
    r_ = schedule.graph.road_list
    t_ = {}
    # 트랜스포터가 도로에 들리는거 추가하는 부분
    for trans in schedule.trans_m:
        for data in trans.total_data:
            data[4].add_(trans, data)

    # 각 도로별 시간 순대로 정렬
    for road in r_:
        road.sort_dict()
        
    




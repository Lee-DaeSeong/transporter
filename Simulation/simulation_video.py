from global_var import init_position, work_stime, work_etime
import cv2
import numpy as np
import random

pathOut = './UI/simulation.avi'
fps = 5
frame_array = []

font = cv2.FONT_HERSHEY_SIMPLEX

def map_view(graph, trans_manager):
    img = cv2.imread('.\\UI\\MAP.png')
    # cv2.
    height, width, layers = img.shape
    size = (width, height)

    pop_trans = []
    for trans in trans_manager.transporter_list:
        if trans.total_distance == 0:
            pop_trans.append(trans)

    sequence = 0
    for index, t in enumerate(range(work_stime, work_etime, 5)):
        i = img.copy()
        i, pop_trans = trans_view(t, graph, trans_manager, pop_trans, i)
        frame_array.append(i)

        if index % 200 == 0:
            out = cv2.VideoWriter('./UI/simulation{}.avi'.format(sequence), cv2.VideoWriter_fourcc(*'XVID'), fps, size)
            sequence = sequence + 1
            for index, i in enumerate(frame_array):
                # writing to a image array
                out.write(i)
                print("진행률: {}%".format(index / len(frame_array) * 100))
            frame_array.clear()
            out.release()


    print(len(frame_array))

def trans_view(time, graph, trans_manager, pop_trans, img):
    for trans in trans_manager.transporter_list:
        gene_string = '{}: wait'.format(trans.no)
        if trans in pop_trans:
            node = None
            if trans.total_distance == 0:
                node = graph.node(init_position)
            else:
                node = graph.node(trans.total_data[-1][-1][2])
            x = node.x
            y = node.y
            cv2.circle(img, (x, y), 20, trans.color, -1)
            cv2.putText(img, gene_string, (x, y+40), font, 1, trans.color, 2, cv2.LINE_AA)
            continue

        i, i2 = trans.simul_task, trans.simul_i
        # print(trans.total_data, trans.total_distance)
        total_i, total_i2 = len(trans.total_data)-1, len(trans.total_data[i])-1

        x, y = None, None
        # 시작 시각 이전이면 --> start_node에 위치 / 무조건 대기상태
        if time < trans.total_data[i][i2][0]:
            node = graph.node(trans.total_data[i][i2][2])
            x, y = node.x, node.y
            
        # 이동중이면
        elif trans.total_data[i][i2][0] <= time and time < trans.total_data[i][i2][1]:
            # 상차 중
            if trans.total_data[i][i2][-1] == 2:
                gene_string = '{}: Loading'.format(trans.no)
                node = graph.node(trans.total_data[i][i2][2])
                x, y = node.x, node.y

            # 하차 중
            elif trans.total_data[i][i2][-1] == 4:
                gene_string = '{}: unLoading'.format(trans.no)
                node = graph.node(trans.total_data[i][i2][2])
                x, y = node.x, node.y

            else:
                if trans.total_data[i][i2][-1] == 1:
                    gene_string = '{}: Empty_move'.format(trans.no)
                elif trans.total_data[i][i2][-1] == 3:
                    gene_string = '{}: Work_move'.format(trans.no)
                # 해당 도로 이동시간
                diff = time - trans.total_data[i][i2][0]
                speed = trans.total_data[i][i2][5]
                # 이동거리
                d = diff * speed
                # 이동거리 비율
                fi = d / trans.total_data[i][i2][4].d

                # 도착 노드
                n1 = graph.node(trans.total_data[i][i2][3])
                # 시작 노드
                n2 = graph.node(trans.total_data[i][i2][2])

                x = n2.x + int((n1.x - n2.x) * fi)
                y = n2.y + int((n1.y - n2.y) * fi)
            
        else:
            if trans.total_data[i][i2][-1] == 1:
                gene_string = '{}: Empty_move'.format(trans.no)
            elif trans.total_data[i][i2][-1] == 3:
                gene_string = '{}: Work_move'.format(trans.no)
                
            # 현재 작업의 길이가 마지막이면 다음 작업으로 
            if total_i2 == i2:
                # 현재 작업이 마지막이면 pop_trans에 추가
                if total_i == i:
                    pop_trans.append(trans)
                else:
                    trans.simul_task = trans.simul_task + 1
                    trans.simul_i = 0
            else:
                trans.simul_i = trans.simul_i + 1
            node = graph.node(trans.total_data[i][i2][3])
            x, y = node.x, node.y

        cv2.circle(img, (x, y), 20, trans.color, -1)
        cv2.putText(img, gene_string, (x, y + 40), font, 1, trans.color, 2, cv2.LINE_AA)

    return img, pop_trans

            

            

if __name__ == '__main__':
    map_view()

import cv2
import numpy as np
from Object.graph import Graph
from Data_create.call_data import object_data
import random

def map_view(graph):
    img = np.zeros((1400, 3100, 3), np.uint8)

    for road in graph.road_list:
        s_n = road.s
        e_n = road.e
        s_x, s_y = s_n.x, s_n.y
        e_x, e_y = e_n.x, e_n.y
        B = 255 # random.randint(0,255)
        G = 255 # random.randint(0,255)
        R = 255 # random.randint(0,255)
        img = cv2.line(img, (s_x, s_y), (e_x, e_y), (B, G, R), 2)

    for stock in graph.stock_list:
        x = stock.x
        y = stock.y
        img = cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '{}'.format(stock.no), (x, y+30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    for inter in graph.inter_list:
        x = inter.x
        y = inter.y
        img = cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '{}'.format(inter.no), (x, y + 40), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('..\\UI\\MAP_1.png', img)

if __name__=='__main__':
    # 데이터 불러오기
    stock_data, inter_data, road_data = object_data()
    # 그래프 생성
    graph = Graph(stock_data, inter_data, road_data)
    map_view(graph)
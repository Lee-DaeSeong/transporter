import numpy as np
import pandas as pd
from openpyxl.workbook import Workbook

df = pd.DataFrame(
                {
                    'stockyard_no': [],
                    'stockyard_x': [],
                    'stockyard_y': [],
                    'intersection_no': [],
                    'intersection_x': [],
                    'intersection_y': []
                })
df_road = pd.DataFrame(
                {
                    'road_start': [],
                    'road_end': [],
                    'road_width': [],
                })
stockyard_no = []
stockyard_x = []
stockyard_y = []
stockyard_num = 37
intersection_no = []
intersection_x = []
intersection_y = []

for i in range(stockyard_num):
    stockyard_no.append(i)

stockyard_position = [(0, 0), (200, 0), (1000, 0), (900, 200), (1100, 200), (900, 300), (1100, 300), (900, 500), (1100, 500), (1300,600), (1500,700), (600, 600), (700, 900), (900, 1000), (1300, 800), (1500, 800), (1200, 1100), (1300, 1100), (1200, 1200), (1300, 1200), (2500, 0), (1900, 300), (2200, 300), (2400, 500), (1900, 700), (2200, 700), (2600, 600), (2700, 700), (2300, 1000), (2500, 1000), (2300, 1200), (2500, 1200), (2900, 700), (3000, 500), (2800, 1000), (2900, 1000), (2800, 1300)]

for i in stockyard_position:
    stockyard_x.append(i[0])
    stockyard_y.append(i[1])
intersection_num = 74-37
for i in range(intersection_num):
    intersection_no.append(i+stockyard_num)

intersection_position = [(1000,	200), (1000,	300), (600,	300), (600,	500), (1000,	500), (1200,	500), (1200,	600), (1200,	700), (600,	800), (1200,	800), (600,	900), (800,	900), (800,	1000), (1000,	1000), (1200,	1000), (1200,	900), (1400,	900), (1400,	800), (1000,	1300), (1300,	1300), (2200,	1300), (2200,	1000), (2200,	800), (1900,	800), (1900,	500), (1900,	100), (2200,	100), (2500,	100), (2500,	500), (2200,	500), (2700,	800), (2900,	800), (3000,	800), (2700,	1000), (3000,	1000), (2700,	1300), (3000,	1300)]
for i in intersection_position:
    intersection_x.append(i[0])
    intersection_y.append(i[1])

road_start_end = [(0,1), (1,2), (2,38), (3,38), (4,38), (38,39), (39,6), (39,5), (5,40), (39,42), (40,41), (41,7), (7,42), (42,8), (8,43), (41,11), (11,46), (46,48), (43,44), (44,9), (44,45), (45,10), (45,47), (46,47), (48,12), (12,49), (49,50), (50,13), (13,51), (51,52), (47,53), (49,53), (53,52), (53,54), (54,55), (55,14), (55,15), (54,15), (51,56), (52,16), (16,18), (18,56), (54,17), (17,19), (16,17), (18,19), (19,57), (15,57), (57,58), (58,59), (58,30), (30,31), (31,73), (59,28), (28,29), (29,71), (59,60), (60,68), (60,61), (61,24), (24,62), (62,21), (21,63), (63,64), (64,22), (60,25), (25,67), (67,22), (67,23), (23,66), (65,66), (64,65), (65,20), (66,26), (24,25), (62,67), (21,22), (29,68), (27,68), (68,71), (71,73), (68,69), (69,32), (69,70), (33,70), (71,34), (34,35), (69,35), (35,72), (72,36), (70,72), (73,37), (34,37), (37,74), (72,74)]
road_start = []
road_end = []
road_width = []
for i in road_start_end:
    s = i[0]
    e = i[1]
    if s >= 37:
        s -= 1
    if e >= 37:
        e -= 1

    road_start.append(s)
    road_end.append(e)
    road_width.append(100)

print(len(stockyard_x), len(intersection_x))
df['stockyard_no'] = stockyard_no
df['stockyard_x'] = stockyard_x
df['stockyard_y'] = stockyard_y
df['intersection_no'] = intersection_no
df['intersection_x'] = intersection_x
df['intersection_y'] = intersection_y

df_road['road_start'] = road_start
df_road['road_end'] = road_end
df_road['road_width'] = road_width

df = df[['stockyard_no', 'stockyard_x', 'stockyard_y', 'intersection_no', 'intersection_x', 'intersection_y']]
df_raod = df_road[['road_start', 'road_end', 'road_width']]

df.to_excel('map_object_position.xlsx', index=False)
df_road.to_excel('road_object.xlsx', index=False)

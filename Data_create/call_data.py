import pandas as pd

def object_data():
    object_df = pd.read_excel("map_object_position.xlsx")
    road_df = pd.read_excel("road_object.xlsx")


    stock_data = zip(object_df['stockyard_no'], object_df['stockyard_x'],object_df['stockyard_y'])
    inter_data = zip(object_df['intersection_no'], object_df['intersection_x'],object_df['intersection_y'])
    road_data = zip(road_df['road_start'], road_df['road_end'], road_df['road_width'])

    return stock_data, inter_data, road_data
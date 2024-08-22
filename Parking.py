from Map import Map
import os
import cv2
import numpy as np
from utils import plot_one_box, cal_iou, xyxy_to_xywh, xywh_to_xyxy, updata_trace_list, draw_trace, intersect, \
    is_point_inside_rectangle, subtract_tuples, coord_to_pixel
import datetime
from tracks import Tracks
from tracker import Tracker
from kalmanfilter import KalmanFilter
from scipy.optimize import linear_sum_assignment
import json
from globle_variable import ax
class Parking():
    '''

    '''

    def __int__(self, content, area_inf_list=None, lane_inf=None, frame_rate=6):
        self.map_list = [Map(content, area_inf_list, lane_inf, frame_rate, 'scene_1'),
                         Map(content, area_inf_list, lane_inf, frame_rate, 'scene_2')]



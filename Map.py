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
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean
import copy


class Map:
    def __init__(self, content, area_inf_list=None, lane_inf=None, frame_rate=6):
        self.scene_list = self.init_scene(content, frame_rate)
        self.tracks_set = []
        if area_inf_list is None:
            area_inf_list = []
        self.areas = []
        # self.tracker_list = [Tracker(content, frame_rate)]
        self.threshold_in = 1  # 距离阈值，此处要改
        self.threshold_out = 1
        for i, area_inf in enumerate(area_inf_list):
            self.areas.append(Area(i, area_inf[0], area_inf[1]))
        self.lane = Lane(0, lane_inf[0], lane_inf[1]) if lane_inf else None
        self.update_info = None  # 此参数用于记录每次做更新操作时新创建的轨迹以及被移除的轨迹

    def init_scene(self, content, frame_rate):
        """
        初始化跟踪器列表
        """
        content1 = [item for item in content if item[-8:] == 'scene_1\n']
        content2 = [item for item in content if item[-8:] == 'scene_2\n']
        Scene1 = Scene(content1, frame_rate=frame_rate, camera_id='scene_1')
        Scene2 = Scene(content2, frame_rate=frame_rate, camera_id='scene_2')
        return [Scene1, Scene2]

    def update(self, content):
        content1 = [item for item in content if item[-8:] == 'scene_1\n']
        content2 = [item for item in content if item[-8:] == 'scene_2\n']
        self.scene_list[0].update(content1)
        self.scene_list[1].update(content2)
        self.match_tracks()

    def match_tracks(self):
        """
        用于不同摄像头下的跟踪器之间的轨迹匹配
        """
        self.preprocess_matched_tracks()
        self.scene_list[0].update_tracks()
        self.scene_list[1].update_tracks()

        n1 = len(self.scene_list[0].not_matched_tracks)
        n2 = len(self.scene_list[1].not_matched_tracks)
        if n1 == 0 and n2 == 0:
            return

        if n1 > 0 and n2 > 0:
            point_n1_list = np.array([track.box_center for track in self.scene_list[0].not_matched_tracks])
            point_n2_list = np.array([track.box_center for track in self.scene_list[1].not_matched_tracks])
            distance_matrix = cdist(point_n1_list, point_n2_list)

            if not np.all(distance_matrix == 0):
                track1_indices, track2_indices = linear_sum_assignment(distance_matrix)
            else:
                track1_indices, track2_indices = ([], [])

            matched_pairs_scene1 = set()
            matched_pairs_scene2 = set()
            for j in range(len(track1_indices)):
                index1 = track1_indices[j]
                index2 = track2_indices[j]
                track1 = self.scene_list[0].not_matched_tracks[index1]
                track2 = self.scene_list[1].not_matched_tracks[index2]
                dis = euclidean(point_n1_list[index1], point_n2_list[index2])
                if dis <= 2:
                    # Update matching properties
                    track1.match_between_scene = True
                    track1.matched_id = track2.track_id
                    track1.matched_scene = self.scene_list[1].camera_id

                    track2.match_between_scene = True
                    track2.matched_id = track1.track_id
                    track2.matched_scene = self.scene_list[0].camera_id

                    # Store matched tracks as a tuple
                    self.tracks_set.append((track1, track2))
                    matched_pairs_scene1.add(index1)
                    matched_pairs_scene2.add(index2)

            # Add unmatched tracks from both scenes
            for i, track in enumerate(self.scene_list[0].not_matched_tracks):
                if i not in matched_pairs_scene1:
                    self.tracks_set.append(track)
            for i, track in enumerate(self.scene_list[1].not_matched_tracks):
                if i not in matched_pairs_scene2:
                    self.tracks_set.append(track)
        else:
            # Handle cases where one of the scene lists is empty
            if n1 == 0:
                self.tracks_set.extend(self.scene_list[1].not_matched_tracks)
            if n2 == 0:
                self.tracks_set.extend(self.scene_list[0].not_matched_tracks)

    def preprocess_matched_tracks(self):
        """
        处理已匹配轨迹对的距离，如果距离大于8，则解除匹配关联。
        只保留距离小于8的匹配对。
        """
        new_tracks = []
        for pair in self.tracks_set:
            if isinstance(pair, tuple):  # 确认是否为匹配对
                track1, track2 = pair
                distance = euclidean(track1.box_center, track2.box_center)
                if distance <= 2:
                    # 保持匹配状态
                    new_tracks.append(pair)
                else:
                    # 解除匹配关联
                    track1.match_between_scene = False
                    track1.matched_id = None
                    track1.matched_scene = None
                    track2.match_between_scene = False
                    track2.matched_id = None
                    track2.matched_scene = None
        self.tracks_set = new_tracks

    def match_tracks__(self):
        """
        用于不同摄像头下的跟踪器之间的轨迹匹配
        """

        n1 = len(self.scene_list[0].not_matched_tracks)
        n2 = len(self.scene_list[1].not_matched_tracks)
        dis_mat = np.zeros((n1, n2))
        point_n1_list = np.array([track.box_center for track in self.scene_list[0].not_matched_tracks])
        point_n2_list = np.array([track.box_center for track in self.scene_list[1].not_matched_tracks])
        # 使用 cdist 函数计算距离矩阵
        if n1 == 0:
            if n2 == 0:
                self.tracks = []
                return
            else:
                self.tracks = self.scene_list[1].tracks
                return

        if n2 == 0:
            if n1 == 0:
                self.tracks = []
                return
            else:
                self.tracks = self.scene_list[0].tracks
                return
        distance_matrix = cdist(point_n1_list, point_n2_list)  # 矩阵中的每个元素 (i, j) 代表 array1[i] 到 array2[j] 的欧几里得距离。
        if np.all(distance_matrix == 0):
            track1_indices, track2_indices = ([], [])
        else:
            track1_indices, track2_indices = linear_sum_assignment(distance_matrix)
        # self.tracks = self.tracker_list[0].tracks
        self.tracks = copy.deepcopy(self.scene_list[0].tracks)
        # for i, track2 in enumerate(self.scene_list[1].tracks):
        #     if i not in track2_indices:
        #         self.tracks.append(self.scene_list[1].tracks[i])
        #     else:
        #         # try:
        #         # 获取元素在列表中的索引
        #         # j = track2_indices.index(i)  # 场景2中i匹配到场景1中的j
        #         # numpy.where 函数返回一个元组，其中元素是包含符合条件的索引的数组。在这种情况下，我们只关心第一个维度上的索引，因此使用 indices[0]。
        #         j = np.where(track2_indices == i)[0][0]
        #         dis = euclidean(point_n1_list[j], point_n2_list[i])
        #         if dis > 0.1:
        #             self.tracks.append(copy.deepcopy(self.scene_list[1].tracks[i]))
        #             # import sys
        #             # sys.exit()
        #         # except ValueError:
        #         #     print(f"Error in linear_sum_assignment")

        for i, track2 in enumerate(self.scene_list[1].tracks):
            if i in track2_indices:
                j = np.where(track2_indices == i)[0][0]
                track1 = self.scene_list[0].tracks[j]
                dis = euclidean(point_n1_list[j], point_n2_list[i])
                if dis <= 0.1:
                    # Update matched tracks properties
                    track1.match_between_scene = True
                    track1.matched_id = track2.track_id
                    track1.matched_scene = self.scene_list[1].camera_id

                    track2.match_between_scene = True
                    track2.matched_id = track1.track_id
                    track2.matched_scene = self.scene_list[0].camera_id
            else:
                self.tracks.append(copy.deepcopy(track2))

    def get_area(self, id):
        for area in self.areas:
            if area.id == id:
                return area
        return None

    def get_track(self, id):
        for track in self.tracks_set:
            if track.track_id == id:
                return track
        return None

    def draw_tracks(self, img):
        for track in self.tracks_set:
            if isinstance(track, tuple):
                track = track[1]
            if track.confirmflag:
                track.draw(img)

    def draw_area(self, frame):
        for area in self.areas:
            area.draw(frame)
        self.lane.draw(frame)


class Scene(Tracker):
    '''
    class Map inherit from class Tracker
    '''

    def __init__(self, content, area_inf_list=None, lane_inf=None, frame_rate=6, camera_id=''):
        super(Scene, self).__init__(content, frame_rate, camera_id)
        '''
        parking_id由Map分配
        :param content: 第一帧检测到的信息，用于初始化跟踪器
        :param parking_file: 拿到的地图的位置信息的文件
        '''
        self.range = [(0, 0), (1, 1)]  # 该摄像头所管理的场景的范围
        self.camera_location = (0, 0)
        self.tracks = []  # 检测到的所有的轨迹
        self.matched_tracks = []  # 与其他scene中的轨迹匹配上了的轨迹的列表
        self.not_matched_tracks = []  # 没有与其他scene中的轨迹匹配上的轨迹的列表
        if area_inf_list is None:
            area_inf_list = []
        self.areas = []
        # self.tracker_list = [Tracker(content, frame_rate)]
        self.threshold_in = 1  # 距离阈值，此处要改
        self.threshold_out = 1
        # self.camera_id = camera_id
        for i, area_inf in enumerate(area_inf_list):
            self.areas.append(Area(i, area_inf[0], area_inf[1]))
        self.lane = Lane(0, lane_inf[0], lane_inf[1]) if lane_inf else None
        self.update_info = None  # 此参数用于记录每次做更新操作时新创建的轨迹以及被移除的轨迹

    def init_tracker(self, content, frame_rate):
        """
        初始化跟踪器列表
        """
        content1 = [item for item in content if item[-8:] == 'scene_1\n']
        content2 = [item for item in content if item[-8:] == 'scene_2\n']
        tracker1 = Tracker(content1, frame_rate, 'scene_1')
        tracker2 = Tracker(content2, frame_rate, 'scene_2')
        return [tracker1, tracker2]

    def update(self, content):
        # content1 = [item for item in content if item[-8:] == 'scene_1\n']
        # content2 = [item for item in content if item[-8:] == 'scene_2\n']
        # self.tracker_list[0].update(content1)
        # self.tracker_list[1].update(content2)
        # self.match_tracks()

        removed_track, added_track = super().update(content)
        self.update_tracks()

        # self.update_count((removed_track, added_track))
        # self.update_events()  # 事件的更新

    def update_tracks(self):
        """
        更新matched_tracks， not_matched_tracks这些
        """
        self.matched_tracks = [track for track in self.tracks if track.match_between_scene]  # 浅拷贝
        self.not_matched_tracks = [track for track in self.tracks if not track.match_between_scene]

    def match_tracks(self):
        '''
        用于不同摄像头下的跟踪器之间的轨迹匹配
        '''

        n1 = len(self.tracker_list[0].tracks)
        n2 = len(self.tracker_list[1].tracks)
        dis_mat = np.zeros((n1, n2))
        point_n1_list = np.array([track.box_center for track in self.tracker_list[0].tracks])
        point_n2_list = np.array([track.box_center for track in self.tracker_list[1].tracks])
        # 使用 cdist 函数计算距离矩阵
        if n1 == 0:
            if n2 == 0:
                self.tracks = []
                return
            else:
                self.tracks = self.tracker_list[1].tracks
                return

        if n2 == 0:
            if n1 == 0:
                self.tracks = []
                return
            else:
                self.tracks = self.tracker_list[0].tracks
                return
        distance_matrix = cdist(point_n1_list, point_n2_list)  # 矩阵中的每个元素 (i, j) 代表 array1[i] 到 array2[j] 的欧几里得距离。
        if np.all(distance_matrix == 0):
            track1_indices, track2_indices = ([], [])
        else:
            track1_indices, track2_indices = linear_sum_assignment(distance_matrix)
        # self.tracks = self.tracker_list[0].tracks
        self.tracks = copy.deepcopy(self.tracker_list[0].tracks)
        for i, track2 in enumerate(self.tracker_list[1].tracks):
            if i not in track2_indices:
                self.tracks.append(self.tracker_list[1].tracks[i])
            else:
                # try:
                # 获取元素在列表中的索引
                # j = track2_indices.index(i)  # 场景2中i匹配到场景1中的j
                # numpy.where 函数返回一个元组，其中元素是包含符合条件的索引的数组。在这种情况下，我们只关心第一个维度上的索引，因此使用 indices[0]。
                j = np.where(track2_indices == i)[0][0]
                dis = euclidean(point_n1_list[j], point_n2_list[i])
                if dis > 8:
                    self.tracks.append(copy.deepcopy(self.tracker_list[1].tracks[i]))
                    # import sys
                    # sys.exit()
                # except ValueError:
                #     print(f"Error in linear_sum_assignment")

    def get_area(self, id):
        for area in self.areas:
            if area.id == id:
                return area
        return None

    def get_track(self, id):
        for track in self.tracks:
            if track.track_id == id:
                return track
        return None

    def intersect(self, track, point, previous_point):
        for area in self.areas:
            area.intersect(track, point, previous_point)

    def draw_area(self, frame):
        for area in self.areas:
            area.draw(frame)
        self.lane.draw(frame)

    def draw_tracks(self, img):
        for track in self.tracks:
            if track.confirmflag:
                track.draw(img)

    def nearest_area_distance(self, track, flag):
        if flag == 'remove':
            point = track.trace_point_list[-1]
        elif flag == 'added':  # 对于新增加的轨迹应该计算第一个点离区域的距离而不是最新的
            point = track.trace_point_list[0]
        min_distance = float('inf')
        min_area_id = -1

        for i, area in enumerate(self.areas):
            distance = area.point_to_border_distance(point)
            if distance < min_distance:
                min_distance = distance
                min_area_id = area.id

        return min_area_id, min_distance

    def update_count(self, update_info):
        """
        更新Map图中的区域计数信息等
        :return: None
        """
        if not self.lane:
            return
        self.lane.entry_event, self.lane.exit_event = False, False
        self.lane.parking_violation_event, self.lane.traffic_congestion_event = False, False
        removed_tracks, added_tracks = update_info
        for track in self.tracks:
            if len(track.trace_point_list) < 2:
                break
            point = track.trace_point_list[-1]
            previous_point = track.trace_point_list[-2]
            if track.confirmflag and not (track in added_tracks):
                self.intersect(track, point, previous_point)
            if self.lane.isenter(point):
                track.in_lane = True
            else:
                track.in_lane = False

        # 处理新添加的轨迹
        for track in added_tracks:
            nearest_area_index, nearest_distance = self.nearest_area_distance(track, flag='added')
            is_track_inside_area = False

            for area in self.areas:
                if area.isenter(track.trace_point_list[-1]):  # 新添加的轨迹应该计算最后一个点的
                    is_track_inside_area = True
                    break

            if (not is_track_inside_area) and nearest_distance < self.threshold_out:  # 如果距离小于60/80
                # if not self.areas[nearest_area_index].isenter(track.trace_point_list[-1]):  # 新出现的在某个area外面时才会判断是否出去了
                #     self.areas[nearest_area_index].count_car -= 1
                self.areas[nearest_area_index].count_car -= 1
                track.parking_id = None
            elif self.lane.isenter(track.trace_point_list[0]):  # 第一次出现时候在车道里面且判断不是在车位里面出来的则判定为进入车道
                self.lane.entry_event = True

        # 处理消失的轨迹
        for track in removed_tracks:
            point = track.trace_point_list[-1]
            previous_point = track.trace_point_list[-2]
            if track.confirmflag and not (track in added_tracks):
                self.intersect(track, point, previous_point)
            nearest_area_id, nearest_distance = self.nearest_area_distance(track, flag='remove')
            is_track_inside_area = False

            for area in self.areas:
                if area.isenter(track.trace_point_list[-1]):
                    is_track_inside_area = True
                    break

            if not is_track_inside_area and nearest_distance < self.threshold_in:
                if track.confirmflag:
                    self.areas[nearest_area_id].count_car += 1
                    track.parking_id = nearest_area_id
            elif self.lane.isenter(track.trace_point_list[-1]):  # 消失的时候在车道里面且判断不是在车位里面的则判定为驶出了车道
                self.lane.entry_event = True

    def update_direction(self, track, reference_point=(0, 0)):
        '''
        以相机为参照物，给出目标远离和靠近状态
        :param track:需要判断的轨迹
        :param reference_point:相机位置
        :return:
        '''
        track.approaching_event, track.leaving_event = False, False
        if len(track.trace_point_list) < 2:
            return
        current_point = np.array(track.trace_point_list[-1])
        previous_point = np.array(track.trace_point_list[-2])
        current_distance = np.linalg.norm(current_point - reference_point)
        previous_distance = np.linalg.norm(previous_point - reference_point)

        if current_distance < previous_distance:
            track.approaching_event = True
        else:
            track.leaving_event = True

    def update_events(self):
        if not self.lane:
            return
        # self.lane.entry_event, self.lane.exit_event = False, False
        # self.lane.parking_violation_event, self.lane.traffic_congestion_event = False, False

        # reference_point = np.array([width / 2, height / 2])  # 使用视频中心作为参考点
        for track in self.tracks:
            self.update_direction(track)
            if len(track.trace_point_list) < 2:
                continue
            point = track.trace_point_list[-1]
            previous_point = track.trace_point_list[-2]

            # 更新车辆进入和离开区域事件
            if self.lane.intersect(point, previous_point):
                inside = self.lane.isenter(point)
                self.lane.entry_event |= inside
                self.lane.exit_event |= not inside

            # import sys
            # if self.lane.exit_event:
            #     sys.exit()
            # 更新车辆靠近和远离相机事件
            # self.update_direction(track, reference_point)

        # 更新车道拥堵事件
        count_stop = 0
        for track in self.tracks:
            if track.stoptime > 30:
                count_stop = count_stop + 1
        if count_stop >= 2:
            self.lane.traffic_congestion_event = True

        # 更新车辆违停事件
        for track in self.tracks:
            if track.parking_violation:
                self.lane.parking_violation_event = True
                break

        # if self.lane:
        # print("Entry event:", self.lane.entry_event)
        # print("Exit event:", self.lane.exit_event)
        # print("Traffic congestion event:", self.lane.traffic_congestion_event)
        # print("Parking violation event:", self.lane.parking_violation_event)

    def print_event(self):
        if self.lane.entry_event:
            print("lane_entry_event")
        if self.lane.exit_event:
            print("lane_exit_event")
        if self.lane.parking_violation_event:
            print("lane_parking_violation_event")
        if self.lane.traffic_congestion_event:
            print("lane_traffic_congestion_event")
        for track in self.tracks:
            if len(track.trace_v_list) > 1:
                if track.approaching_event:
                    print(f"id = {track.track_id}, v = {track.trace_v_list[-1]}, 靠近摄像头")
                else:
                    print(f"id = {track.track_id}, v = {track.trace_v_list[-1]}, 远离摄像头")

    def evalue(self, label_path):
        folder_path = 'label_json'
        # 读取JSON文件
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # lane_entry_capture_rate = 0.9999
        # parking_entry_accuracy = 0.9999
        # lane_direction_accuracy = 0.9999
        # parking_occupancy_accuracy = 0.9999
        #
        # parking_occupancy_up = 0  # 车位有无车辆
        # parking_occupancy_down = 0.1  # 分母
        #
        # lane_direction_up = 0  # 行驶方向
        # lane_direction_d = 0.1
        #
        # parking_entry_up = 0
        # parking_entry_d = 0.1
        #
        # lane_entry_capture_up = 0
        # lane_entry_capture_d = 0.1

        for label in data:
            if label['label_cls'] == 'Area Vehicle Count':
                self.parking_occupancy_down += 1
                if label['count'] == self.get_area(label['id']).count_car:
                    self.parking_occupancy_up += 1

            if label['label_cls'] == 'Vehicle Direction':
                self.lane_direction_d += 1
                if self.get_track(label['id']) == None:
                    self.lane_direction_up += 1
                elif label['direction'] and self.get_track(label['id']).approaching_event:
                    self.lane_direction_up += 1
                elif not label['direction'] and self.get_track(label['id']).leaving_event:
                    self.lane_direction_up += 1

            if label['label_cls'] == 'Enter Area':
                self.parking_entry_d += 1
                if self.get_track(label['id']) == None:
                    self.lane_direction_up += 1
                elif label['entered'] and self.get_track(label['id']).in_lane:
                    self.lane_direction_up += 1
                elif not label['entered'] and not self.get_track(label['id']).in_lane:
                    self.lane_direction_up += 1
            if label['label_cls'] == 'Exit Area':
                self.parking_entry_d += 1
            if label['label_cls'] == 'Enter Lane':
                self.lane_entry_capture_d += 1
            if label['label_cls'] == 'Exit Lane':
                self.lane_entry_capture_d += 1

        self.lane_direction_accuracy = self.lane_direction_up / self.lane_direction_d
        self.parking_occupancy_accuracy = self.parking_occupancy_up / self.parking_occupancy_down
        # 打印数据
        # print('parking_occupancy_down = ', parking_occupancy_down)
        # print('lane_direction_d = ', lane_direction_d)
        # print('')


class Border(object):
    '''
    一个区域的边界，主要是两个点以及一个判断哪边是区域内部的函数
    '''

    def __init__(self, point1, point2, color=(0, 255, 255)):
        '''

        :param point1: 第一个点
        :param point2: 第二个点
        '''
        self.point = [point1, point2]
        self.color = color
        self.yuancolor = color
        self.thickness = 2
        if point1[0] == point2[0]:  # 如果两个点的x坐标相同
            self.axis = 'y'  # y方向
        elif point1[1] == point2[1]:  # 如果两个点的y坐标相同
            self.axis = 'x'  # x方向
        else:
            self.axis = None  # 无法确定方向

    def __getitem__(self, index):
        return self.point[index]

    def intersect(self, point, previous_point):
        self.color = self.yuancolor
        self.thickness = 2
        isintersect = intersect(point, previous_point, self.point[0], self.point[1])
        if self.point[0][0] == self.point[1][0] and point[0] == self.point[0][0]:
            if point[1] > min(self.point[0][1], self.point[1][1]) and point[1] < max(self.point[0][1],
                                                                                     self.point[1][1]):
                if previous_point[0] != point[0]:
                    isintersect = True
        if self.point[0][0] == self.point[1][0] and previous_point[0] == self.point[0][0]:
            if previous_point[1] > min(self.point[0][1], self.point[1][1]) and previous_point[1] < max(self.point[0][1],
                                                                                                       self.point[1][
                                                                                                           1]):
                if previous_point[0] != point[0]:
                    isintersect = True
        if isintersect:
            self.color = (0, 0, 255)  # 如果相交则这条边界线画成红色
            self.thickness = 5
        return isintersect

    def draw_border(self, frame):
        point0 = coord_to_pixel(ax, self.point[0])
        point1 = coord_to_pixel(ax, self.point[1])
        cv2.line(frame, point0, point1, self.color, self.thickness)


class Area(object):

    def __init__(self, area_id, x1y1, x2y2, count_car=0, color=(0, 255, 255)):
        '''

        :param area_id: id
        :param count_car:此区域内现有的car的数量
        :param xyxy:此区域的坐标，我们使用左上右下两点的坐标表示矩形区域
        '''
        self.id, self.count_car, self.color = area_id, count_car, color
        self.rectangle_left_top = x1y1
        self.rectangle_right_bottom = x2y2
        self.border_lines = [Border(x1y1, (x1y1[0], x2y2[1]), self.color), Border(x1y1, (x2y2[0], x1y1[1]), self.color),
                             Border((x1y1[0], x2y2[1]), x2y2, self.color), Border((x2y2[0], x1y1[1]), x2y2, self.color)]
        self.centre = tuple((a + b) / 2 for a, b in zip(x1y1, x2y2))
        self.id_in = []  # 进入该area的tracks的ID
        self.in_out = []  # out该area的tracks的ID

    def update(self):
        pass

    def isenter(self, point, previous_point=None):
        '''
        判断point是否在area里面
        :param previous_point:
        :param point:
        :return:
        '''

        result = is_point_inside_rectangle(self.rectangle_left_top, self.rectangle_right_bottom, point)
        if previous_point is None:
            return result
        result = result and (
            not is_point_inside_rectangle(self.rectangle_left_top, self.rectangle_right_bottom, previous_point))
        return result

    def ifout(self, point, previous_point=None):
        result = not is_point_inside_rectangle(self.rectangle_left_top, self.rectangle_right_bottom, point)
        if previous_point is None:
            return result
        result = result and (
            is_point_inside_rectangle(self.rectangle_left_top, self.rectangle_right_bottom, previous_point))
        return result

    def intersect(self, track, point, previous_point):
        for border_line in self.border_lines:
            if border_line.intersect(point, previous_point):
                if self.isenter(point, previous_point):
                    self.count_car = self.count_car + 1
                    track.parking_id = self.id
                elif self.ifout(point, previous_point):
                    self.count_car = self.count_car - 1
                    track.parking_id = None

                return border_line

    def draw(self, frame):
        text_coord = subtract_tuples(coord_to_pixel(ax, (self.centre[0], self.centre[1])), (3, 3))
        cv2.putText(frame,
                    f"ID={self.id}",
                    text_coord, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)
        for i in range(len(self.border_lines)):
            self.border_lines[i].draw_border(frame)

    def point_to_border_distance(self, point: tuple) -> float:
        min_distance = float('inf')  # 初始化一个较大的数值作为最小距离（最短距离）

        # 遍历区域四个边界线
        for border_line in self.border_lines:
            p1, p2 = border_line.point  # 提取边界线的两个端点
            line_vec = np.array(p2) - np.array(p1)  # 求得边界线向量
            point_vec = np.array(point) - np.array(p1)  # 求得从端点p1到点point的向量

            line_len = np.linalg.norm(line_vec)  # 计算边界线长度
            line_unit_vec = line_vec / line_len  # 计算单位方向向量
            point_vec_proj_len = np.dot(point_vec, line_unit_vec)  # 计算点到端点的向量在边界线方向上投影的长度

            # 检查投影是否在线段上
            if 0 <= point_vec_proj_len <= line_len:
                # 如果投影在线段上，则计算垂足到边界线的垂直距离
                point_vec_proj = point_vec_proj_len * line_unit_vec  # 计算投影向量
                perpendicular_vec_len = np.linalg.norm(point_vec - point_vec_proj)  # 计算垂直于边界线方向的向量长度
                min_distance = min(min_distance, perpendicular_vec_len)
            else:
                # 如果投影不在线段上，则更新最小距离，将其设置为点到边界两个端点p1和p2的最小距离
                min_distance = min(min_distance, np.linalg.norm(np.array(p1) - np.array(point)))
                min_distance = min(min_distance, np.linalg.norm(np.array(p2) - np.array(point)))

        return min_distance


class Lane(Area):

    def __init__(self, lane_id, x1y1, x2y2, count_car=0, lane_direction='y'):
        '''
        进出事件：在这些示例代码中，entry_event 和 exit_event 分别表示车辆进入和离开设定的多边形区域。
        车行方向：在这些示例代码中，approaching_event 表示车辆靠近相机（参考点），而 leaving_event 表示车辆远离相机（参考点）。
        entry_event：车辆进入多边形区域。
        exit_event：车辆离开多边形区域。
        approaching_event：车辆靠近相机（参考点）。
        leaving_event：车辆远离相机（参考点）
        :param lane_id:
        :param x1y1:
        :param x2y2:
        :param count_car:
        '''
        super(Lane, self).__init__(lane_id, x1y1, x2y2, count_car=0, color=(0, 0, 0))
        self.entry_event = False
        self.exit_event = False
        # self.approaching_event = False
        # self.leaving_event = False
        self.traffic_congestion_event = False
        self.parking_violation_event = False
        self.lane_direction = lane_direction

    def intersect(self, point, previous_point):
        '''
        与Area中的方法不同，车道区域我们只判断有没有轨迹与进出方向的车道边界相交，有停车位的两侧不做判断。
        :param point:
        :param previous_point:
        :return:
        '''
        for border_line in self.border_lines:
            if border_line.axis == self.lane_direction:
                continue
            if border_line.intersect(point, previous_point):
                if self.isenter(point):
                    self.count_car = self.count_car + 1
                else:
                    self.count_car = self.count_car - 1
                return border_line

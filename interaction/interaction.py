# -- coding: utf-8 --
import _thread

import cv2
import numpy as np

from interaction.audio_thread import play_sounds
from interaction.detect.detect import detect
from interaction.utils import compute_distance, compute_direction

# 方向列表:上8下2左4右6
action_list = {
    "8": "close",
    "2": "menu",
    "6": "forward",
    "4": "backward",
    "68": "copy",
    "62": "paste",
    "64": "start",  # 启动应用
    "46": "clear",  # 刷新
}


class Interactor:
    """
    用户交互模块：
    对经过Tracker加工过的结果再进行一次处理。
    1. 判断用户是否具有交互意图
    2. 鲁棒性增强
    3. 调用具体应用

    """

    def __init__(self, no_act_thres=15, stop_thres=20, stable_thres=32):
        # 设置
        self.no_act_thres = no_act_thres  # 可以容忍的错误帧数
        self.stop_thres = stop_thres  # 判断为停滞的移动距离
        self.stable_thres = stable_thres  # 判断为停滞触发的时间

        # 缓存
        self.pose = np.array([0, 0])  # 登记的响应手势(当前版本下，仅支持单一手势的运动组合，当变换手势时将取消动作)
        self.pose_will = np.array([0, 0])  # 即将转变的手势，作为缓存
        self.direction_list = ["", ""]  # 方向列表，添加上8下2左4右6
        self.track = ([], [])  # 响应追踪路径 -> 画图
        self.stable_time = np.array([0, 0])  # 累计停滞时间
        self.no_active = np.array([0, 0])  # 忽略帧数累计
        self.pose_end = np.array([0, 0])  # 标记语音输出
        self.direction_intent = np.array([0, 0])  # 移动方向预测

        # 音效模块
        self.share_act = []
        _thread.start_new_thread(play_sounds, (self.share_act,))  # 启动声音提示线程

        # 应用模块
        self.app_dict = {
            "mouse": None,
            "img": None,
            "left_top": None,
            "right_bottom": None,
            "result": None,
        }  # 共享信息
        self.app_start = False
        # 启动应用模块线程，传入app_dict，线程共享该变量
        _thread.start_new_thread(detect, (self.app_dict,))  # 启动图像识别线程

    def interact(self, im0, order):
        """
        order[[x,y,p],[x,y,p]]，已经以id为序列存储
        x,y,p分别为x坐标，y坐标，手势编号
        """
        # print('interact launched')
        for i, o in enumerate(order):
            # print('pose ', o[2])
            # 终止手势：手势5、8、10---------------------------------------
            if o[2] == 10 or o[2] == 8 or o[2] == 5:  # 清空内存并发出指令
                self.action(i, im0)
                self.__clear_cache(i)

            # elif o[2] == 5:  # 清空内存但不发出指令
            #     self.__clear_cache(i)

            # 无手势或不是响应手势或无检测目标或手势改变---------------------------------------

            elif o[2] != 1 and o[2] != 2 and o[2] != 3 and o[2] != 11 or o[2] != self.pose_will[i] or o[0] == 0:
                # print(f'pose will {i}: {self.pose_will[i]}, pose{i}: {o[2]}')
                if self.no_active[i] == self.no_act_thres:
                    self.__clear_cache(i)
                    # print(f'clear cache {i}')
                else:
                    self.no_active[i] += 1
                    self.pose_will[i] = o[2]  # 通过检测阈值才能真正地响应
                # print(f'no_active[{i}]', self.no_active[i])

            # 响应手势---------------------------------------
            elif o[2] == 1 or o[2] == 2 or o[2] == 3:
                self.no_active[i] = 0
                self.pose[i] = o[2]  # 更新状态

                if o[2] == 1 or o[2] == 2 or o[2] == 3 and not self.pose_end[i]:  # 对于1,2,3手势做实时检测
                    # 没有移动过
                    if not self.direction_list[i]:
                        if self.stable_time[i] == 1:
                            self.share_act.append("click")  # 点击
                        elif self.stable_time[i] == self.stable_thres - 1:
                            self.share_act.append("press")  # 长按
                    # 移动过
                    elif len(self.track[i]) != 1:
                        self.share_act.append("drag")  # 拖动
                        self.pose_end[i] = 1

                if not len(self.track[i]):  # 首次记录，无坐标
                    self.track[i].append(o.tolist())

                # 移动距离小于移动阈值，判定为停滞，不更新方向，不更新轨迹
                if compute_distance(o[0], o[1], self.track[i][len(self.track[i]) - 1][0], self.track[i][len(self.track[i]) - 1][1]) < self.stop_thres ** 2:
                    if self.stable_time[i] < self.stable_thres and not len(self.direction_list[i]):
                        self.stable_time[i] += 1

                # 判定为移动
                else:
                    direction = compute_direction(o[0], o[1], self.track[i][len(self.track[i]) - 1][0],
                                                  self.track[i][len(self.track[i]) - 1][1])
                    # 记录响应轨迹的坐标
                    self.track[i].append(o.tolist())

                    # print(self.direction_list[i])

                    # 首次移动，无方向，直接赋值给方向列表
                    if not self.direction_list[i]:
                        self.direction_list[i] = str(direction)

                    if self.direction_intent[i] == direction:
                        # 当前方向为新方向则更新
                        if not self.direction_list[i].endswith(str(direction)):
                            self.direction_list[i] = self.direction_list[i] + str(direction)
                    else:
                        self.direction_intent[i] = direction

                    self.stable_time[i] = 0  # 当移动时不对停滞记录进行更新(保持充能圈)

                # 应用获取点击位置信息done
                if self.app_start and o[2] == 3:  # 只在应用启动，且手势为3时添加手指响应位置信息
                    self.app_dict["mouse"] = [o[0], o[1]]

            # 响应手势11
            elif o[2] == 11:
                print('capture! pose 11')
                self.no_active[i] = 0
                self.pose[i] = o[2]  # 更新状态
                self.action(i, im0)

            else:
                print("wrong in interaction")

            # 绘制移动轨迹
            if len(self.track[i]):  # 判断是否有记录
                last = None  # 创建变量存储上一个循环的记录
                for t, p in enumerate(self.track[i]):  # 遍历每个记录
                    if last is None:  # 第一个记录
                        last = (p[0], p[1])
                    else:
                        cv2.line(im0, last, (p[0], p[1]), (240, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                        last = (p[0], p[1])

                fill_cnt = (self.stable_time[i] / self.stable_thres) * 360  # 计算充能圈的终止角度
                cv2.circle(im0, (o[0], o[1]), 5, (0, 150, 255), -1)
                # 绘制充能圈
                if 0 < fill_cnt < 360:
                    cv2.ellipse(im0, (o[0], o[1]), (self.stop_thres, self.stop_thres), 0, 0, fill_cnt, (255, 255, 0), 2)
                else:
                    cv2.ellipse(im0, (o[0], o[1]), (self.stop_thres, self.stop_thres), 0, 0, fill_cnt, (0, 150, 255), 4)

                cv2.line(im0, (o[0], o[1]), last, (240, 255, 255), thickness=2, lineType=cv2.LINE_AA)

            # 绘制轨迹方向说明
            if len(self.direction_list[i]):
                detail = "valid:"
                for s in self.direction_list[i]:
                    if s == '2':
                        detail += 'down,'
                    elif s == '4':
                        detail += 'left,'
                    elif s == '6':
                        detail += 'right,'
                    elif s == '8':
                        detail += 'up,'
                # 图片，添加的文字，左上角坐标(整数)，字体，字体大小，颜色，字体粗细
                cv2.putText(im0, detail, (0, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        # 画框及显示框内的图片
        if self.app_dict["left_top"] is not None and self.app_dict["right_bottom"] is not None:
            # 画框
            cv2.rectangle(im0, (self.app_dict["left_top"][0], self.app_dict["left_top"][1]),
                          (self.app_dict["right_bottom"][0], self.app_dict["right_bottom"][1]),
                          (0, 255, 127), 2)
            if self.app_dict["result"] is not None:
                cv2.putText(im0, self.app_dict["result"]["class"],
                            (self.app_dict["left_top"][0], self.app_dict["left_top"][1]),
                            cv2.FONT_HERSHEY_PLAIN, 2, [255, 0, 255], 2)
            # 显示框内图片
            img_cut = im0[int(self.app_dict["left_top"][1]):int(self.app_dict["right_bottom"][1]),
                  int(self.app_dict["left_top"][0]):int(self.app_dict["right_bottom"][0])]  # 先切y轴，再切x轴
            # cv2.imshow('img_cut', img_cut)
            h_ori, w_ori = im0.shape[:2]  # 高，宽，图像通道数量
            h, w = 128, 128  # 调整框内图片大小
            # 若方框大小符合要求，则显示在右下角
            if int(self.app_dict["right_bottom"][1]) - int(self.app_dict["left_top"][1]) > 1 and int(self.app_dict["right_bottom"][0]) - int(self.app_dict["left_top"][0]) > 1:
                img_cut = cv2.resize(img_cut, (w, h))
                im0[(h_ori-h):h_ori, (w_ori-w):w_ori] = img_cut

    def __clear_cache(self, i):
        self.track[i].clear()  # 清除轨迹记录
        self.direction_list[i] = ""  # 清除方向记录
        self.no_active[i] = 0
        self.pose[i] = 0
        self.pose_will[i] = 0
        self.stable_time[i] = 0
        self.pose_end[i] = 0

    def action(self, i, img=None):
        if (self.pose[i] == 2 or self.pose[i] == 3) and self.direction_list[i] in action_list.keys():  # 需要登记指令
            self.share_act.append(action_list[self.direction_list[i]])

            if self.direction_list[i] == "64":  # 启动应用
                self.app_start = True
                # print("\n\n启动应用\n\n")
            elif self.direction_list[i] == "8":  # 关闭应用
                self.app_start = False
                # print("\n\n关闭应用\n\n")
            elif self.direction_list[i] == "46":  # 清空以新建新的分类任务
                self.app_dict["result"] = None
                self.app_dict["left_top"] = self.app_dict["right_bottom"] = None
                # print("\n\n新建任务\n\n")

        if self.pose[i] == 11 and self.app_start and self.app_dict["result"] is None:  # 启动分类
            self.app_dict["img"] = img
            print("\n\n---启动分类---\n\n")

        # print("direction_list %d: " % i, self.direction_list[i])

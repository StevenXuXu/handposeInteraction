# -- coding: utf-8 --
import _thread

import cv2
import numpy as np

from interaction.audio_thread import play_piano
from interaction.detect.detect import detect
from interaction.utils import compute_distance, compute_direction, if_contain_circle, if_contain_button, press_white_key

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

# 键盘关键字
keys = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 'del'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'blk', 'ent'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '?', '!']]


# 钢琴音调
tunes = ['do', 're', 'mi', 'fa', 'so', 'la', 'si', 'do+', 're+', 'fa+', 'so+', 'la+']


# 定义键盘按钮
class KbdButton:
    def __init__(self, pos, text, size):
        self.pos = pos
        self.text = text
        self.size = size


# 定义钢琴按钮
class PianoButton:
    def __init__(self, pos, color, size, tune=None):
        self.pos = pos
        self.color = color
        self.size = size
        self.tune = tune


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
        self.pose_last = np.array([0, 0])  # 记录上一帧的手势
        self.direction_list = ["", ""]  # 方向列表，添加上8下2左4右6
        self.track = ([], [])  # 响应追踪路径 -> 画图
        self.stable_time = np.array([0, 0])  # 累计停滞时间
        self.no_active = np.array([0, 0])  # 忽略帧数累计
        self.pose_end = np.array([0, 0])  # 标记语音输出
        self.direction_intent = np.array([0, 0])  # 移动方向预测
        self.detected = np.array([0, 0])  # 框选图片是否检测过
        self.app_start = 3  # 当前在哪个应用
        self.menu = 0  # 是否在菜单模式
        self.menu_mouse = [320, 400]  # 菜单鼠标坐标
        self.delta_x = 0  # 坐标的x改变量
        self.delta_y = 0  # 坐标的y改变量
        self.buttonList = []  # 键盘按键表
        self.buttonUpdate = True  # 键盘按键是否需要更新
        self.kb_text = ""  # 输入的文本
        self.kb_pos = np.array([0, 0])  # 键盘初始位置
        self.last_pos = np.array([[-1, -1, -1], [-1, -1, -1]])  # 记录上一帧的食指位置
        self.piano_pos = np.array([0, 0])  # 钢琴初始位置
        self.pianoList = []  # 钢琴按键表
        self.pianoUpdate = True  # 钢琴按键是否需要更新

        # todo 音效模块
        self.share_act = []  # 音效
        self.piano_data = []  # 钢琴声音
        _thread.start_new_thread(play_piano, (self.piano_data,))  # 启动声音提示线程

        # 图像内容识别模块
        self.app_dict = {
            "mouse": None,
            "img": None,
            "left_top": None,
            "right_bottom": None,
            "result": None,
        }  # 共享信息
        # 启动应用模块线程，传入app_dict，线程共享该变量
        _thread.start_new_thread(detect, (self.app_dict,))  # 启动图像识别线程

    def interact(self, im0, order):
        """
        order[[x,y,p],[x,y,p]]，已经以id为序列存储
        x,y,p分别为x坐标，y坐标，手势编号
        """
        width = im0.shape[1]
        height = im0.shape[0]
        # print('interact launched')
        for i, o in enumerate(order):
            # todo 呼出菜单 done
            # 手势4呼出菜单(x)
            # 1.当手势4出现，即进入菜单模式，维持手势即可维持菜单
            # 2.点击“退出菜单”按钮或者手势出现改变且改变时间超过阈值即可退出菜单
            # 3.菜单以若干圆圈将食指尖作为圆心作扇型展开
            # if o[2] == 4:
            #     if self.pose_will[i] != o[2]:
            #         self.pose_will[i] = o[2]
            #         continue
            #     self.pose[i] == o[2]
            #
            # 下拉呼出菜单(o)
            # 1.下拉手势出现后进入菜单
            # 2.菜单中有若干圆圈按钮（如何确定位置？如何判断点击到？）
            # 3.在不同应用下菜单不同

            # self.pose = o[2]
            if self.menu:
                # 若非相应手势则清空前坐标信息
                self.pose[i] = o[2]
                if o[2] not in [1, 2, 3]:
                    self.track[i].clear()
                else:
                    # 更新横纵坐标的位置改变量
                    if len(self.track[i]):
                        self.delta_x = o[0] - self.track[i][len(self.track[i]) - 1][0]
                        self.delta_y = o[1] - self.track[i][len(self.track[i]) - 1][1]
                    self.track[i].append(o.tolist())
                    # 更新菜单光标的坐标
                    self.menu_mouse[0] += self.delta_x * 2
                    self.menu_mouse[1] += self.delta_y * 2
                    self.menu_mouse[0] = min(self.menu_mouse[0], width)
                    self.menu_mouse[0] = max(self.menu_mouse[0], 0)
                    self.menu_mouse[1] = min(self.menu_mouse[1], height)
                    self.menu_mouse[1] = max(self.menu_mouse[1], 0)
                continue

            # 终止手势：手势5、8、10---------------------------------------
            if o[2] == 10 or o[2] == 8 or o[2] == 5:  # 清空内存并发出指令
                self.action(i, im0)
                self.__clear_cache(i)

            # elif o[2] == 5:  # 清空内存但不发出指令
            #     self.__clear_cache(i)

            # 无手势或不是响应手势或无检测目标或手势改变---------------------------------------
            elif o[2] not in [1, 2, 3, 11] or o[2] != self.pose_will[i] or o[0] == 0:
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
                            if self.app_start == 1:
                                self.app_dict["mouse"] = [o[0], o[1]]
                    # 移动过
                    elif len(self.track[i]) != 1:
                        self.share_act.append("drag")  # 拖动
                        self.pose_end[i] = 1

                if not len(self.track[i]):  # 首次记录，无坐标
                    self.track[i].append(o.tolist())

                # 移动距离小于移动阈值，判定为停滞，不更新方向，不更新轨迹
                if compute_distance(o[0], o[1], self.track[i][len(self.track[i]) - 1][0],
                                    self.track[i][len(self.track[i]) - 1][1]) < self.stop_thres ** 2:
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
                if self.app_start == 1 and o[2] == 3:  # 只在应用1启动，且手势为3时添加手指响应位置信息
                    self.app_dict["mouse"] = [o[0], o[1]]

            # 响应手势11,开始目标检测
            elif o[2] == 11 and self.app_start == 1:
                # print('capture! pose 11')
                self.no_active[i] = 0
                self.pose[i] = o[2]  # 更新状态
                if not self.detected[i]:
                    self.action(i, im0)
                    self.detected[i] = True

            else:
                # print("wrong in interaction")
                pass

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

        # 在应用1下，画框及显示框内的图片
        if self.app_dict["left_top"] is not None and self.app_dict["right_bottom"] is not None and self.app_start == 1:
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
            if int(self.app_dict["right_bottom"][1]) - int(self.app_dict["left_top"][1]) > 1 and int(
                    self.app_dict["right_bottom"][0]) - int(self.app_dict["left_top"][0]) > 1:
                img_cut = cv2.resize(img_cut, (w, h))
                im0[(h_ori - h):h_ori, (w_ori - w):w_ori] = img_cut

        # 在应用2下
        if self.app_start == 2:
            # todo 虚拟键盘 done
            # 键盘按钮有26个英文字母、空格、逗号、句号、退格键、回车键组成
            # 功能：1.选取字母显示在文本框中；2.可以拖动虚拟键盘以调整其位置
            # print("SHAPE: ", im0.shape)
            kb_h, kb_w = int(height * 0.6), int(width * 0.6)
            if self.kb_pos[0] == 0 and self.kb_pos[1] == 0:
                self.kb_pos = np.array([int(height * 0.3), int(width / 2 - kb_w / 2)])
            b_w = int(kb_w / 11)
            b_h = int(kb_h * 0.7 / 3)

            for i, o in enumerate(order):
                if o[2] in [10, 11]:
                    # 更新横纵坐标的位置改变量
                    if self.pose_last[i] not in [10, 11]:
                        self.last_pos = np.array([[-1, -1, -1], [-1, -1, -1]])
                    if self.last_pos[i][0] != -1 and self.last_pos[i][1] != -1:
                        self.delta_x = o[0] - self.last_pos[i][0]
                        self.delta_y = o[1] - self.last_pos[i][1]
                    self.last_pos[i] = o
                    # 更新键盘坐标
                    self.kb_pos[0] += self.delta_x
                    self.kb_pos[1] += self.delta_y
                    self.kb_pos[0] = min(self.kb_pos[0], width - kb_w)
                    self.kb_pos[0] = max(self.kb_pos[0], 0)
                    self.kb_pos[1] = min(self.kb_pos[1], height - kb_h)
                    self.kb_pos[1] = max(self.kb_pos[1], 0)
                    self.buttonUpdate = True

            # 更新键盘按键坐标
            if self.buttonUpdate:
                self.buttonList.clear()
                for j in range(len(keys)):
                    for x, key in enumerate(keys[j]):
                        # 循环创建buttonList对象列表
                        self.buttonList.append(
                            KbdButton((self.kb_pos[0] + x * b_w + 5, self.kb_pos[1] + int(kb_h * 0.3) + j * b_h), key,
                                      (b_w, b_h)))
                self.buttonUpdate = False

            # 绘制半透明背景以及键盘并判断是否悬停在某个按键上
            zeros = np.zeros(im0.shape, dtype=np.uint8)
            zeros_mask = cv2.rectangle(zeros, (int(self.kb_pos[0]), int(self.kb_pos[1])),
                                       (int(self.kb_pos[0] + kb_w), int(self.kb_pos[1] + kb_h)),
                                       color=(255, 255, 255), thickness=-1)  # thickness=-1 表示矩形框内颜色填充
            flag = 1
            for button in self.buttonList:
                x, y = button.pos
                w, h = button.size
                # print("ORDER: ", order)
                # print("BUTTON: ", button.pos)
                for i, o in enumerate(order):
                    if if_contain_button(button, o[0], o[1]):
                        cv2.rectangle(zeros_mask, button.pos, (x + w, y + h), (15, 25, 30), cv2.FILLED)
                        cv2.putText(zeros_mask, button.text, (x + 10, y + 40), cv2.FONT_HERSHEY_PLAIN, 1,
                                    (255, 255, 255), 2)
                        if o[2] == 3 and self.pose_last[i] != o[2]:
                            if button.text == 'blk':
                                self.kb_text += ' '
                            elif button.text == 'del':
                                self.kb_text = self.kb_text[:-1]
                            elif button.text == 'ent':
                                self.kb_text = ""
                            else:
                                self.kb_text += button.text
                        break
                    else:
                        if flag == 1:
                            cv2.rectangle(zeros_mask, button.pos, (x + w, y + h), (0, 150, 255), cv2.FILLED)
                        else:
                            cv2.rectangle(zeros_mask, button.pos, (x + w, y + h), (100, 190, 255), cv2.FILLED)
                        cv2.putText(zeros_mask, button.text, (x + 10, y + 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
                flag ^= 1

            self.pose_last[0] = order[0][2]
            self.pose_last[1] = order[1][2]
            cv2.putText(zeros_mask, self.kb_text, (self.kb_pos[0], self.kb_pos[1] + int(kb_h * 0.3) - 20),
                        cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 2)
            cv2.line(zeros_mask, (self.kb_pos[0] + 5, int(self.kb_pos[1] + kb_h * 0.3) - 10),
                     (self.kb_pos[0] + kb_w - 5, int(self.kb_pos[1] + kb_h * 0.3) - 10),
                     (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            im0 = cv2.addWeighted(im0, 1, zeros_mask, 0.6, 0)

        if self.app_start == 3:
            # todo 虚拟钢琴应用
            piano_h, piano_w = int(height * 0.6), int(width * 0.6)
            if self.piano_pos[0] == 0 and self.piano_pos[1] == 0:
                self.piano_pos = np.array([int(height * 0.3), int(width / 2 - piano_w / 2)])
            b_white_w = int(piano_w / 7)
            b_white_h = piano_h
            b_black_w = int(b_white_w * 0.7)
            b_black_h = int(b_white_h * 0.6)

            for i, o in enumerate(order):
                if o[2] in [10, 11]:
                    # 更新横纵坐标的位置改变量
                    if self.pose_last[i] not in [10, 11]:
                        self.last_pos = np.array([[-1, -1, -1], [-1, -1, -1]])
                    if self.last_pos[i][0] != -1 and self.last_pos[i][1] != -1:
                        self.delta_x = o[0] - self.last_pos[i][0]
                        self.delta_y = o[1] - self.last_pos[i][1]
                    self.last_pos[i] = o
                    # 更新钢琴坐标
                    self.piano_pos[0] += self.delta_x
                    self.piano_pos[1] += self.delta_y
                    self.piano_pos[0] = min(self.piano_pos[0], width - piano_w)
                    self.piano_pos[0] = max(self.piano_pos[0], 0)
                    self.piano_pos[1] = min(self.piano_pos[1], height - piano_h)
                    self.piano_pos[1] = max(self.piano_pos[1], 0)
                    self.pianoUpdate = True

            # 更新钢琴按键坐标
            if self.pianoUpdate:
                self.pianoList.clear()
                for i in range(7):
                    self.pianoList.append(PianoButton((self.piano_pos[0] + i * b_white_w, self.piano_pos[1]),
                                                      'white',
                                                      (b_white_w, b_white_h)))
                for i in range(1, 7):
                    if i == 3:
                        continue
                    self.pianoList.append(PianoButton((self.pianoList[i].pos[0] - int(b_black_w * 0.5), self.piano_pos[1]),
                                                      'black',
                                                      (b_black_w, b_black_h)))
                for i in range(12):
                    self.pianoList[i].tune = tunes[i]
                self.pianoUpdate = False

            zeros_mask = np.zeros(im0.shape, dtype=np.uint8)

            for button in self.pianoList:
                x, y = button.pos
                w, h = button.size
                # print("ORDER: ", order)
                # print("BUTTON: ", button.pos)
                for i, o in enumerate(order):
                    if button.color == 'black' and if_contain_button(button, o[0], o[1]) or button.color == 'white' and press_white_key(button, o[0], o[1], b_black_h):
                        cv2.rectangle(zeros_mask, button.pos, (x + w, y + h), (15, 25, 30), cv2.FILLED)
                        if o[2] == 3 and self.pose_last[i] != o[2]:
                            self.piano_data.append(button.tune)
                        break
                    else:
                        if button.color == 'black':
                            cv2.rectangle(zeros_mask, button.pos, (x + w, y + h), (50, 50, 50), cv2.FILLED)
                        else:
                            cv2.rectangle(zeros_mask, button.pos, (x + w, y + h), (180, 180, 180), cv2.FILLED)

            self.pose_last[0] = order[0][2]
            self.pose_last[1] = order[1][2]
            im0 = cv2.addWeighted(im0, 1, zeros_mask, 0.6, 0)

        # 若在菜单模式下，绘制相关按钮以及光标，并判断是否命中按钮
        if self.menu:
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            # cv2.rectangle(img, pt1, pt2, color, thickness, lineType, shift )
            # color: BGR
            # cv2.rectangle(im0, (0, 0), (width, height), (0, 100, 0), -1)

            # 绘制半透明背景
            zeros = np.zeros(im0.shape, dtype=np.uint8)
            zeros_mask = cv2.rectangle(zeros, (0, 0), (width, height),
                                       color=(255, 255, 255), thickness=-1)  # thickness=-1 表示矩形框内颜色填充
            im0 = cv2.addWeighted(im0, 1, zeros_mask, 0.6, 0)

            if self.app_start == 0:  # 没有开启应用
                c1_x, c1_y = int(width / 5), int(height / 5) * 2  # 关闭菜单
                c2_x, c2_y = int(width / 5) * 2, int(height / 5) * 2  # 启动应用1
                c3_x, c3_y = int(width / 5) * 3, int(height / 5) * 2  # 启动应用2
                c4_x, c4_y = int(width / 5) * 4, int(height / 5) * 2  # 启动应用3
                c_radius = 50

                cv2.circle(im0, (c1_x, c1_y), c_radius, (0, 204, 0), -1)
                cv2.putText(im0, 'back', (c1_x - 30, c1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                cv2.circle(im0, (c2_x, c2_y), c_radius, (0, 204, 0), -1)
                cv2.putText(im0, 'detect', (c2_x - 33, c2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                cv2.circle(im0, (c3_x, c3_y), c_radius, (0, 204, 0), -1)
                cv2.putText(im0, 'kbd', (c3_x - 20, c3_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                cv2.circle(im0, (c4_x, c4_y), c_radius, (0, 204, 0), -1)
                cv2.putText(im0, 'piano', (c4_x - 32, c4_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

                cv2.circle(im0, (self.menu_mouse[0], self.menu_mouse[1]), 5, (153, 0, 204), -1)
                cv2.circle(im0, (self.menu_mouse[0], self.menu_mouse[1]), 5, (102, 0, 255), 2)

                # print("POSE: ", self.pose)
                if self.pose[0] not in [1, 2, 3] and self.pose[1] not in [1, 2, 3]:
                    # 关闭菜单
                    if if_contain_circle(c1_x, c1_y, c_radius, self.menu_mouse[0], self.menu_mouse[1]):
                        self.close_menu()

                    # 打开应用1
                    if if_contain_circle(c2_x, c2_y, c_radius, self.menu_mouse[0], self.menu_mouse[1]):
                        self.app_start = 1
                        self.close_menu()

                    # 打开应用2
                    if if_contain_circle(c3_x, c3_y, c_radius, self.menu_mouse[0], self.menu_mouse[1]):
                        self.app_start = 2
                        self.close_menu()

                    # 打开应用3
                    if if_contain_circle(c4_x, c4_y, c_radius, self.menu_mouse[0], self.menu_mouse[1]):
                        self.app_start = 3
                        self.close_menu()

            elif self.app_start == 1:  # 打开了应用1
                c1_x, c1_y = int(width / 4), int(height / 5) * 2  # 关闭菜单
                c2_x, c2_y = int(width / 4) * 2, int(height / 5) * 2  # 关闭应用1
                c3_x, c3_y = int(width / 4) * 3, int(height / 5) * 2  # 新建任务
                c_radius = 50

                cv2.circle(im0, (c1_x, c1_y), c_radius, (0, 204, 0), -1)
                cv2.putText(im0, 'back', (c1_x - 30, c1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                cv2.circle(im0, (c2_x, c2_y), c_radius, (0, 204, 0), -1)
                cv2.putText(im0, 'close', (c2_x - 30, c2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                cv2.circle(im0, (c3_x, c3_y), c_radius, (0, 204, 0), -1)
                cv2.putText(im0, 'reset', (c3_x - 30, c3_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

                cv2.circle(im0, (self.menu_mouse[0], self.menu_mouse[1]), 5, (153, 0, 204), -1)
                cv2.circle(im0, (self.menu_mouse[0], self.menu_mouse[1]), 5, (102, 0, 255), 2)

                if self.pose[0] not in [1, 2, 3] and self.pose[1] not in [1, 2, 3]:
                    # 关闭菜单
                    if if_contain_circle(c1_x, c1_y, c_radius, self.menu_mouse[0], self.menu_mouse[1]):
                        self.close_menu()

                    # 关闭应用1
                    elif if_contain_circle(c2_x, c2_y, c_radius, self.menu_mouse[0], self.menu_mouse[1]):
                        self.app_start = 0
                        self.app_dict["result"] = None
                        self.app_dict["left_top"] = self.app_dict["right_bottom"] = None
                        self.close_menu()

                    # 清空以新建新的分类任务
                    elif if_contain_circle(c3_x, c3_y, c_radius, self.menu_mouse[0], self.menu_mouse[1]):
                        self.app_dict["result"] = None
                        self.app_dict["left_top"] = self.app_dict["right_bottom"] = None
                        self.close_menu()

            elif self.app_start == 2:  # 打开了应用2
                c1_x, c1_y = int(width / 3), int(height / 5) * 2  # 关闭菜单
                c2_x, c2_y = int(width / 3) * 2, int(height / 5) * 2  # 关闭应用2
                c_radius = 50

                cv2.circle(im0, (c1_x, c1_y), c_radius, (0, 204, 0), -1)
                cv2.putText(im0, 'back', (c1_x - 30, c1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                cv2.circle(im0, (c2_x, c2_y), c_radius, (0, 204, 0), -1)
                cv2.putText(im0, 'close', (c2_x - 30, c2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

                cv2.circle(im0, (self.menu_mouse[0], self.menu_mouse[1]), 5, (153, 0, 204), -1)
                cv2.circle(im0, (self.menu_mouse[0], self.menu_mouse[1]), 5, (102, 0, 255), 2)

                if self.pose[0] not in [1, 2, 3] and self.pose[1] not in [1, 2, 3]:
                    # 关闭菜单
                    if if_contain_circle(c1_x, c1_y, c_radius, self.menu_mouse[0], self.menu_mouse[1]):
                        self.close_menu()

                    # 关闭应用2
                    elif if_contain_circle(c2_x, c2_y, c_radius, self.menu_mouse[0], self.menu_mouse[1]):
                        self.app_start = 0
                        self.buttonUpdate = True
                        self.buttonList.clear()
                        self.kb_text = ""
                        self.kb_pos = np.array([0, 0])
                        self.last_pos = np.array([[-1, -1, -1], [-1, -1, -1]])
                        self.close_menu()

            elif self.app_start == 3:  # 打开了应用3
                c1_x, c1_y = int(width / 3), int(height / 5) * 2  # 关闭菜单
                c2_x, c2_y = int(width / 3) * 2, int(height / 5) * 2  # 关闭应用3
                c_radius = 50

                cv2.circle(im0, (c1_x, c1_y), c_radius, (0, 204, 0), -1)
                cv2.putText(im0, 'back', (c1_x - 30, c1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                cv2.circle(im0, (c2_x, c2_y), c_radius, (0, 204, 0), -1)
                cv2.putText(im0, 'close', (c2_x - 30, c2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

                cv2.circle(im0, (self.menu_mouse[0], self.menu_mouse[1]), 5, (153, 0, 204), -1)
                cv2.circle(im0, (self.menu_mouse[0], self.menu_mouse[1]), 5, (102, 0, 255), 2)

                if self.pose[0] not in [1, 2, 3] and self.pose[1] not in [1, 2, 3]:
                    # 关闭菜单
                    if if_contain_circle(c1_x, c1_y, c_radius, self.menu_mouse[0], self.menu_mouse[1]):
                        self.close_menu()

                    # 关闭应用3
                    elif if_contain_circle(c2_x, c2_y, c_radius, self.menu_mouse[0], self.menu_mouse[1]):
                        self.app_start = 0
                        self.pianoUpdate = True
                        self.pianoList.clear()
                        self.piano_pos = np.array([0, 0])
                        self.last_pos = np.array([[-1, -1, -1], [-1, -1, -1]])
                        self.close_menu()

        return im0

    def __clear_cache(self, i):
        self.track[i].clear()
        self.direction_list[i] = ""
        self.no_active[i] = 0
        self.pose[i] = 0
        self.pose_will[i] = 0
        self.stable_time[i] = 0
        self.pose_end[i] = 0
        self.detected[i] = 0
        # self.menu = 0
        # self.menu_mouse = [320, 480]
        self.delta_x = 0
        self.delta_y = 0

    def action(self, i, img=None):
        if (self.pose[i] == 2 or self.pose[i] == 3) and self.direction_list[i] in action_list.keys():  # 需要登记指令
            self.share_act.append(action_list[self.direction_list[i]])
            # if self.direction_list[i] == "64":  # 启动应用
            #     self.app_start = True
            # print("\n\n启动应用\n\n")
            # if self.direction_list[i] == "8":  # 关闭应用
            #     self.app_start = False
            #     # print("\n\n关闭应用\n\n")
            # elif self.direction_list[i] == "46":  # 清空以新建新的分类任务
            #     self.app_dict["result"] = None
            #     self.app_dict["left_top"] = self.app_dict["right_bottom"] = None
            #     # print("\n\n新建任务\n\n")
            if self.direction_list[i] == "2":  # 打开菜单
                self.menu_mouse[0] = int(img.shape[1] / 2)
                self.menu_mouse[1] = int(img.shape[0] / 5) * 4
                self.open_menu()
                print('-----菜单打开-----')

        if self.pose[i] == 11 and self.app_start == 1 and self.app_dict["result"] is None:  # 启动分类
            self.app_dict["img"] = img
            print("\n\n---启动分类---\n\n")

        # print("direction_list %d: " % i, self.direction_list[i])

    # 打开菜单
    def open_menu(self):
        self.track[0].clear()
        self.track[1].clear()
        self.menu = True

    # 关闭菜单
    def close_menu(self):
        self.track[0].clear()
        self.track[1].clear()
        self.menu = 0
        self.delta_x = 0
        self.delta_y = 0

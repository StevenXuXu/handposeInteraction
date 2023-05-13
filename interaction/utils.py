# -- coding: utf-8 --


# 计算距离
def compute_distance(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


# 计算运动方向
def compute_direction(x1, y1, x2, y2):
    """
    p2旧 -> p1新
    若运动轨迹为斜的，则比较x轴方向的改变量和y轴方向的改变量
    """
    dx = x1 - x2
    dy = y1 - y2
    if -dx < dy < dx and dx > 0:  # 向右
        return 6
    elif dx < dy < -dx and dx < 0:  # 向左
        return 4
    elif -dy < dx < dy and dy > 0:  # 向下
        return 2
    elif dy < dx < -dy and dy < 0:  # 向上
        return 8
    else:  # 完全没有移动
        return 0


# 判断是否在圆内
def if_contain_circle(cx, cy, r, x, y):
    return (cx - x) ** 2 + (cy - y) ** 2 <= r ** 2


# 判断是否在矩形内
def if_contain_rectangle(rx, ry, w, h, x, y):
    return rx <= x <= rx + w and ry <= y <= ry + h


# 判断是否在按键内
def if_contain_button(button, x, y):
    return button.pos[0] <= x <= button.pos[0] + button.size[0] and button.pos[1] <= y <= button.pos[1] + button.size[1]


# 判断是否按白键
def press_white_key(button, x, y, black_h):
    return button.pos[0] <= x <= button.pos[0] + button.size[0] and button.pos[1] + black_h <= y <= button.pos[1] + button.size[1]
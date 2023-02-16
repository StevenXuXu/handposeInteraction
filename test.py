import _thread
import time
from playsound import playsound
import os
import cv2


# def recursive_listdir(path):
#
#     files = os.listdir(path)
#     for file in files:
#         print(file)
#         file_path = os.path.join(path, file)
#
#         if os.path.isfile(file_path):
#             print(file)
#
#         elif os.path.isdir(file_path):
#             recursive_listdir(file_path)
#
#
# recursive_listdir(r'./inference/input/pose_setting')
#
#
# def solve(app_dict):
#     while True:
#         time.sleep(0.01)
#         if app_dict['img'] is not None:
#             # playsound('handposeInteraction/inference/audio/click.mp3')
#             print('yes\n')
#             app_dict['img'] = None
#
#
# if __name__ == '__main__':
#     app_dict = {
#         "mouse": None,
#         "img": None,
#         "left_top": None,
#         "right_bottom": None,
#         "result": None,
#     }
#     _thread.start_new_thread(solve, (app_dict,))
#     while True:
#         time.sleep(0.01)
#         app_dict['img'] = input('input: ')

# img = cv2.imread('inference/input/pose_setting/1/1.jpg')
# img = cv2.resize(img, (640, 640))
# H, W, bytesPerComponent = img.shape
# h, w = 100, 200
# img_cut = img[639:640, 0:2]
# img_cut = cv2.resize(img_cut, (w, h))
# img[(640-h):640, (640-w):640] = img_cut
# print(H, W)
#
# cv2.imshow('test', img)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()

res = ''
if not len(res):
    print('NOTHING')
res += 'haha'
print(res)

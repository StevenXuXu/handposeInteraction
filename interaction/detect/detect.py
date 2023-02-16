# -- coding: utf-8 --
import json
import os
import time
import numpy as np

import torch
from playsound import playsound
from torchvision import transforms
from PIL import Image

from interaction.detect.model import resnet34
from interaction.detect.models.common import DetectMultiBackend
from interaction.detect.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from interaction.detect.utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
from interaction.detect.utils.plots import Annotator, colors, save_one_box


def detect(app_dict):
    weights = 'interaction/detect/yolov5l.pt'
    bs = 1

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(weights=weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = (640, 640)

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    print('-----------------------detect model ready-----------------------------')

    """
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    json_path = 'interaction/detect/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = "inference/weights/object_weight/resnet34-best.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    """

    while True:
        time.sleep(0.01)

        # todo 优化清除逻辑

        # 有需要预测的图像，并且没有检测过
        if app_dict["img"] is not None and not app_dict["result"]:
            try:
                assert app_dict["left_top"] is not None and app_dict["right_bottom"] is not None

                img = app_dict["img"].copy()
                # 根据所选框切割图片
                im0 = img[int(app_dict["left_top"][1]):int(app_dict["right_bottom"][1]),
                      int(app_dict["left_top"][0]):int(app_dict["right_bottom"][0])]  # 先切y轴，再切x轴
                # print('-----1 finished------')
                assert im0 is not None, f'Image Not Found'
                im = letterbox(im0, 640, stride=32, auto=True)[0]  # padded resize
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous
                # print('-----2 finished------')

                result = ""
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    pred = model(im, visualize=False)

                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred)

                for i, det in enumerate(pred):  # per image
                    seen += 1
                    im0 = im0.copy()
                    # s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    annotator = Annotator(im0, line_width=3, example=str(names))  # 注释器
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        print(f"{i}--------------")
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per classqq
                            # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            print(f"{n} {names[int(c)]}{'s' * (n > 1)}, ")
                            result += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # print(f"detect: {result}")
                """
                # 预测
                img = app_dict["img"].copy()
                # 根据所选框切割图片
                img = img[int(app_dict["left_top"][1]):int(app_dict["right_bottom"][1]),
                      int(app_dict["left_top"][0]):int(app_dict["right_bottom"][0])]  # 先切y轴，再切x轴
                
                # print(img.shape)
                # print(app_dict["left_top"], app_dict["right_bottom"])
                
                img = Image.fromarray(img)  # ndarray->pil
                img = data_transform(img)
                img = torch.unsqueeze(img, dim=0)
                with torch.no_grad():
                    # predict class
                    output = torch.squeeze(model(img.to(device))).cpu()
                    predict = torch.softmax(output, dim=0)
                    predict_cla = torch.argmax(predict).numpy()

                result = class_indict[str(predict_cla)]
                score = "{:.2}".format(predict[predict_cla].numpy())

                print("classification success. {} with {:.3}".format(result, score))
                """

                if not len(result):
                    result = 'nothing'
                app_dict["result"] = {
                    "class": result,
                    # "score": score
                }
                app_dict["img"] = None
                # 语音提示
                try:
                    playsound('inference/audio/' + result + '.mp3')
                except Exception as e:
                    print(e)

            except Exception as e:
                print(e)
                try:
                    playsound('inference/audio/' + '' + '.mp3')  # todo 识别失败语音
                except Exception as e:
                    print(e)

        # 更新手绘方框的两个定位点坐标
        else:
            if app_dict["mouse"] is not None:
                x = app_dict["mouse"][0]
                y = app_dict["mouse"][1]

                # print(app_dict["mouse"])
                # print(app_dict["left_top"], app_dict["right_bottom"])

                if app_dict["left_top"] is not None and app_dict["right_bottom"] is not None:
                    if x < app_dict["left_top"][0]:
                        app_dict["left_top"][0] = x
                        # print("t1")
                    if y < app_dict["left_top"][1]:
                        app_dict["left_top"][1] = y
                        # print("t2")
                    if x > app_dict["right_bottom"][0]:
                        app_dict["right_bottom"][0] = x
                        # print("t3")
                    if y > app_dict["right_bottom"][1]:
                        app_dict["right_bottom"][1] = y
                        # print("t4")

                    # print(app_dict["left_top"], app_dict["right_bottom"])

                else:
                    app_dict["left_top"] = [x, y]
                    app_dict["right_bottom"] = [x, y]

                app_dict["mouse"] = None
            else:
                continue

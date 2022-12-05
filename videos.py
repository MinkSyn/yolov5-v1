import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Load PATH
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

ROOT_p = os.getcwd()

import torch
import torch.backends.cudnn as cudnn
from facenet_pytorch import InceptionResnetV1

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                       increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from general import liner_InceptionResnetv1, all_KNN, load_model, classify_FC, Net, class_train


@torch.no_grad()
def run(
        weights='',  # model.pt path(s)
        source='',  # file/dir/URL/glob, 0 for webcam
        data='',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='',  # save results to project/name
        name='Face_Mask',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        last_layer='', 
):  
    source = str(source)
    resnet, emb_not_mask, emb_mask = load_model(last_layer=last_layer)

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size


    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    count =0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0, frame = path[i], im0s[i].copy(), dataset.count # p là định danh đường dẫn
            count+=1
            p = Path(p)  # to Path
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for x1, y1, x2, y2, conf, cls in reversed(det):
                    w, h = int(x2 - x1), int(y2 - y1)
                    x, y = int((x1+x2)/2 - w/2), int((y1+y2)/2 - h/2)
                    
                    c = int(cls)  # integer class
                    if c == 2:
                        label = 'Incorrect'
                    else:
                        emb_face = emb_mask if c == 1 else emb_not_mask
                        if last_layer == 'all_KNN':
                            result, distance = all_KNN(im0[y:y+h,x:x+w], model=resnet, emb_face=emb_face, threshold=0.7)
                            label = result+', '+str(float(distance[0]))[:5]
                        elif last_layer == 'linear':
                            result, distance = liner_InceptionResnetv1(im0[y:y+h,x:x+w], model=resnet, emb_face=emb_face, threshold=0.6)
                            label = result+', '+str(float(distance[0]))[:5]
                        elif last_layer == 'fully':
                            result = classify_FC(im0[y:y+h,x:x+w], resnet=resnet, net=emb_face, class_train=class_train, threshold=35.)
                            label = result
                        
                    annotator.box_label(torch.tensor([x1, y1, x2, y2]), label, color=colors(c, True))
                    
            # Stream results
            im0 = annotator.result()
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'Model_Parameter/YOLOv5/weight_train.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'Images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'yolov5/data/facemask.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'Detect', help='save results to project/name')
    parser.add_argument('--name', default='output', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    # run(
    #     weights=ROOT_p + '/models/parameter/YOLOv5/weight_train_v2.pt',
    #     source= 0, # file/dir/URL/glob, 0 for webcam
    #     data=ROOT_p + '/data/facemask.yaml',
    #     imgsz=(640, 640),  # inference size (height, width)
    #     conf_thres=0.25,  # confidence threshold
    #     iou_thres=0.45,  # NMS IOU threshold
    #     max_det=1000,
    #     save_crop=False,  # save cropped prediction boxes
    #     view_img=False,  # maximum detections per image
    #     project=ROOT_p + '/detect',  # save results to project/name
    #     name='Face_Mask',  # save results to project/name
    #     last_layer='fully',
    #     )

    path1 = ROOT_p + '/models/parameter/YOLOv5/weight_train_v1.pt'
    path2 = ROOT_p + '/models/parameter/YOLOv5/weight_train_v2.pt'

    dict1 = torch.load(path1)
    dict2 = torch.load(path2)
    print(dict1.keys())
    print(dict2.keys())
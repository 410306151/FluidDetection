from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
import os
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
import torch
from torch.backends import cudnn
import time
import json
import numpy as np
import PIL.Image
from labelme import utils

targetJSON = "Negative_high-Split1_non-labeled.json"
targetFolder = "Validate/Negative_high"
subFolder = "/Split1"
groundTruthFolder = "./GroundTruth/Test_negative_0612_A/negative_high-2/"

# 註冊資料集到detectron2
register_coco_instances("image_test", {}, "./" + targetFolder + "/" + targetJSON, "./" + targetFolder + subFolder)

# 測試集
metadata_test = MetadataCatalog.get("image_test")
dataset_dicts_test = DatasetCatalog.get("image_test")


def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def getbbox(points, height, width):
    polygons = points
    mask = polygons_to_mask([height, width], polygons)
    return mask

def annotation(points, label, height, width, classesNames):
    annotation = {}
    annotation["mask"] = getbbox(points, height, width)
    annotation["label_name"] = label[0]
    if label[0] in classesNames:
        annotation["label_id"] = classesNames.index(label[0])
    else:
        annotation["label_id"] = None
    return annotation

# ??Model
def testModel():
    #print("This is test for predictor", file=open("testDet2.txt", "w"))
    #print("This is test for predictor", file=open("mask_output.txt", "w"))
    #print("This is pixel result of images", file=open("PixelResult.txt", "w"))
    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    #cfg.merge_from_file("/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/my.yaml")
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-09-30-NewTraining.pth")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-11-19-AllAndMy800.pth")
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-11-18-TrainMyDraw.pth")
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-12-13-AllAndMy1600(MN).pth")
    cfg.DATASETS.TRAIN = () # ??Train????
    cfg.DATASETS.TEST = ("image_test", )
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2 # 設定Non-Maximum Suppression的IoU門檻
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # 0.4   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 2 classes (腎臟, 肝臟)
    
    #####
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ['p2', 'p3', 'p4', 'p5']
    cfg.MODEL.RESNETS.OUT_FEATURES = ['res2', 'res3', 'res4', 'res5']
    cfg.MODEL.FPN.IN_FEATURES = ['res2', 'res3', 'res4', 'res5']
    cfg.MODEL.RPN.IN_FEATURES = ['p2', 'p3', 'p4', 'p5', 'p6']
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    #####
    #print(cfg, file=open("cfgInfo.txt", "w"))
    #print("This is test", file=open("predictOutput.txt", "w"))
    
    predictor = DefaultPredictor(cfg)
    
    crop_coord = [-1, -1, -1, -1]
    first_image = True
    total_data_len = len(dataset_dicts_test)
    start_time = time.time()
    for i in range(total_data_len):
    # for i in range(0, 3):
    # for i in range(total-1, -1, -1):
        d = dataset_dicts_test[i]
        filename = d["file_name"].split('/')[-1]
        im = cv2.imread(d["file_name"])
        if first_image:
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=metadata_test,
                           scale=1,
                           instance_mode=ColorMode.SEGMENTATION#IMAGE_BW   # remove the colors of unsegmented pixels
                           )
        else:
            crop_im = im[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]]
            outputs = predictor(crop_im)
            v = Visualizer(crop_im[:, :, ::-1],
                           metadata=metadata_test,
                           scale=1,
                           instance_mode=ColorMode.SEGMENTATION#IMAGE_BW   # remove the colors of unsegmented pixels
                           )
        result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #print("result === ", file=open("predictOutput.txt", "a"))
        #print(result, file=open("predictOutput.txt", "a"))
        if not first_image:
            #cv2.imwrite("crop_" + filename, crop_im)
            #cv2.imwrite("crop_pre_" + filename, result.get_image()[:, :, ::-1])
            im[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]] = result.get_image()[:, :, ::-1]
            cv2.imwrite(filename, im)
        else:
            cv2.imwrite(filename, result.get_image()[:, :, ::-1])
        first_image = False 
        
        min_x = min_y = 800
        max_x = max_y = -1
        for j in range(len(outputs["instances"])):
            temp_box = outputs["instances"].pred_boxes.tensor[j]
            min_x = min(min_x, crop_coord[0] + temp_box[0].item())
            min_y = min(min_y, crop_coord[1] + temp_box[1].item())
            max_x = max(max_x, crop_coord[0] + temp_box[2].item())
            max_y = max(max_y, crop_coord[1] + temp_box[3].item())
        # crop_add_width_length = (max_x - min_x) * 0.2
        # crop_add_height_length = (max_y - min_y) * 0.2
        
        #print(crop_add_height_length, file=open("mask_output.txt", "a"))
        crop_coord[0] = int(min_x - 25) #crop_add_width_length)
        crop_coord[1] = int(min_y - 40) #crop_add_height_length)
        crop_coord[2] = int(max_x + 30) #crop_add_width_length)
        crop_coord[3] = int(max_y + 30) #crop_add_height_length)
        
    print("--- %s seconds ---" % (time.time() - start_time))
    
def testCUDA():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count()) 
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    x = torch.Tensor([1.0])
    xx = x.cuda()
    print(xx)
    # 測試cudnn
    print(cudnn.is_acceptable(xx))
    
testModel()
#testCUDA()
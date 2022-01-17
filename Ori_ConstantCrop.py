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

targetJSON = "Test_negative_high-2_non-labeled.json"
targetFolder = "Validate/Test_negative_0612_A"
subFolder = "/negative_high-2"
groundTruthFolder = "./GroundTruth/Test_negative_0612_A/negative_high-1/"

# ??????detectron2
register_coco_instances("image_test", {}, "./" + targetFolder + "/" + targetJSON, "./" + targetFolder + subFolder)

# ???
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
    print("This is test for predictor", file=open("mask_output.txt", "w"))
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
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2 # ??Non-Maximum Suppression?IoU??
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # 0.4   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 2 classes (??, ??)
    predictor = DefaultPredictor(cfg)
    #print(cfg, file=open("cfgInfo.txt", "w"))
    #print("This is test", file=open("predictOutput.txt", "w"))
    
    #crop_coord = [30, 420, 75, 590] # 0522_C, 0522_B
    crop_coord = [75, 30, 590, 420] # 0522_C, 0522_B
    total_data_len = len(dataset_dicts_test)
    start_time = time.time()
    for i in range(total_data_len):
    # for i in range(0, 3):
    # for i in range(total-1, -1, -1):
        d = dataset_dicts_test[i]
        filename = d["file_name"].split('/')[-1]
        im = cv2.imread(d["file_name"])
        crop_im = im[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]]
        outputs = predictor(crop_im)
        v = Visualizer(crop_im[:, :, ::-1],
                       metadata=metadata_test,
                       scale=1,
                       instance_mode=ColorMode.SEGMENTATION#IMAGE_BW   # remove the colors of unsegmented pixels
                       )
        result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2.imwrite("crop_" + filename, im[crop_coord[0]: crop_coord[1], crop_coord[2]: crop_coord[3]])
        #print(outputs["instances"].to("cpu").pred_masks, file=open("mask_output.txt", "w"))
        im[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]] = result.get_image()[:, :, ::-1]
        cv2.imwrite(filename, im)
        
    print("--- %s seconds ---" % (time.time() - start_time))
    
def testCUDA():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count()) 
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    x = torch.Tensor([1.0])
    xx = x.cuda()
    print(xx)
    # ??cudnn
    print(cudnn.is_acceptable(xx))
    
testModel()
#testCUDA()
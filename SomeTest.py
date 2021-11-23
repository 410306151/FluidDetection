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

targetJSON = "Test_negative_high-1_non-labeled.json"
targetFolder = "Validate/Test_negative_0612_A"
subFolder = "/negative_high-1"
groundTruthFolder = "./GroundTruth/"

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
    print("This is test for predictor", file=open("mask_output.txt", "w"))
    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-11-19-AllAndMy800.pth")
    cfg.DATASETS.TRAIN = () # ??Train????
    cfg.DATASETS.TEST = ("image_test", )
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2 # 設定Non-Maximum Suppression的IoU門檻
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 # 0.6   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 2 classes (腎臟, 肝臟)
    predictor = DefaultPredictor(cfg)
    #print(cfg, file=open("cfgInfo.txt", "w"))
    #print("This is test", file=open("predictOutput.txt", "w"))
    
    total = len(dataset_dicts_test)
    start_time = time.time()
    for i in range(total):
    # for i in range(0, 1):
    # for i in range(total-1, -1, -1):
        d = dataset_dicts_test[i]
        filename = d["file_name"].split('/')[-1]
        im = cv2.imread(d["file_name"])

        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata_test,
                       scale=1,
                       instance_mode=ColorMode.SEGMENTATION#IMAGE_BW   # remove the colors of unsegmented pixels
                       )
        #outputs["instances"].pred_masks = None
        #print(type(outputs["instances"].pred_masks), file=open("predictOutput.txt", "a"))
        #print(outputs["instances"].pred_masks, file=open("predictOutput.txt", "a"))
        #print("--pred_instances (shape): ", file=open("testDet2.txt", "a"))
        #print(len(outputs["instances"].pred_masks[0][0]), file=open("testDet2.txt", "a"))
        result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(filename, result.get_image()[:, :, ::-1])
        
        pre, ext = os.path.splitext(filename)
        pre = groundTruthFolder + pre + ".json"
        if os.path.exists(pre):
            GT_fp = open(pre, "r")
            GT_data = json.load(GT_fp)
            GT_height = GT_data["imageHeight"]
            GT_width = GT_data["imageWidth"]
            GroundTruth = []
            classesNames = metadata_test.get("thing_classes", None)
            # Make ground truth object
            PixelResult = {}
            for GT_shapes in GT_data["shapes"]:
                GT_points = GT_shapes["points"]
                GT_label = GT_shapes["label"].split("_")
                GT_output = annotation(GT_points, GT_label, GT_height, GT_width, classesNames)
                PixelResult[GT_label + "_Accuracy"] = 0
                PixelResult[GT_label + "_Counter"] = 0
                GroundTruth.append(GT_output)
            
            for i in range(len(outputs["instances"])):
                for j in range(len(GroundTruth)):
                    if GroundTruth[j]["label_id"] == outputs["instances"][i].pred_classes:
                        MaskInNumpy = outputs["instances"].to("cpu").pred_masks[i].detach().numpy()
                        GTandResult = np.logical_and(GroundTruth[j]["mask"], MaskInNumpy)
                        GroundTruthPixel = np.count_nonzero(GroundTruth[j]["mask"]
                        
                        PixelResult[GroundTruth[j]["label_id"] + "_Accuracy"] += (GTandResult / GroundTruthPixel)
                        PixelResult[GroundTruth[j]["label_id"] + "_Counter"] += 1
                        break
            print(PixelResult)
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
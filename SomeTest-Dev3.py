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
    print("This is pixel result of images", file=open("PixelResult.txt", "w"))
    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    #cfg.merge_from_file("/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/my.yaml")
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-09-30-NewTraining.pth")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-11-19-AllAndMy800.pth")
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-11-18-TrainMyDraw.pth")
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-12-12-AllAndMy800(MN).pth")
    cfg.DATASETS.TRAIN = () # ??Train????
    cfg.DATASETS.TEST = ("image_test", )
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2 # 設定Non-Maximum Suppression的IoU門檻
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 # 0.6   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 2 classes (腎臟, 肝臟)
    predictor = DefaultPredictor(cfg)
    #print(cfg, file=open("cfgInfo.txt", "w"))
    #print("This is test", file=open("predictOutput.txt", "w"))
    
    total_data_len = len(dataset_dicts_test)
    start_time = time.time()
    PixelResult = {} 
    PixelResult["liver_Accuracy"] = 0
    PixelResult["kidney_Accuracy"] = 0
    PixelResult["liver_IoU"] = 0
    PixelResult["kidney_IoU"] = 0
    PixelResult["liver_DiceScore"] = 0
    PixelResult["kidney_DiceScore"] = 0
    PixelResult["liver_Counter"] = 0
    PixelResult["kidney_Counter"] = 0
    crop_coord = [-1, -1, -1, -1]
    const_crop_coord = [75, 30, 590, 420]
    first_image = True
    for i in range(total_data_len):
    # for i in range(0, 3):
    # for i in range(total-1, -1, -1):
        d = dataset_dicts_test[i]
        filename = d["file_name"].split('/')[-1]
        im = cv2.imread(d["file_name"])
        if first_image:
            #crop_im = im[const_crop_coord[1]:const_crop_coord[3], const_crop_coord[0]:const_crop_coord[2]]
            #outputs = predictor(crop_im, const_crop_coord)
            outputs = predictor(im, crop_coord)
            v = Visualizer(im[:, :, ::-1],
                           metadata=metadata_test,
                           scale=1,
                           instance_mode=ColorMode.SEGMENTATION#IMAGE_BW   # remove the colors of unsegmented pixels
                           )
            first_image = False
        else:
            #crop_im = im[const_crop_coord[1]:const_crop_coord[3], const_crop_coord[0]:const_crop_coord[2]]
            #outputs = predictor(crop_im, const_crop_coord)
            crop_im = im[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]]
            #cv2.imwrite("crop_" + filename, crop_im)
            outputs = predictor(crop_im, crop_coord)
            #outputs = predictor(crop_im, [0, 0, crop_coord[2] - crop_coord[0], crop_coord[3] - crop_coord[1]])
            v = Visualizer(im[:, :, ::-1],
                           metadata=metadata_test,
                           scale=1,
                           instance_mode=ColorMode.SEGMENTATION#IMAGE_BW   # remove the colors of unsegmented pixels
                           )
        min_x = min_y = 800
        max_x = max_y = -1
        for j in range(len(outputs["instances"].pred_boxes.tensor)):
            min_x = min(min_x, outputs["instances"].pred_boxes.tensor[j][0].item())
            min_y = min(min_y, outputs["instances"].pred_boxes.tensor[j][1].item())
            max_x = max(max_x, outputs["instances"].pred_boxes.tensor[j][2].item())
            max_y = max(max_y, outputs["instances"].pred_boxes.tensor[j][3].item())
        crop_coord[0] = int(min_x - 25) #crop_add_width_length)
        crop_coord[1] = int(min_y - 40) #crop_add_height_length)
        crop_coord[2] = int(max_x + 30) #crop_add_width_length)
        crop_coord[3] = int(max_y + 30) #crop_add_height_length)
        result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(filename, result.get_image()[:, :, ::-1])
        
        pre, ext = os.path.splitext(filename)
        pre = groundTruthFolder + pre + ".json"
        if os.path.exists(pre):
            #print("File: " + filename, file=open("PixelResult.txt", "a"))
            GT_fp = open(pre, "r")
            GT_data = json.load(GT_fp)
            GT_height = GT_data["imageHeight"]
            GT_width = GT_data["imageWidth"]
            GroundTruth = []
            classesNames = metadata_test.get("thing_classes", None)
            # Make ground truth object
            for GT_shapes in GT_data["shapes"]:
                GT_points = GT_shapes["points"]
                GT_label = GT_shapes["label"].split("_")
                GT_output = annotation(GT_points, GT_label, GT_height, GT_width, classesNames)
                GroundTruth.append(GT_output)
            
            for k in range(len(outputs["instances"])):
                for j in range(len(GroundTruth)):
                    if GroundTruth[j]["label_id"] == outputs["instances"][k].pred_classes:
                        MaskInNumpy = outputs["instances"].to("cpu").pred_masks[k].detach().numpy()
                        PredictPixel = np.count_nonzero(MaskInNumpy)
                        GTandResult = np.count_nonzero(np.logical_and(GroundTruth[j]["mask"], MaskInNumpy))
                        GTorResult = np.count_nonzero(np.logical_or(GroundTruth[j]["mask"], MaskInNumpy))
                        GroundTruthPixel = np.count_nonzero(GroundTruth[j]["mask"])
                        #print("Index = " + str(i), file=open("PixelResult.txt", "a"))
                        #print(GroundTruth[j]["label_name"] + "_Accuracy: " + str(round((GTandResult / GroundTruthPixel), 4)), file=open("PixelResult.txt", "a"))
                        #print(GroundTruth[j]["label_name"] + "_IoU: " + str(round((GTandResult / GTorResult), 4)), file=open("PixelResult.txt", "a"))
                        #print(GroundTruth[j]["label_name"] + "_DiceScore: " + str(round((2 * GTandResult) / (GroundTruthPixel + PredictPixel), 4)), file=open("PixelResult.txt", "a"))
                        PixelResult[GroundTruth[j]["label_name"] + "_Accuracy"] += (GTandResult / GroundTruthPixel)
                        PixelResult[GroundTruth[j]["label_name"] + "_IoU"] += (GTandResult / GTorResult)
                        PixelResult[GroundTruth[j]["label_name"] + "_DiceScore"] += ((2 * GTandResult) / (GroundTruthPixel + PredictPixel))
                        PixelResult[GroundTruth[j]["label_name"] + "_Counter"] += 1
                        break
    print("Average liver_Accuracy: " + str(round(PixelResult["liver_Accuracy"] / PixelResult["liver_Counter"], 4)), file=open("PixelResult.txt", "a"))
    print("Average kidney_Accuracy: " + str(round(PixelResult["kidney_Accuracy"] / PixelResult["kidney_Counter"], 4)), file=open("PixelResult.txt", "a"))
    print("Average liver_IoU: " + str(round(PixelResult["liver_IoU"] / PixelResult["liver_Counter"], 4)), file=open("PixelResult.txt", "a"))
    print("Average kidney_IoU: " + str(round(PixelResult["kidney_IoU"] / PixelResult["kidney_Counter"], 4)), file=open("PixelResult.txt", "a"))
    print("Average liver_DiceScore: " + str(round(PixelResult["liver_DiceScore"] / PixelResult["liver_Counter"], 4)), file=open("PixelResult.txt", "a"))
    print("Average kidney_DiceScore: " + str(round(PixelResult["kidney_DiceScore"] / PixelResult["kidney_Counter"], 4)), file=open("PixelResult.txt", "a"))
    
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
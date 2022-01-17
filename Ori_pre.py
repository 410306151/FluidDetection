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
import csv

targetJSON = "Test_negative_high-1_non-labeled.json"
targetFolder = "Validate/Test_negative_0612_A"
subFolder = "/negative_high-1"
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
    
def build_pixel_result():
    PixelResult = {}
    PixelResult["liver_Accuracy"] = 0
    PixelResult["kidney_Accuracy"] = 0
    PixelResult["liver_IoU"] = 0
    PixelResult["kidney_IoU"] = 0
    PixelResult["liver_DiceScore"] = 0
    PixelResult["kidney_DiceScore"] = 0
    PixelResult["liver_Counter"] = 0
    PixelResult["kidney_Counter"] = 0
    PixelResult["min_liver_Accuracy"] = 1
    PixelResult["min_kidney_Accuracy"] = 1
    PixelResult["min_liver_IoU"] = 1
    PixelResult["min_kidney_IoU"] = 1
    PixelResult["min_liver_DiceScore"] = 1
    PixelResult["min_kidney_DiceScore"] = 1
    PixelResult["max_liver_Accuracy"] = 0
    PixelResult["max_kidney_Accuracy"] = 0
    PixelResult["max_liver_IoU"] = 0
    PixelResult["max_kidney_IoU"] = 0
    PixelResult["max_liver_DiceScore"] = 0
    PixelResult["max_kidney_DiceScore"] = 0
    
    return PixelResult

def print_image_index(labelName, accuracy, iou, diceScore):
    print(labelName + "_Accuracy: " + str(accuracy), file=open("ImageResult.txt", "a"))
    print(labelName + "_IoU: " + str(iou), file=open("ImageResult.txt", "a"))
    print(labelName + "_DiceScore: " + str(diceScore), file=open("ImageResult.txt", "a"))
    
def find_total_index(PixelResult, labelName, accuracy, iou, diceScore):
    if PixelResult["max_" + labelName + "_Accuracy"] < accuracy:
        PixelResult["max_" + labelName + "_Accuracy"] = accuracy
    if PixelResult["min_" + labelName + "_Accuracy"] > accuracy:
        PixelResult["min_" + labelName + "_Accuracy"] = accuracy
    if PixelResult["max_" + labelName + "_IoU"] < iou:
        PixelResult["max_" + labelName + "_IoU"] = iou
    if PixelResult["min_" + labelName + "_IoU"] > iou:
        PixelResult["min_" + labelName + "_IoU"] = iou
    if PixelResult["max_" + labelName + "_DiceScore"] < diceScore:
        PixelResult["max_" + labelName + "_DiceScore"] = diceScore
    if PixelResult["min_" + labelName + "_DiceScore"] > diceScore:
        PixelResult["min_" + labelName + "_DiceScore"] = diceScore
    PixelResult[labelName + "_Accuracy"] += accuracy
    PixelResult[labelName + "_IoU"] += iou
    PixelResult[labelName + "_DiceScore"] += diceScore
    PixelResult[labelName + "_Counter"] += 1
    
def print_total_index(PixelResult, start_time):
    print("--- %s seconds ---" % (time.time() - start_time), file=open("PixelResult.txt", "a"))
    print("Average liver_Accuracy: " + str(round(PixelResult["liver_Accuracy"] / PixelResult["liver_Counter"], 4)), file=open("PixelResult.txt", "a"))
    print("Average kidney_Accuracy: " + str(round(PixelResult["kidney_Accuracy"] / PixelResult["kidney_Counter"], 4)), file=open("PixelResult.txt", "a"))
    print("Average liver_IoU: " + str(round(PixelResult["liver_IoU"] / PixelResult["liver_Counter"], 4)), file=open("PixelResult.txt", "a"))
    print("Average kidney_IoU: " + str(round(PixelResult["kidney_IoU"] / PixelResult["kidney_Counter"], 4)), file=open("PixelResult.txt", "a"))
    print("Average liver_DiceScore: " + str(round(PixelResult["liver_DiceScore"] / PixelResult["liver_Counter"], 4)), file=open("PixelResult.txt", "a"))
    print("Average kidney_DiceScore: " + str(round(PixelResult["kidney_DiceScore"] / PixelResult["kidney_Counter"], 4)), file=open("PixelResult.txt", "a"))
    print("Max liver_Accuracy: " + str(PixelResult["max_liver_Accuracy"]), file=open("PixelResult.txt", "a"))
    print("Min liver_Accuracy: " + str(PixelResult["min_liver_Accuracy"]), file=open("PixelResult.txt", "a"))
    print("Max kidney_Accuracy: " + str(PixelResult["max_kidney_Accuracy"]), file=open("PixelResult.txt", "a"))
    print("Min kidney_Accuracy: " + str(PixelResult["min_kidney_Accuracy"]), file=open("PixelResult.txt", "a"))
    print("Max liver_IoU: " + str(PixelResult["max_liver_IoU"]), file=open("PixelResult.txt", "a"))
    print("Min liver_IoU: " + str(PixelResult["min_liver_IoU"]), file=open("PixelResult.txt", "a"))
    print("Max kidney_IoU: " + str(PixelResult["max_kidney_IoU"]), file=open("PixelResult.txt", "a"))
    print("Min kidney_IoU: " + str(PixelResult["min_kidney_IoU"]), file=open("PixelResult.txt", "a"))
    print("Max liver_DiceScore: " + str(PixelResult["max_liver_DiceScore"]), file=open("PixelResult.txt", "a"))
    print("Min liver_DiceScore: " + str(PixelResult["min_liver_DiceScore"]), file=open("PixelResult.txt", "a"))
    print("Max kidney_DiceScore: " + str(PixelResult["max_kidney_DiceScore"]), file=open("PixelResult.txt", "a"))
    print("Min kidney_DiceScore: " + str(PixelResult["min_kidney_DiceScore"]), file=open("PixelResult.txt", "a"))

# ??Model
def testModel():
    #print("This is test for predictor", file=open("testDet2.txt", "w"))
    #print("This is test for predictor", file=open("mask_output.txt", "w"))
    print("This is pixel result of images", file=open("PixelResult.txt", "w"))
    print("This is pixel result of images", file=open("ImageResult.txt", "w"))
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4 # 0.7 # 0.4   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 2 classes (??, ??)
    predictor = DefaultPredictor(cfg)
    #print(cfg, file=open("cfgInfo.txt", "w"))
    #print("This is test", file=open("predictOutput.txt", "w"))
    
    index_overall = {}
    index_overall["min_liver_index"] = 1
    index_overall["min_kidney_index"] = 1
    index_overall["max_liver_index"] = 0
    index_overall["max_kidney_index"] = 0
    PixelResult = build_pixel_result() 
    total_data_len = len(dataset_dicts_test)
    start_time = time.time()
    for i in range(total_data_len):
    # for i in range(0, 96):
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
        result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2.imwrite(filename, result.get_image()[:, :, ::-1])
        
        pre, ext = os.path.splitext(filename)
        pre = groundTruthFolder + pre + ".json"
        if os.path.exists(pre):
            #print("File: " + filename, file=open("ImageResult.txt", "a"))
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
                        accuracy = round((GTandResult / GroundTruthPixel), 4)
                        iou = round((GTandResult / GTorResult), 4)
                        diceScore = round((2 * GTandResult) / (GroundTruthPixel + PredictPixel), 4)
                        
                        #print_image_index(GroundTruth[j]["label_name"], accuracy, iou, diceScore)
                        find_total_index(PixelResult, GroundTruth[j]["label_name"], accuracy, iou, diceScore)
                        '''if i != 0:
                            object_score = outputs["instances"][k].scores
                            if index_overall["min_" + GroundTruth[j]["label_name"] + "_index"] > object_score:
                                index_overall["min_" + GroundTruth[j]["label_name"] + "_index"] = object_score
                            if index_overall["max_" + GroundTruth[j]["label_name"] + "_index"] < object_score:
                                index_overall["max_" + GroundTruth[j]["label_name"] + "_index"] = object_score'''
                        break
    print_total_index(PixelResult, start_time)
    '''print("min_liver_index: " + str(index_overall["min_liver_index"]), file=open("PixelResult.txt", "a"))
    print("min_kidney_index: " + str(index_overall["min_kidney_index"]), file=open("PixelResult.txt", "a"))
    print("max_liver_index: " + str(index_overall["max_liver_index"]), file=open("PixelResult.txt", "a"))
    print("max_kidney_index: " + str(index_overall["max_kidney_index"]), file=open("PixelResult.txt", "a"))'''
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
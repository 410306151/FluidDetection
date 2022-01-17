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
import json
import numpy as np
import PIL.Image
from labelme import utils

targetJSON = "Test_negative_high-2_non-labeled.json"
targetFolder = "Validate/Test_negative_0612_A"
subFolder = "/negative_high-2"
groundTruthFolder = "./GroundTruth/Test_negative_0612_A/negative_high-2/"

# ??????detectron2
register_coco_instances("image_test", {}, "./" + targetFolder + "/" + targetJSON, "./" + targetFolder + subFolder)
register_coco_instances("image_test2", {}, "./" + targetFolder + "/Test_negative_high-1_non-labeled.json", "./" + targetFolder + "/negative_high-1")
register_coco_instances("image_test3", {}, "./Validate/Test_negative_0522_B/Test_negative_0522_B-negative_2-1_non-labeled.json", "./Validate/Test_negative_0522_B/negative_2-1")
register_coco_instances("image_test4", {}, "./Validate/Test_negative_0522_C/Test_negative_0522_C-negative_2-2_non-labeled.json", "./Validate/Test_negative_0522_C/negative_2-2")
register_coco_instances("image_test5", {}, "./Validate/Negative_high/Negative_high-Split2_non-labeled.json", "./Validate/Negative_high/Split2")

# ???
metadata_test = MetadataCatalog.get("image_test")
dataset_dicts_test = DatasetCatalog.get("image_test")
dataset_dicts_test2 = DatasetCatalog.get("image_test2")
dataset_dicts_test3 = DatasetCatalog.get("image_test3")
dataset_dicts_test4 = DatasetCatalog.get("image_test4")
dataset_dicts_test5 = DatasetCatalog.get("image_test5")


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
def print_parameter(candidate_xy):
    print("Parameter left = " + str(candidate_xy[0]) + ", right = " + str(candidate_xy[2]) + ", top = " + str(candidate_xy[1]) + ", bottom = " + str(candidate_xy[3]) + ", Total = " + str(candidate_xy[0] + candidate_xy[1] + candidate_xy[2] + candidate_xy[3]), file=open("ParameterResult.txt", "a"))

def test_other_data(predictor, dataset_dicts, dataset_name, candidate_xy):
    total_data_len = len(dataset_dicts)
    crop_coord = [-1, -1, -1, -1]
    first_image = True
    
    for i in range(total_data_len):
        d = dataset_dicts[i]
        filename = d["file_name"].split('/')[-1]
        im = cv2.imread(d["file_name"])
        if first_image:
            outputs = predictor(im, crop_coord)
            first_image = False
        else:
            crop_im = im[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]]
            outputs = predictor(crop_im, crop_coord)
        if len(outputs["instances"]) < 2:
            #print_parameter(candidate_xy)
            print("Dataset: { " + str(dataset_name) + " } doesn't have 2 organ outputs. File: "+ str(filename), file=open("ParameterResult.txt", "a"))
            return True
        min_x = min_y = 800
        max_x = max_y = -1
        for j in range(len(outputs["instances"].pred_boxes.tensor)):
            min_x = min(min_x, outputs["instances"].pred_boxes.tensor[j][0].item())
            min_y = min(min_y, outputs["instances"].pred_boxes.tensor[j][1].item())
            max_x = max(max_x, outputs["instances"].pred_boxes.tensor[j][2].item())
            max_y = max(max_y, outputs["instances"].pred_boxes.tensor[j][3].item())
        crop_coord[0] = int(min_x - candidate_xy[0])
        crop_coord[1] = int(min_y - candidate_xy[1])
        crop_coord[2] = int(max_x + candidate_xy[2])
        crop_coord[3] = int(max_y + candidate_xy[3])
        
    return False

# ??Model
def testModel():
    print("This is parameter result", file=open("ParameterResult.txt", "w"))
    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-11-19-AllAndMy800.pth")
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-11-18-TrainMyDraw.pth")
    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-12-12-AllAndMy800(MN).pth")
    cfg.DATASETS.TRAIN = () # ??Train????
    #cfg.DATASETS.TEST = ("image_test", )
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2 # ??Non-Maximum Suppression?IoU??
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 2 classes (??, ??)
    predictor = DefaultPredictor(cfg)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-11-18-TrainMyDraw.pth")
    predictor2 = DefaultPredictor(cfg)
    
    total_data_len = len(dataset_dicts_test)
    crop_coord = [-1, -1, -1, -1]
    candidate_x = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    candidate_y = [10, 15, 20, 25, 30, 35, 40]
    first_image = True
    PixelResult = {}
    for n in range(len(candidate_x)):
        for k in range(len(candidate_x)):
            for l in range(len(candidate_y)):
                for m in range(len(candidate_y)):
                    PixelResult["liver_Accuracy"] = 0
                    PixelResult["kidney_Accuracy"] = 0
                    PixelResult["liver_IoU"] = 0
                    PixelResult["kidney_IoU"] = 0
                    PixelResult["liver_DiceScore"] = 0
                    PixelResult["kidney_DiceScore"] = 0
                    PixelResult["liver_Counter"] = 0
                    PixelResult["kidney_Counter"] = 0
                    crop_coord = [-1, -1, -1, -1]
                    first_image = True
                    print_parameter([candidate_x[n], candidate_y[l], candidate_x[k], candidate_y[m]])
                    for i in range(total_data_len):
                        d = dataset_dicts_test[i]
                        filename = d["file_name"].split('/')[-1]
                        im = cv2.imread(d["file_name"])
                        if first_image:
                            outputs = predictor(im, crop_coord)
                            first_image = False
                        else:
                            crop_im = im[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]]
                            outputs = predictor(crop_im, crop_coord)
                        if len(outputs["instances"]) < 2:
                            #print(str(filename) + ". DON'T have 2 organ outputs", file=open("ParameterResult.txt", "a"))
                            break
                        min_x = min_y = 800
                        max_x = max_y = -1
                        for j in range(len(outputs["instances"].pred_boxes.tensor)):
                            min_x = min(min_x, outputs["instances"].pred_boxes.tensor[j][0].item())
                            min_y = min(min_y, outputs["instances"].pred_boxes.tensor[j][1].item())
                            max_x = max(max_x, outputs["instances"].pred_boxes.tensor[j][2].item())
                            max_y = max(max_y, outputs["instances"].pred_boxes.tensor[j][3].item())
                        crop_coord[0] = max(int(min_x - candidate_x[n]), 0) #crop_add_width_length)
                        crop_coord[1] = max(int(min_y - candidate_y[l]), 0) #crop_add_height_length)
                        crop_coord[2] = min(int(max_x + candidate_x[k]), 640) #crop_add_width_length)
                        crop_coord[3] = min(int(max_y + candidate_y[m]), 480) #crop_add_height_length)
                        
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
                            for GT_shapes in GT_data["shapes"]:
                                GT_points = GT_shapes["points"]
                                GT_label = GT_shapes["label"].split("_")
                                GT_output = annotation(GT_points, GT_label, GT_height, GT_width, classesNames)
                                GroundTruth.append(GT_output)
                            
                            for z in range(len(outputs["instances"])):
                                for y in range(len(GroundTruth)):
                                    if GroundTruth[y]["label_id"] == outputs["instances"][z].pred_classes:
                                        MaskInNumpy = outputs["instances"].to("cpu").pred_masks[z].detach().numpy()
                                        PredictPixel = np.count_nonzero(MaskInNumpy)
                                        GTandResult = np.count_nonzero(np.logical_and(GroundTruth[y]["mask"], MaskInNumpy))
                                        GTorResult = np.count_nonzero(np.logical_or(GroundTruth[y]["mask"], MaskInNumpy))
                                        GroundTruthPixel = np.count_nonzero(GroundTruth[y]["mask"])
                                        PixelResult[GroundTruth[y]["label_name"] + "_Accuracy"] += (GTandResult / GroundTruthPixel)
                                        PixelResult[GroundTruth[y]["label_name"] + "_IoU"] += (GTandResult / GTorResult)
                                        PixelResult[GroundTruth[y]["label_name"] + "_DiceScore"] += ((2 * GTandResult) / (GroundTruthPixel + PredictPixel))
                                        PixelResult[GroundTruth[y]["label_name"] + "_Counter"] += 1
                                        break
                                 
                    if ((PixelResult["liver_Accuracy"] / PixelResult["liver_Counter"]) < 0.75) or ((PixelResult["kidney_Accuracy"] / PixelResult["kidney_Counter"]) < 0.85):
                        #print("Does NOT have enough average accuracy", file=open("ParameterResult.txt", "a"))
                        continue
                    else:
                        if test_other_data(predictor, dataset_dicts_test2, "image_test2", [candidate_x[n], candidate_y[l], candidate_x[k], candidate_y[m]]):
                            continue
                        if test_other_data(predictor2, dataset_dicts_test3, "image_test3", [candidate_x[n], candidate_y[l], candidate_x[k], candidate_y[m]]):
                            continue
                        if test_other_data(predictor2, dataset_dicts_test4, "image_test4", [candidate_x[n], candidate_y[l], candidate_x[k], candidate_y[m]]):
                            continue
                        if test_other_data(predictor, dataset_dicts_test5, "image_test5", [candidate_x[n], candidate_y[l], candidate_x[k], candidate_y[m]]):
                            continue
                        
                        #print_parameter([candidate_x[n], candidate_y[l], candidate_x[k], candidate_y[m]])
                        print("Average liver_Accuracy: " + str(round(PixelResult["liver_Accuracy"] / PixelResult["liver_Counter"], 4)), file=open("ParameterResult.txt", "a"))
                        print("Average kidney_Accuracy: " + str(round(PixelResult["kidney_Accuracy"] / PixelResult["kidney_Counter"], 4)), file=open("ParameterResult.txt", "a"))
                        print("Average liver_IoU: " + str(round(PixelResult["liver_IoU"] / PixelResult["liver_Counter"], 4)), file=open("ParameterResult.txt", "a"))
                        print("Average kidney_IoU: " + str(round(PixelResult["kidney_IoU"] / PixelResult["kidney_Counter"], 4)), file=open("ParameterResult.txt", "a"))
                        print("Average liver_DiceScore: " + str(round(PixelResult["liver_DiceScore"] / PixelResult["liver_Counter"], 4)), file=open("ParameterResult.txt", "a"))
                        print("Average kidney_DiceScore: " + str(round(PixelResult["kidney_DiceScore"] / PixelResult["kidney_Counter"], 4)), file=open("ParameterResult.txt", "a"))
                        print("liver_Counter: " + str(PixelResult["liver_Counter"]), file=open("ParameterResult.txt", "a"))
                        print("kidney_Counter: " + str(PixelResult["kidney_Counter"]), file=open("ParameterResult.txt", "a"))
    
testModel()
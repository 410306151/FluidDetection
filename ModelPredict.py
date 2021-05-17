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
from datetime import date
import json
import numpy as np
import pycocotools.mask as mask_util
from imantics import Polygons, Mask
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
import base64
from labelme import utils
import PIL
import io
import codecs

targetJSON = "positive_3_non-labeled.json"
targetFolder = "positive_3"
subFolder = "/Non-labeled"

# 註冊資料集到detectron2
register_coco_instances("image_test", {}, "./" + targetFolder + "/" + targetJSON, "./" + targetFolder + subFolder)

# 測試集
fluid_metadata_test = MetadataCatalog.get("image_test")
dataset_dicts_test = DatasetCatalog.get("image_test")

def countsToPolygons(counts, size):
    result = []
    
    segment = {'counts': counts, 'size': size}
    # 對binary mask解碼
    mask = mask_util.decode(segment)[:, :]
    array = mask
    polygons = Mask(array).polygons()
    # JSON不能序列化ndarray，所以要把每個element轉成list再放進陣列
    if len(polygons.points) > 0:
        for i in range(len(polygons.points[0])):
            result.append(polygons.points[0][i].tolist())
    else:
        return None

    return result

def getLabelmeFormat(filename, maskData, metadata, datasetDict, directory):
    labelmeData = {}
    shapes = []
    
    # 一些labelme預設欄位
    labelmeData['version'] = "4.5.7"
    labelmeData['flags'] = {}
    
    # 從註冊的資料集取出Class name
    classesNames = metadata.get("thing_classes", None)
    # 從預測結果取得Class編號
    boxes = maskData.pred_boxes if maskData.has("pred_boxes") else None
    classes = maskData.pred_classes if maskData.has("pred_classes") else None
    scores = maskData.scores if maskData.has("scores") else None
    
    coco = instances_to_coco_json(maskData, 0)
    for i in range(len(coco)):
        shape = {}
        shape['label'] = classesNames[classes[i]]
        points = countsToPolygons(coco[i]['segmentation']['counts'], coco[i]['segmentation']['size'])
        if points is not None:
            shape['points'] = points
        else:
            continue
        # 以下欄位寫死
        shape['group_id'] = None
        shape['shape_type'] = "polygon"
        shape['flags'] = {}
        shape['confidence_score'] = scores[i].tolist() # 用來觀察NMS會刪除哪個重複的label
        
        shapes.append(shape)
        
    labelmeData['shapes'] = shapes
    labelmeData['imagePath'] = filename
    
    im = cv2.imread(datasetDict["file_name"])
    encoded = utils.img_arr_to_b64(im).decode('utf-8')
    img_pil = PIL.Image.fromarray(im, mode='RGB')
    f = io.BytesIO()
    img_pil.save(f, format='JPEG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    
    labelmeData['imageData'] = encData
    labelmeData['imageHeight'] = datasetDict['height']
    labelmeData['imageWidth'] = datasetDict['width']
    
    base = os.path.splitext(filename)[0]
    json.dump(labelmeData, open(directory + base + ".json", "w"), indent=4)

# 看訓練圖的邊框
def showImage():
    for i in range(4, 10):
        d = dataset_dicts_n[i]
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=fluid_metadata_n, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(d["file_name"], vis.get_image()[:, :, ::-1])
        cv2.waitKey(0)

# 驗證模型
def validateModel(testData, datasetDict, metadata, directory):
    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-04-22-Manual.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST  = 0.1 # 設定Non-Maximum Suppression的IoU門檻
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (積水, 腎臟, 肝臟)
    cfg.DATASETS.TEST = (testData, )
    predictor = DefaultPredictor(cfg)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(len(datasetDict)):
    # for i in range(29, 30):
        d = datasetDict[i]
        filename = d["file_name"].split('/')[-1]
        im = cv2.imread(d["file_name"])

        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=1,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                       )
        result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(directory + filename, result.get_image()[:, :, ::-1])
        getLabelmeFormat(filename, outputs["instances"].to("cpu"), metadata, d, directory)

today = date.today()
# showImage()
validateModel("image_test", dataset_dicts_test, fluid_metadata_test, "./Predict-" + targetFolder + "-" + str(today) + "/")
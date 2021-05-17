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

# 註冊資料集到detectron2
register_coco_instances("alldata", {}, "./AllData/AllData.json", "./AllData/Labeled")
register_coco_instances("image_test", {}, "./positive_3/positive_3_non-labeled.json", "./positive_3/Non-labeled")

# 測試集
fluid_metadata_test = MetadataCatalog.get("image_test")
dataset_dicts_test = DatasetCatalog.get("image_test")

# ??Model
def testModel():
    print("This is test for predictor", file=open("testDet2.txt", "w"))
    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-04-22-Manual.pth")
    cfg.DATASETS.TRAIN = ("alldata",) # ??Train????
    cfg.DATASETS.TEST = ("image_test", )
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST  = 0.1123 # 設定Non-Maximum Suppression的IoU門檻
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (積水, 腎臟, 肝臟)
    # trainer = DefaultTrainer(cfg)
    predictor = DefaultPredictor(cfg)
    print(cfg, file=open("cfgInfo.txt", "w"))
    # print(trainer, file=open("modelInfo.txt", "a"))
    print("This is test", file=open("predictOutput.txt", "w"))
    
    for i in range(941, 943):
        d = dataset_dicts_test[i]
        filename = d["file_name"].split('/')[-1]
        im = cv2.imread(d["file_name"])

        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=fluid_metadata_test,
                       scale=1,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                       )
        print(outputs, file=open("predictOutput.txt", "a"))
        print(type(outputs), file=open("predictOutput.txt", "a"))
        print("--pred_instances (shape): ", file=open("testDet2.txt", "a"))
        print(len(outputs["instances"].pred_masks[0][0]), file=open("testDet2.txt", "a"))
        result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(filename, result.get_image()[:, :, ::-1])
    
    
testModel()
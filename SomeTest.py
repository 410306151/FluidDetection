from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import random
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
import os
from detectron2.utils.visualizer import ColorMode

# 註冊資料集到detectron2
register_coco_instances("alldata", {}, "./AllData/AllData.json", "./AllData/Labeled")
register_coco_instances("image_test", {}, "./negative_4/negative_4_non-labeled.json", "./negative_4/Non-labeled")

# 測試集
fluid_metadata_test = MetadataCatalog.get("image_test")
dataset_dicts_test = DatasetCatalog.get("image_test")

# ??Model
def testModel():
    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = ("alldata",) # ??Train????
    cfg.DATASETS.TEST = ("image_test", )
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST  = 0.1 # 設定Non-Maximum Suppression的IoU門檻
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set the testing threshold for this model
    trainer = DefaultTrainer(cfg)
    predictor = DefaultPredictor(cfg)
    # print(trainer, file=open("modelInfo.txt", "a"))
    
    for i in range(549, 550):
        d = dataset_dicts_test[i]
        filename = d["file_name"].split('/')[-1]
        im = cv2.imread(d["file_name"])

        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=fluid_metadata_test,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                       )
        print(outputs, file=open("predictOutput.txt", "a"))
        # result = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imwrite(directory + filename, result.get_image()[:, :, ::-1])
    
    
testModel()
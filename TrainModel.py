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
register_coco_instances("positive", {}, "./Positive_high/positive_high.json", "./Positive_high/Labeled")
register_coco_instances("negative", {}, "./Negative_high/negative_high.json", "./Negative_high/Labeled")
register_coco_instances("alldata", {}, "./AllData/AllData.json", "./AllData/Labeled")

dataset_dicts_test = DatasetCatalog.get("alldata")
IMS_PER_BATCH = 2

# 訓練Model
def trainModel():

    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ) # 從Detectron2拿Mask RCNN的模型
    cfg.DATASETS.TRAIN = ("alldata",) # 拿來Train的資料集
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2 # 2個workers取圖片
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # 從網路上拿Pretrained的權重
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # 讀取已訓練過的權重
    cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH # 每個 iteration 看兩張圖片
    cfg.SOLVER.BASE_LR = 0.02 # learning rate
    cfg.SOLVER.MAX_ITER = int(len(dataset_dicts_test) / IMS_PER_BATCH) * 100  # number of iteration: 1 Epoch = MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # 在訓練中，當RPN的Proposals達這個數字後，會進行一次loss計算。
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (積水, 腎臟, 肝臟)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

trainModel()

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
from detectron2.utils.logger import setup_logger
from detectron2_backbone.config import add_backbone_config

# 註冊資料集到detectron2
# register_coco_instances("alldata", {}, "./AllData/AllData(Manual).json", "./AllData/Labeled(Manual)")
register_coco_instances("alldata", {}, "./AllData/AllAndMy.json", "./AllData/AllAndMy")


dataset_dicts_test = DatasetCatalog.get("alldata")
IMS_PER_BATCH = 2

# 訓練Model
def trainModel():

    cfg = get_cfg()
    #cfg.merge_from_file(
    #    "/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    #) # 從Detectron2拿Mask RCNN的模型
    add_backbone_config(cfg)
    cfg.merge_from_file(
        "/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/my.yaml"
    )
    cfg.DATASETS.TRAIN = ("alldata",) # 拿來Train的資料集
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2 # 2個workers取圖片
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # 從網路上拿Pretrained的權重
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-09-30-NewTraining.pth") # 讀取已訓練過的權重
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-12-12-AllAndMy800(MN).pth") # 讀取已訓練過的權重
    cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH # 每個 iteration 看兩張圖片
    cfg.SOLVER.BASE_LR = 0.002 # learning rate
    cfg.SOLVER.MAX_ITER = int(len(dataset_dicts_test) / IMS_PER_BATCH) * 800 # 2000  # number of iteration: 1 Epoch = MAX_ITER * BATCH_SIZE / TOTAL_NUM_IMAGES
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # 在訓練中，當RPN的Proposals達這個數字後，會進行一次loss計算。
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 2 classes (腎臟, 肝臟)


    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

trainModel()

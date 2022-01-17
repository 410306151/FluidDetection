from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2
from detectron2.utils.logger import setup_logger
#setup_logger()
# import some common libraries
import numpy as np
import tqdm
import cv2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import time
import os

targetJSON = "Test_negative_high-2_non-labeled.json"
targetFolder = "Validate/Test_negative_0612_A"
subFolder = "/negative_high-2"
# ??????detectron2
register_coco_instances("image_test", {}, "./" + targetFolder + "/" + targetJSON, "./" + targetFolder + subFolder)

# ???
metadata_test = MetadataCatalog.get("image_test")
dataset_dicts_test = DatasetCatalog.get("image_test")

# Extract video properties
video = cv2.VideoCapture('Validate/Test_negative_0612_A/0612A_negative_high-2.mp4')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize video writer
video_writer = cv2.VideoWriter('HaveCrop.mp4', fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

def setup_predictor():
    # Initialize predictor
    cfg = get_cfg()
    cfg.merge_from_file(
        "/home/ad/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2 # ??Non-Maximum Suppression?IoU??
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # 0.4   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # 2 classes (??, ??)
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ("image_test", )
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final-2021-11-19-AllAndMy800.pth")
    
    predictor = DefaultPredictor(cfg)
    
    return predictor

def runOnVideo_EveryFrame(video, maxFrames):
    crop_coord = [-1, -1, -1, -1]
    #crop_coord = [75, 30, 590, 420] # 0522_C, 0522_B
    first_predict = True
    later_predict = False
    readFrames = 0
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break
        if first_predict:
            # Get prediction results for this frame
            outputs = predictor(frame)
            # Draw a visualization of the predictions using the video visualizer
            visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
        else:
            min_x = min_y = 800
            max_x = max_y = -1
            for j in range(len(outputs["instances"])):
                temp_box = outputs["instances"].pred_boxes.tensor[j]
                min_x = min(min_x, crop_coord[0] + temp_box[0].item())
                min_y = min(min_y, crop_coord[1] + temp_box[1].item())
                max_x = max(max_x, crop_coord[0] + temp_box[2].item())
                max_y = max(max_y, crop_coord[1] + temp_box[3].item())
            crop_coord[0] = int(min_x - 25)
            crop_coord[1] = int(min_y - 40)
            crop_coord[2] = int(max_x + 30)
            crop_coord[3] = int(max_y + 30)
            crop_im = frame[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]]
            outputs = predictor(crop_im)
            # Draw a visualization of the predictions using the video visualizer
            visualization = v.draw_instance_predictions(crop_im, outputs["instances"].to("cpu"))
        
        if first_predict:
            visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
        else:
            frame[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]] = visualization.get_image()[:, :, ::-1]
            visualization = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            later_predict = True
        first_predict = False 
    
        # Convert Matplotlib RGB format to OpenCV BGR format
        visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    
        yield visualization
            
        readFrames += 1

def runOnVideo_SkipFrame(video, maxFrames):
    """ Runs the predictor on every frame in the video (unless maxFrames is given),
    and returns the frame with the predictions drawn.
    """
    crop_coord = [-1, -1, -1, -1]
    #crop_coord = [75, 30, 590, 420] # 0522_C, 0522_B
    first_predict = True
    later_predict = False
    readFrames = 0
    while True:
        hasFrame, frame = video.read()
        if not hasFrame:
            break
        if readFrames % 15 == 0:
            if first_predict:
                # Get prediction results for this frame
                outputs = predictor(frame)
                # Draw a visualization of the predictions using the video visualizer
                visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
            else:
                min_x = min_y = 800
                max_x = max_y = -1
                for j in range(len(outputs["instances"])):
                    temp_box = outputs["instances"].pred_boxes.tensor[j]
                    min_x = min(min_x, crop_coord[0] + temp_box[0].item())
                    min_y = min(min_y, crop_coord[1] + temp_box[1].item())
                    max_x = max(max_x, crop_coord[0] + temp_box[2].item())
                    max_y = max(max_y, crop_coord[1] + temp_box[3].item())
                crop_coord[0] = int(min_x - 25)
                crop_coord[1] = int(min_y - 40)
                crop_coord[2] = int(max_x + 30)
                crop_coord[3] = int(max_y + 30)
                crop_im = frame[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]]
                outputs = predictor(crop_im)
                # Draw a visualization of the predictions using the video visualizer
                visualization = v.draw_instance_predictions(crop_im, outputs["instances"].to("cpu"))
            
            if first_predict:
                visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
            else:
                frame[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]] = visualization.get_image()[:, :, ::-1]
                visualization = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                later_predict = True
            first_predict = False 
    
            # Convert Matplotlib RGB format to OpenCV BGR format
            visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    
            yield visualization
        else:
            # Make sure the frame is colored
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if later_predict:
                crop_im = frame[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]]
                visualization = v.draw_instance_predictions(crop_im, outputs["instances"].to("cpu"))
                frame[crop_coord[1]:crop_coord[3], crop_coord[0]:crop_coord[2]] = visualization.get_image()[:, :, ::-1]
                visualization = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Convert Matplotlib RGB format to OpenCV BGR format
                visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            else:
                # Draw a visualization of the predictions using the video visualizer
                visualization = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
                # Convert Matplotlib RGB format to OpenCV BGR format
                visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
    
            yield visualization
        readFrames += 1
            
predictor = setup_predictor()
# Initialize visualizer
v = VideoVisualizer(metadata=metadata_test,
                           instance_mode=ColorMode.IMAGE)

start_time = time.time()
# Enumerate the frames of the video
for visualization in runOnVideo_EveryFrame(video, num_frames):
    # Write to video file
    video_writer.write(visualization)
# Release resources
video.release()
video_writer.release()
cv2.destroyAllWindows()
print("--- %s seconds ---" % (time.time() - start_time))
print("num_frames ===" + str(num_frames))
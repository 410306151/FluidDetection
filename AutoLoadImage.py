import os
from PIL import Image
import json

def get_categories():
    # 目前種類只有固定3種，所以寫死
    categories = []
    category = {}
    category["supercategory"] = "fluid"
    category["id"] = 0
    category["name"] = "fluid"
    categories.append(category)
    category = {}
    category["supercategory"] = "kidney"
    category["id"] = 1
    category["name"] = "kidney"
    categories.append(category)
    category = {}
    category["supercategory"] = "liver"
    category["id"] = 2
    category["name"] = "liver"
    categories.append(category)

    return categories

def read_directory(directoryName, outputFile):
    data_coco = {}
    images = []
    image_id = 0
    for imageName in os.listdir(r"./" + directoryName):
        image = {}
        # 自己抓出圖片的size
        im = Image.open(directoryName + "/" + imageName)
        width, height = im.size
        image["id"] = image_id
        image["file_name"] = imageName
        image["width"] = width
        image["height"] = height
        images.append(image)
        image_id += 1

    data_coco["images"] = images
    data_coco["categories"] = get_categories()
    json.dump(data_coco, open(outputFile, "w"), indent=4)


# read_directory("negative_6/", "negative_6/negative_6.json")
read_directory("tempData/negative_4/Non-labeled", "tempData/negative_4/negative_4_non-labeled.json")
read_directory("tempData/negative_6/Non-labeled", "tempData/negative_6/negative_6_non-labeled.json")
read_directory("tempData/negative_high/Non-labeled", "tempData/negative_high/negative_high_non-labeled.json")
read_directory("tempData/negative_low/Non-labeled", "tempData/negative_low/negative_low_non-labeled.json")
read_directory("tempData/positive_3/Non-labeled", "tempData/positive_3/positive_3_non-labeled.json")
read_directory("tempData/positive_5/Non-labeled", "tempData/positive_5/positive_5_non-labeled.json")
read_directory("tempData/positive_high/Non-labeled", "tempData/positive_high/positive_high_non-labeled.json")
read_directory("tempData/positive_high_2/Non-labeled", "tempData/positive_high_2/positive_high_2_non-labeled.json")
read_directory("tempData/positive_low/Non-labeled", "tempData/positive_low/positive_low_non-labeled.json")
read_directory("tempData/positive_low_2/Non-labeled", "tempData/positive_low_2/positive_low_2_non-labeled.json")
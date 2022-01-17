import os
from PIL import Image
import json

def get_categories():
    # 目前種類只有固定3種，所以寫死
    categories = []
    category = {}
    category["supercategory"] = "kidney"
    category["id"] = 0
    category["name"] = "kidney"
    categories.append(category)
    category = {}
    category["supercategory"] = "liver"
    category["id"] = 1
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


read_directory("negative_high/Non-labeled", "negative_high/negative_high_non-labeled.json")
#read_directory("positive_3/Non-labeled", "positive_3/positive_3_non-labeled.json")
#read_directory("Validation/Test_negative_0522_C/negative_2-2", "Validation/Test_negative_0522_C/Test_negative_0522_C-negative_2-2_non-labeled.json")
#read_directory("Validation/Test_negative_0612_B/negative_high-1-Part2", "Validation/Test_negative_0612_B/Test_negative_0612_B-negative_high-1-Part2_non-labeled.json")
#read_directory("Validation/Test_negative_0612_B/negative_high-2", "Validation/Test_negative_0612_B/Test_negative_0612_B-negative_high-2_non-labeled.json")




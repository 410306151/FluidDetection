import os

def read_directory(directoryName):
    # 建立資料夾
    if not os.path.exists(directoryName + "/Labeled"):
        os.makedirs(directoryName + "/Labeled")
    if not os.path.exists(directoryName + "/Non-labeled"):
        os.makedirs(directoryName + "/Non-labeled")
    # 開始分有標註跟沒標註的圖片
    for fileName in os.listdir(r"./" + directoryName):
        # 避免讀到不必要的檔案，因此限制從.jpg找.json
        if fileName.endswith(".jpg") or fileName.endswith(".JPG"):
            base = os.path.splitext(fileName)[0]
            if os.path.isfile(directoryName + "/" + base + ".json"):
                os.replace(directoryName + "/" + base + ".json", directoryName + "/Labeled/" + base + ".json") # 移動Json
                os.replace(directoryName + "/" + fileName, directoryName + "/Labeled/" + fileName) # 移動圖片
            elif os.path.isfile(directoryName + "/" + base + ".JSON"):
                os.replace(directoryName + "/" + base + ".JSON", directoryName + "/Labeled/" + base + ".JSON") # 移動Json
                os.replace(directoryName + "/" + fileName, directoryName + "/Labeled/" + fileName) # 移動圖片
            else:
                # 沒有標註
                os.replace(directoryName + "/" + fileName, directoryName + "/Non-labeled/" + fileName) # 移動圖片

read_directory("tempData/negative_4/")
read_directory("tempData/negative_6/")
read_directory("tempData/negative_high/")
read_directory("tempData/negative_low/")
read_directory("tempData/positive_3/")
read_directory("tempData/positive_5/")
read_directory("tempData/positive_high/")
read_directory("tempData/positive_high_2/")
read_directory("tempData/positive_low/")
read_directory("tempData/positive_low_2/")
import os
import cv2
import json

data_dir = 'G:\\dataset\\test-B-image' #测试集路径

img_files = os.listdir(data_dir)
id = 0
imgs_info = []

for img_flie in img_files:
  id += 1
  img_path = os.path.join(data_dir, img_flie)
  img = cv2.imread(img_path)
  h = img.shape[0]
  w = img.shape[1]

  ann_info = {"file_name":img_flie,"height":h, "width":w , "id":id}
  imgs_info.append(ann_info)

annotations_info = {}
annotations_info['images'] = imgs_info

with open('./test2014.json', 'w') as f:
  json.dump(annotations_info, f, indent=4)

#注：在转化成功的json文件末尾加入"categories": [{"supercategory": "none", "id": 1, "name": "holothurian"}, 
# {"supercategory": "none", "id": 2, "name": "echinus"}, {"supercategory": "none", "id": 3, "name": "scallop"}, 
# {"supercategory": "none", "id": 4, "name": "starfish"}]  #name是标签类别
# 1 准备DataSet

##   LabelImg标注的.xml文件转成MSCoco的.json文件

**参考代码：**

```python
import os
import glob
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET
 
path2 = "."
 
START_BOUNDING_BOX_ID = 1
 
 
def get(root, name):
    return root.findall(name)
 
 
def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars
 
 
def convert(xml_list, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}
    for index, line in enumerate(xml_list):
        # print("Processing %s"%(line))
        xml_f = line
        tree = ET.parse(xml_f)
        root = tree.getroot()
        
        filename = os.path.basename(xml_f)[:-4] + ".jpg"
        image_id = 20210000001 + index
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width, 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category in all_categories:
                all_categories[category] += 1
            else:
                all_categories[category] = 1
            if category not in categories:
                if only_care_pre_define_categories:
                    continue
                new_id = len(categories) + 1
                print("[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(category, pre_define_categories, new_id))
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert(xmax > xmin), "xmax <= xmin, {}".format(line)
            assert(ymax > ymin), "ymax <= ymin, {}".format(line)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
 
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print("------------create {} done--------------".format(json_file))
    print("find {} categories: {} -->>> your pre_define_categories {}: {}".format(len(all_categories), all_categories.keys(), len(pre_define_categories), pre_define_categories.keys()))
    print("category: id --> {}".format(categories))
    print(categories.keys())
    print(categories.values())
 
if __name__ == '__main__':
    classes = ['holothurian', 'echinus', 'scallop', 'starfish']#数据集标签类别
    pre_define_categories = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i + 1
    # pre_define_categories = {'a1': 1, 'a3': 2, 'a6': 3, 'a9': 4, "a10": 5}
    only_care_pre_define_categories = True
    # only_care_pre_define_categories = False
 
    train_ratio = 0.9
    save_json_train = 'instances_train2014.json'
    save_json_val = 'instances_val2014.json'
    xml_dir = "./tmp_xml"
 
    xml_list = glob.glob(xml_dir + "/*.xml")
    xml_list = np.sort(xml_list)
    np.random.seed(100)
    np.random.shuffle(xml_list)
 
    train_num = int(len(xml_list)*train_ratio)
    xml_list_train = xml_list[:train_num]
    xml_list_val = xml_list[train_num:]
 
    convert(xml_list_train, save_json_train)
    convert(xml_list_val, save_json_val)
 
    if os.path.exists(path2 + "/annotations"):
        shutil.rmtree(path2 + "/annotations")
    os.makedirs(path2 + "/annotations")
    if os.path.exists(path2 + "/images/train2014"):
        shutil.rmtree(path2 + "/images/train2014")
    os.makedirs(path2 + "/images/train2014")
    if os.path.exists(path2 + "/images/val2014"):
        shutil.rmtree(path2 +"/images/val2014")
    os.makedirs(path2 + "/images/val2014")
 
    f1 = open("train.txt", "w")
    for xml in xml_list_train:
        img = xml[:-4] + ".jpg"
        f1.write(os.path.basename(xml)[:-4] + "\n")
        shutil.copyfile(img, path2 + "/images/train2014/" + os.path.basename(img))
 
    f2 = open("test.txt", "w")
    for xml in xml_list_val:
        img = xml[:-4] + ".jpg"
        f2.write(os.path.basename(xml)[:-4] + "\n") 
        shutil.copyfile(img, path2 + "/images/val2014/" + os.path.basename(img))
    f1.close()
    f2.close()
    print("-------------------------------")
    print("train number:", len(xml_list_train))
    print("val number:", len(xml_list_val))
```

注：将xml标注文件和图片放到和代码文件同一个目录下的tmp_xml文件内，运行代码即可。

参考链接：https://blog.csdn.net/lcqin111/article/details/103146945

多种数据集格式相互转化参考：https://github.com/spytensor/prepare_detection_dataset

##   测试集制作json文件

**参考代码：**

```python
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
```

注：在转化成功的.json文件末加  `"categories": [{"supercategory": "none", "id": 1, "name": "holothurian"}, {"supercategory": "none", "id": 2, "name": "echinus"}, {"supercategory": "none", "id": 3, "name": "scallop"}, {"supercategory": "none", "id": 4, "name": "starfish"}]  #name是标签类别`

**制作好数据集之后，官方推荐coco数据集按照以下的目录形式存储：**

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

# 2 Training前修改相关文件

##  定义数据种类

需要修改的地方在`mmdet\datasets`，在这个目录下新建一个文件，例如：`underwater_dataset.py`，然后把`coco.py`的内容复制过来，修改class类名为`Underwater_Dataset`最后把`CLASSES`的那个tuple改为自己数据集对应的种类tuple即可，例如：

```
CLASSES = ('holothurian','echinus','scallop','starfish')
```

然后在`mmdet\datasets\__init__.py`文件中引入自定义的数据集。在这个py文件中开头加入：

```
from .underwater_dataset import  Underwater_Dataset
```

最后在`__all__`列表中加入自己的`Underwater_Dataset`

##   加入数据集类别

需要修改的地方在`mmdet\core\evaluation\class_names.py`，在这个py文件中加入：

```
def underwater_classes():
    return [
            'holothurian','echinus','scallop','starfish'
    ]
```

然后在`dataset_aliases`字典中加入：

```
'underwater':['underwater']
```

最后在`mmdet\core\evaluation\__init__.py`引入自定义的数据集。在`__all__`列表中加入自己的`underwater_classes`；在开头的`from .class_names import()`中加入 `underwater_classes`。

##   引入数据路径

需要修改的地方在`config`文件夹下。首先创建自己的文件夹，例如`underwater_detection`，用于存储训练自己数据集的不同模型的配置文件。例如训练Faster RCNN：

在`underwater_detection`创建`faster_rcnn_r50_fpn_1x_uw_coco.py`，其内容如下：

```
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_uw.py',
    '../_base_/datasets/underwater_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
```

然后在`_base_\models`文件夹下创建自己的模型配置文件`faster_rcnn_r50_fpn_uw.py`，将`faster_rcnn_r50_fpn.py`的内容复制过来，修改`num_classes=4  #标签类别数量`。

最后在`_base_\datasets`文件夹下创建自己的数据集配置文件`underwater_detection.py`，将`coco_detection.py`的内容复制过来，修改`dataset_type = 'Underwater_Dataset'  #类名`，修改`data_root = 'data/uw_coco/' #数据集路径`，修改`data = dict()`中的`train=dict()`、`val=dict()`、`test=dict()`，修改为自己的相关文件路径。

# 3 Training

在mmdetection的目录下新建work_dirs文件夹。

```
python tools/train.py configs/underwater_detection/faster_rcnn_r50_fpn_1x_uw_coco.py --work_dir work_dirs
```

其余的参数配置根据`train.py`中的`parse_args()`函数输入。

训练完之后work_dirs文件夹中会保存下训练过程中的log日志文件、每个epoch的pth文件（这个文件将会用于后面的test测试）。

# 4 Testing

根据work_dirs文件夹下的.json文件，分析mAP指标，找到最佳.pth文件。例如：在`epoch_12.pth`中，mAP最高。

```
python tools/test.py configs/underwater_detection/faster_rcnn_r50_fpn_1x_uw_coco.py  work_dirs/epoch_12.pth --out ./result/result_12.pkl
```

详见：https://mmdetection.readthedocs.io/en/latest/index.html

# 5 Log Analysis

详见：https://mmdetection.readthedocs.io/en/latest/useful_tools.html#


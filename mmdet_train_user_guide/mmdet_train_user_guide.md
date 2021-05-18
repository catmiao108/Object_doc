# mmdetection训练过程指导

## 一、准备DataSet

### 1. MScoco数据集制作

LabelImg标注的xml文件转json文件：参考代码：[xml2coco.py](./xml2coco.py)，参考链接：https://blog.csdn.net/lcqin111/article/details/103146945

多种数据集格式相互转化参考：https://github.com/spytensor/prepare_detection_dataset

### 2.测试集制作json文件

参考代码：[test2json.py](./test2json.py)

**制作好数据集之后，官方推荐coco数据集按照以下的目录形式存储：**

```bash
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
├── demo
```

## 二、Training前修改相关文件

### 1.定义数据种类

修改的位置：`mmdet\datasets\`

- 在该目录下新建一个py文件，例如：`underwater_dataset.py`，复制该目录下的`coco.py`内容，修改class类名为`Underwater_Dataset`，再将`CLASSES`的tuple改为自己数据集对应的种类tuple即可，例如：

```python
CLASSES = ('holothurian','echinus','scallop','starfish')  #数据集标签类别
```

- 在`mmdet\datasets\__init__.py`文件中引入自定义的数据集。在这个py文件中开头加入：

```python
from .underwater_dataset import  Underwater_Dataset
```

- 在`mmdet\datasets\__init__.py`文件的`__all__`列表中加入自己的`Underwater_Dataset`。

### 2.加入数据集类别

修改的位置：`mmdet\core\evaluation\`

- 在`mmdet\core\evaluation\class_names.py`文件中加入：

```python
def underwater_classes():
    return [
            'holothurian','echinus','scallop','starfish'
    ]
```

- 在`mmdet\core\evaluation\class_names.py`文件中的`dataset_aliases`字典中加入：

```python
'underwater':['underwater']
```

- 在`mmdet\core\evaluation\__init__.py`文件中引入自定义的数据集：

在`__all__`列表中加入`underwater_classes`；在开头的`from .class_names import()`中加入 `underwater_classes`。

### 3.引入数据路径

修改的位置：`config\`

- 在这个目录下创建自己的文件夹，例如`underwater_detection`，用于存储训练自己数据集的不同模型的配置文件。例如训练Faster RCNN：

在`underwater_detection`文件夹中创建`faster_rcnn_r50_fpn_1x_uw_coco.py`，其内容如下（此内容可复制`config\faster_rcnn\faster_rcnn_r50_fpn_1x_coco.py`的内容作修改）：

```python
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_uw.py',
    '../_base_/datasets/underwater_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
```

- 在`config\_base_\models`文件夹下创建自己的模型配置文件，如：`faster_rcnn_r50_fpn_uw.py`，将`config\_base_\models\faster_rcnn_r50_fpn.py`的内容复制过来，修改：

```python
num_classes=4  #标签类别数量
```

- 在`config\_base_\datasets`文件夹下创建自己的数据集配置文件`underwater_detection.py`，将`config\_base_\datasets\coco_detection.py`的内容复制过来，修改：

```python
dataset_type = 'Underwater_Dataset  #类名
data_root = 'data/uw_coco/ #数据集路径
```

再修改`data = dict()`中的`train=dict()`、`val=dict()`、`test=dict()`，修改为自己的相关文件路径。

## 三、Training

在mmdetection目录下新建work_dirs文件夹。

```bash
python tools/train.py configs/underwater_detection/faster_rcnn_r50_fpn_1x_uw_coco.py --work_dir work_dirs
```

其余的参数配置根据`train.py`中的`parse_args()`函数输入。

训练完之后work_dirs文件夹中会保存下训练过程中的log日志文件和每个epoch的pth文件（这个文件将会用于后面的test测试）。

## 四、Testing

根据work_dirs文件夹下的json文件，分析各项指标，找到最佳模型文件。例如：最佳模型文件是`epoch_12.pth`。

```bash
python tools/test.py configs/underwater_detection/faster_rcnn_r50_fpn_1x_uw_coco.py  work_dirs/epoch_12.pth --out ./result/result_12.pkl
```

详见：https://mmdetection.readthedocs.io/en/latest/index.html

## 五、Log Analysis

详见：https://mmdetection.readthedocs.io/en/latest/useful_tools.html#


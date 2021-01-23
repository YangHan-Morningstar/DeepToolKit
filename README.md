# DeepToolKit

## 一、简介

DeepToolKit用于帮助用户一键处理数据、提供常用训练模型与工具等，用户不需要多次重复编写代码，只需从DeepToolKit中实例化自己所需要的类即可，为用户完成深度学习任务提供强大助力。

## 二、系统说明

### 2.1 主要功能介绍

* 数据划分：用户可以提供原始数据路径，调用系统中的相应类的成员方法后即可自动将数据划分为训练集与验证集。
* 数据特征提取：为将数据转换为神经网络模型可接受的形式以及加快模型的训练速度，用户可自行选择特征并将其处理成相应json文件。
* 常用神经网络模型：常用的神经网络模型以类的形式存在与DeepToolKit中，用户可直接实例化相应类并调用get_model方法获取。
* 常用训练工具：包括callback、generator等实用工具，用户可直接实例化相应类获取。
* 支持任务：目前仅支持CV、NLP、ASP中的分类任务。

### 2.2 基础环境支持

* python3.7及以上版本、tensorflow2.0及以上版本

## 三、使用说明（以图片数据为例）

### 3.1 数据划分

* 原始数据格式如下所示

```bash
/data/source_data
	--category_1
		--data_1.jpg
		--data_2.jpg
		...
	--category_2
		--data_1.jpg
		--data_2.jpg
		...
	...
```

* python代码如下所示

```python
from DeepToolKit.DataManager.CVDataManager import CVDataManager

cv_data_manager = CVDataManager()
cv_data_manager.segmentation("/data/source_data", "/data/seg_data")
```

* 划分后的数据文件格式如下所示

```bash
/data/seg_data
	--train
		--category_1
			--data_1.jpg
			--data_2.jpg
			...
		--category_2
			--data_1.jpg
			--data_2.jpg
			...
		...
	--val
		--category_1
			--data_3.jpg
			--data_4.jpg
			...
		--category_2
			--data_3.jpg
			--data_4.jpg
			...
		...
```

### 3.2 数据特征提取

#### 3.2.1 CV

* 首先需要设置原始文件路径与目标文件路径，其中每个路径格式都与3.1中的格式相同，即种类目录的父目录，设置路径时可以通过CVDataManager类的构造函数设置，也可以在实例化之后调用相应的成员方法

```python
from DeepToolKit.DataManager.CVDataManager import CVDataManager
source_path = "/data/seg_data/train"
target_path = "/data/json_data/RGB_Feature/train"

# 通过构造函数设置
cv_data_manager = CVDataManager(source_path, target_path)

# 通过成员方法设置
cv_data_manager = CVDataManager()
cv_data_manager.set_source_data_filepath(source_path)
cv_data_manager.set_target_filepath(target_path)
```

* 其次获取每条数据的绝对路径以及其对应标签的词典，代码即词典格式如下

```python
cv_data_manager.get_filepath_label_dict()
```

`{"/data/seg_data/train/category_1/data_1.jpg": category_1 }`

* 最后调用成员方法提取特征并写入json文件，每个json文件中仅含有一条数据的特征及其对应的标签，代码即文件格式如下

```python
cv_data_manager.extract_feature_just_rgb(img_rows=299, img_cols=299)
```

`{"data_features": [[[...]]], "label": category_1}`

```bash
/data/json_data/RGB_Feature
	--train
		0.json
		1.json
		...
```

### 3.3 常用神经网络模型

#### 3.3.1 CV

* 调用代码如下

```python
from DeepToolKit.Models.CV.inception_resnet_v2 import InceptionResnetV2Model

img_rows = 299
img_cols = 299

model_class = InceptionResnetV2Model(img_rows=img_rows, img_cols=img_cols, label_num=len(cv_data_manager.label_dict))
model = model_class.get_model()
```

* 接下来便可以调用compile、fit等成员方法进行模型的编译和训练

### 3.4 常用训练工具

#### 3.4.1 generator

* 通常情况下数据集会很大以至于直接使用fit函数会导致内存（显存）爆炸，故建议直接使用fit_generator函数，在这之前，需要调用DeepToolKit中相应的成员方法，代码如下

```python
from DeepToolKit.Train.generator import Generator

train_generator = generator.train_data_generator(batch_size=batch_size, target_path=target_path + "/train", label_num=label_num)
val_generator = generator.val_data_generator(batch_size=batch_size, target_path=target_path + "/val", label_num=label_num)
```

* 其中target_path参数需要设置为提取特征后的json文件所在的父目录（详见3.2.1），label_num即为标签的数目，可以自行填写或通过如下代码获取

```python
label_num = cv_data_manager..get_file_num_in_path("/data/seg_data/train")
```

#### 3.4.2 callback

* 当需要在每个epoch结束后想做点什么时（如保存当前最优模型），可以设定回调函数，DeepToolKit中也提供了相应方法，如下代码所示

```python
from DeepToolKit.Train.callback import CallBack

model_checkpoint_callback = CallBack().get_model_checkpoint_callback("./model_checkpoint", "val_loss", "min")
```

* 其中"val_loss"是需要监测的评价指标，min所描述的是如果新的epoch比上一个epoch结束是的val_loss更小，则保存模型，否则不会保存
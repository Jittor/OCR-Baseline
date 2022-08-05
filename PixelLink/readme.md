# PixelLink
本模块使用 PixelLink 模型，在店面招牌文字识别任务中，将仅包含店面招牌的图片，检查出招牌内所有文本的位置。

## 示例
输入店面招牌图片：

![image1](https://user-images.githubusercontent.com/73881739/182278142-9f8d54bb-5e41-4273-a9ce-9235970c4100.jpg)

检查文本框的位置：

![image2](https://user-images.githubusercontent.com/73881739/182278003-d610982a-b419-4243-8e2c-eccc55aa61a3.jpg)

## 使用说明

### 1. 数据集路径配置

在 ```PixelLink/config_pl.py``` 中将 ```Config.dataset_path``` 配置为数据集所处路径。

### 2. 训练
```python
python train.py
```
### 3. 评估
```python
python train.py --mode val --val_ckp "./work_dirs/exp/PixelLink.pkl"
```
（注：```--val_ckp``` 传入要评估的模型路径。）

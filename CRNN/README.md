# CRNN

本模块使用 CRNN 模型，在店面招牌文字识别任务中，将仅包含文本的图片，识别出对应的文字。

## 示例

输入图片：

![image](https://user-images.githubusercontent.com/73881739/182277036-facf8ec7-c4ba-43ad-acbb-44cc62f44a92.jpg)

输出文字：**扬州**


## 使用说明


### 1. 数据集路径配置

在 ```CRNN/src/config.py``` 中将 ```datasets_path``` 配置为数据集所处路径。


### 2. 训练

直接开始训练，请在本目录下执行：
```bash
python src/train.py
```

加载预训练模型，再微调训练：
```bash
python src/train.py -r ../ckpts/crnn_pretrain.pkl
```

查看其他选项，请执行：
```bash
python src/train.py -h
```

### 3. 评估

对模型进行评估：
```bash
python src/evaluate.py -r CHECKPOINT
```
（注：```CHECKPOINT``` 为要评估的模型路径。）

查看其他选项，请执行：
```bash
python src/evaluate.py -h
```

### 4. 预测

要使用模型，对图片进行预测，请执行：
```bash
python src/predict.py -r CHECKPOINT --images demo/*.jpg
```
（注：```-r``` 传入模型路径； ```--images``` 传入图片路径。）

查看其他选项，请执行：
```bash
python src/predict.py -h
```

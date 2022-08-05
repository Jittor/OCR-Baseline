# JDet
本模块使用 JDet库 模型，在店面招牌文字识别任务中，将图片中的店面招牌的位置检测出来

## 示例
输入店面招牌图片：
<div style="text-align: center">
<img src="docs/images/img.png"/>
</div>

检测招牌的位置：
<div style="text-align: center">
<img src="docs/images/pred.png"/>
</div>

### 1. 安装
按照原始项目[](https://github.com/Jittor/JDet)安装好相关依赖，在JDet目录下
```shell
python setup.py develop
```

### 2. 数据集路径配置
以使用s2anet+ResNet50为例，在store_sign_detection/s2anet_r50_fpn_5x_ocr_630_1120_bs4.py中将dataset.train.dataset_dir修改为数据集所处路径。

### 3. 训练
```python
python tools/run_net.py --config-file=store_sign_detection/s2anet_r50_fpn_5x_ocr_630_1120_bs4.py --task=train
```
### 4. 评估
在store_sign_detection/s2anet_r50_fpn_5x_ocr_630_1120_bs4.py文件最后一行加上resume_path="JDet.pkl"，
使用训练好的的[权重](https://cloud.tsinghua.edu.cn/f/0b1ed1cc311245ed901c/?dl=1)进行评估。

执行评估：
```python
python tools/run_net.py --config-file=store_sign_detection/s2anet_r50_fpn_5x_ocr_630_1120_bs4.py --task=val
```

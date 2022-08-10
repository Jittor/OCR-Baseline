# 粤港澳街景图像中店面招牌文字识别比赛

本仓库是官方提供的 Baseline 代码，为了方便各位选手使用和交流我们的官方 Baseline 代码和计图框架，我们创建了官方 QQ 群聊 640858766，后续比赛通知、代码更新、技术讨论都会在这个群进行，请扫下面的二维码入群。

<img src="https://user-images.githubusercontent.com/73881739/183235284-8ba759d4-2376-4372-86ad-3db55b75dabe.jpg" width="220" height="260">

Jittor 是一个基于即时编译和元算子的高性能深度学习框架，整个框架在即时编译的同时，还集成了强大的 Op 编译器和调优器，为您的模型生成定制化的高性能代码。Jittor 还包含了丰富的高性能模型库，涵盖范围包括：图像识别、检测、分割、生成、可微渲染、几何学习、强化学习等。

Jittor 前端语言为 Python，使用了主流的包含模块化和动态图执行的接口设计，后端则使用高性能语言进行了深度优化。更多关于 Jittor 的信息可以参考：
*  [Jittor 官网](https://cg.cs.tsinghua.edu.cn/jittor/)
*  [Jittor 教程](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/)
*  [Jittor 模型库](https://cg.cs.tsinghua.edu.cn/jittor/resources/)
*  [Jittor 文档](https://cg.cs.tsinghua.edu.cn/jittor/assets/docs/index.html)
*  [Github 开源仓库](https://github.com/jittor/jittor)
*  [Jittor 论坛](https://discuss.jittor.org)

使用之前请先通过 pip install -U jittor 更新 Jittor 框架，确保版本在 1.3.5 以上。



## 项目介绍

店面招牌是街景图像中的重要信息，自然场景下的文字识别也是计算机视觉的重要研究方向，两者结合的街景店面招牌文字识别技术正在大规模应用在地图导航及推荐、智能城市规划分析、商业区商业价值分析等实际落地领域，具有很高的研究价值和业务使用价值。

比赛网址: https://www.cvmart.net/race/10351/des 

本项目基于国产框架计图 Jittor，实现了粤港澳街景图像中店面招牌文字识别比赛的 Baseline，整体采用了 **‘JDet-PixelLink-CRNN’** 的模型组合。

### *总体思路：*
>    Step 1. 使用 **JDet** 检测出街景图像中的店面招牌；  
>
>    Step 2. 使用 **PixelLink** 从招牌图像中检测出文本框位置，并选择最大的文本框作为候选；  
> 
>    Step 3. 使用 **CRNN** 从文本框图像中识别出文字。


### *示例：*


<img src="https://user-images.githubusercontent.com/73881739/182287493-a53ecba5-6b74-487e-bba8-2712d2771f9a.png" height="450">




## 训练与评测

### 模型训练
请移步到各模型目录下阅读 ```README.md```。

### 评测

请先确认 ```model.py``` 文件中，模型路径已配置正确。随后，请执行：
```bash
python evaluate.py
```

或者，设置其他参数进行评测：
```bash
python evaluate.py -i [街景图片目录] -g [真实标签目录] -t1 [文字对比阈值] -t2 [边框对比阈值]
```




## 模型参数

已训练好的模型参数可通过以下地址进行下载：


**JDet 模型参数**:  
https://cloud.tsinghua.edu.cn/f/0b1ed1cc311245ed901c/?dl=1


**PixelLink 模型参数**:  
https://cloud.tsinghua.edu.cn/f/4e7e633085e241729e14/?dl=1


**CRNN 模型参数**:  
https://cloud.tsinghua.edu.cn/f/2fc669b2dd6e475b9511/?dl=1


**CRNN 预训练模型参数**:   
https://cloud.tsinghua.edu.cn/f/80825967dd344a91a2da/?dl=1

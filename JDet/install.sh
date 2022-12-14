# 下载pretrained model
rm resnet50.pkl
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-39297-files/ba794a50-dbf4-4fb6-8b54-99a6de24788e/resnet50.pkl
rm /project/train/models/resnet50.pkl 
cp resnet50.pkl /project/train/models/resnet50.pkl
# 下载相关依赖
rm terminaltables-3.1.10-py2.py3-none-any.whl
rm PyYAML-5.4.1-cp38-cp38-manylinux1_x86_64.whl
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-39297-files/d7525b7f-15d6-4da4-a8aa-eeb96f0352e7/terminaltables-3.1.10-py2.py3-none-any.whl
wget https://extremevision-js-userfile.oss-cn-hangzhou.aliyuncs.com/user-39297-files/637df426-490d-4150-8893-5e8dbe17b3aa/PyYAML-5.4.1-cp38-cp38-manylinux1_x86_64.whl
# 安装依赖
python -m pip install PyYAML-5.4.1-cp38-cp38-manylinux1_x86_64.whl
python -m pip install terminaltables-3.1.10-py2.py3-none-any.whl
pip uninstall jdet -y
python setup.py install

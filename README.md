# Usage

- Install efficientnet-pytorch
```
pip install efficientnet_pytorch
```

- Download [pretrained models](https://drive.google.com/file/d/136xL5zmsa-OCGGeNKTjMMGqJq5UhFFUe/view?usp=sharing) and unzip them



- Download data files and change directories in configs.py
```
train_label_dir = './train/label_train.txt'
train_dir = './train'
val_dir = './validation'
```
> Make sure train_dir and val_dir only contain .jpg files

- If you want to use pretrained models, you cna do it by changing the configs.py
```
use_pretrain = True
```

- Training

```
python3 train.py
```
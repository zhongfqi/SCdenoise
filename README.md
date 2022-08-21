# SCdenoise: a reference-based scRNA-seq denoising method using semi-supervised learning
## Fengqi Zhong, Yuansong Zeng, Yubao Liu, Yuedong Yang*
Here, we proposed SCdenoise, a semi-supervised denoising method to denoise unlabeled target data based on annotated information in the reference datasets.This repository contains the preprocessed data and Python implementation for SCdenoise.
<img width="1277" alt="model" src="https://user-images.githubusercontent.com/43873722/185773344-248c0192-4f13-478e-94f6-30433a565def.png">

# Requirements
pyTorch >= 1.1.0  
scanpy >= 1.9.1  
python >=3.7  
umap-learn>=0.5.2

# Pipeline
See [tutorial.ipynb](https://github.com/zhongfqi/SCdenoise/blob/master/tutorial.ipynb) for more details.

## Python script usage
```
python main.py --dataset_path ./dataset/ --result_path ./results/ --dataset simulated_drop1 --source_name batch2 --target_name batch1 --gpu_id 0
```



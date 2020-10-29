# MixMatch
This is an unofficial PyTorch implementation of [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249). 
The official Tensorflow implementation is [here](https://github.com/google-research/mixmatch).

The google drive links to the models are provided below for all the trained models. 

The accuracy and loss plots for all the trained models are provided in the folder 'figures' by their names. The figure 'AccLossPlot_50.png' is the plot for the experiment with 50 labeled samples and so on. 

Experiments on Cifar10 dataset are conducted with the different numbers of labeled samples. i.e: 50, 100, 200, 250 and 500 and also on categorized classes(super classes)

The results obtained on the above scenarios are as follows:

## Results (Accuracy)
| #Labels | 50 | 100 | 200 | 250| 500 |
|:---|:---:|:---:|:---:|:---:|:---:|
|Accuracy | 70.48 | 82.86 | 88.74 | 90.08 | 89.88 |

The result obtained fared slightly better than the reported result of the code and matched with that of the paper results.

## Results obtained on categorization of cifar10 classes(4 super classes) with 250 labeled samples

Accuracy: 84.85%, No. of Labeled examples: 250. 

The 10 classes in the dataset are divided into 4 super classes. While airplane and ship are categorized under one class as they are non-road transportation mode, the automobile and trucks are categorized under one super class due to very high similarities between them. The remaining 6 classes are further categorized into  more super classes based on the similarities in appearance. The `cat`, `deer`, `dog` and `horse` are categorized to a 4-legged animals while the `bird` and `frog` are categorizzed into small animals category. Another intuitive way to categorize these classes could be to check the correlation score between the word vectors of these 10 classes and group the similar classes together.

Differences made in the code to adapt the changes:

For experiment with 50 examples, the batch size is taken to be 48. For experiment with categorized classes, the mapping of class labels are made in the `_cifar10.py` file in `getcifar10`, `train_val_split` functions and in `CIFAR10_labeled` class. 


## Requirements
- Python 3.6+
- PyTorch 1.0
- **torchvision 0.2.2 (older versions are not compatible with this code)** 
- tensorboardX
- progress
- matplotlib
- numpy

## Usage

### Train
Train the model by 250 labeled data of CIFAR-10 dataset:

```
python train.py --gpu <gpu_id> --n-labeled 250 --out cifar10@250
```

Train the model by 4000 labeled data of CIFAR-10 dataset:

```
python train.py --gpu <gpu_id> --n-labeled 4000 --out cifar10@4000
```

### Monitoring training progress
```
tensorboard.sh --port 6006 --logdir cifar10@250
```

## References
```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```

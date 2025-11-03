# LACF
This is the PyTorch implementation of the paper that we submitted for under review
>Graph Self-Supervised Learning via Learnable View Augmentation for Recommender System

>Anonymous Author(s)

## Environment Requirement

The code runs well under python 3.8.0. The required packages are as follows:
- torch == 2.0.1
- torch-cluster == 1.6.1+pt20cu118
- torch-scatter == 2.1.1+pt20cu118
- torch-sparse == 0.6.17+pt20cu118
- numpy == 1.22.4
- scipy == 1.7.0
- tqdm == 4.62.3

## How to run the codes

* Gowalla
```
python main.py --dataset gowalla 
```

* Amazon
```
python main.py --dataset amazon 
```

* Tmall
```
python main.py --dataset tmall 
```






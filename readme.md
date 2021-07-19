# Large-scale Dynamic Linear Model (LSDLM)
This is a Python implementation of Large-scale dynamic linear model in the following paper:
https://arxiv.org/abs/2104.13414

# Requirements
```angular2html
pandas==1.3.0
numpy==1.21.0
scikit-learn==0.24.2
scipy==1.7.0
tqdm==4.61.2
```
Dependency can be installed using the following command (recommanded to install in a venv):
```angular2html
pip install -r requirements.txt
```

# Train and evaluation
```angular2html
python main.py
```
```angular2html
loading dataset... done.
pre-processing the dataset...: 100%|█████████| 325/325 [00:01<00:00, 318.11it/s]
splitting dataset to training and test set (8:2 ratio)... done.
prediction for h=3...: 288it [00:01, 248.01it/s]
RMSE: 2.9031507875713225

prediction for h=6...: 288it [00:01, 164.04it/s]
RMSE: 3.764874720345656

prediction for h=12...: 288it [00:03, 79.01it/s]
RMSE: 4.438127452863711
```
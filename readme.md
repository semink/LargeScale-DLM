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
pre-processing the dataset...: 100%|█████████| 325/325 [00:01<00:00, 321.49it/s]
splitting dataset to training and test set (8:2 ratio)... done.
tau_short = 0.0001
tau_long = 1000
model created... start to train...
100%|█████████████████████████████████████████| 288/288 [16:28<00:00,  3.43s/it]
training finished!
prediction for h=12...: 288it [00:04, 57.86it/s]
RMSE: 4.438168067142786

Total computation for prediction: 5.1492767333984375 sec
```
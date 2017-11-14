# Homework3

## Task 1
The implementation of `alexnet` is in `alexnet.py` file. 
In order to run `alexnet.py`, we have to place dataset 
under `data/imagenet10` folder. 

Currently I only use 10 classes image data from `imagenet1000`
dataset because training on 1000 classes from scratch is too slow.
After training 24 hours on a single `GTX1080` can only achieve 
`9%` accuracy on train dataset, which contains 40000 images in 1000
categories. 

The best model trained on 10 classes dataset,
which contains around 12000 images, can achieve `99.4%` 
accuracy on training and achieve `84.1%` accuracy on validation set.

The training process includes data augmentation procedure.

Here includes some training log:
```
Epoch 77/100
1249/1250 [============================>.] - ETA: 0s - loss: 0.0142 - acc: 0.9949Epoch 00077: val_acc improved from 0.84103 to 0.84103, saving model to weights.be$
t.hdf5
1250/1250 [==============================] - 512s 410ms/step - loss: 0.0142 - acc: 0.9949 - val_loss: 1.3126 - val_acc: 0.8410
Epoch 78/100
1249/1250 [============================>.] - ETA: 0s - loss: 0.0138 - acc: 0.9947Epoch 00078: val_acc did not improve
1250/1250 [==============================] - 487s 390ms/step - loss: 0.0138 - acc: 0.9947 - val_loss: 1.5278 - val_acc: 0.8205
Epoch 79/100
1249/1250 [============================>.] - ETA: 0s - loss: 0.0147 - acc: 0.9949Epoch 00079: val_acc did not improve
1250/1250 [==============================] - 498s 398ms/step - loss: 0.0147 - acc: 0.9949 - val_loss: 1.5069 - val_acc: 0.8090
Epoch 80/100
1249/1250 [============================>.] - ETA: 0s - loss: 0.0138 - acc: 0.9947Epoch 00080: val_acc did not improve
1250/1250 [==============================] - 502s 401ms/step - loss: 0.0138 - acc: 0.9947 - val_loss: 1.5680 - val_acc: 0.8192
```

We can see model starting overfit on training set after reach the best 
score on validation set.

Model testing part, prediction, is included in `Prediction.ipynb` file. 

## Task 2



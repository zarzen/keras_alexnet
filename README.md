# Homework3 Report

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
1249/1250 [============================>.] - ETA: 0s - loss: 0.0142 - acc: 0.9949
Epoch 00077: val_acc improved from 0.84103 to 0.84103, saving model to weights.be$
t.hdf5
1250/1250 [==============================] - 512s 410ms/step - loss: 0.0142 - acc: 0.9949 
- val_loss: 1.3126 - val_acc: 0.8410
Epoch 78/100
1249/1250 [============================>.] - ETA: 0s - loss: 0.0138 - acc: 0.9947
Epoch 00078: val_acc did not improve
1250/1250 [==============================] - 487s 390ms/step - loss: 0.0138 - acc: 0.9947 
- val_loss: 1.5278 - val_acc: 0.8205
Epoch 79/100
1249/1250 [============================>.] - ETA: 0s - loss: 0.0147 - acc: 0.9949
Epoch 00079: val_acc did not improve
1250/1250 [==============================] - 498s 398ms/step - loss: 0.0147 - acc: 0.9949 
- val_loss: 1.5069 - val_acc: 0.8090
Epoch 80/100
1249/1250 [============================>.] - ETA: 0s - loss: 0.0138 - acc: 0.9947
Epoch 00080: val_acc did not improve
1250/1250 [==============================] - 502s 401ms/step - loss: 0.0138 - acc: 0.9947 
- val_loss: 1.5680 - val_acc: 0.8192
```

We can see model starting overfit on training set after reach the best 
score on validation set.

Model testing part, prediction, is included in `Prediction.ipynb` file. 

## Task 2
Implementation of `fully convolutional neural networks` is placed in `fcn_vgg16.py`.
The network structure is based on `vgg16`, thus I use pre-trained `vgg16` weights.
The weights of last three layers in `vgg16` have been reshaped to fit into `FCN_VGG16`.

The reshape scripts refer following:
```
vgg16 = VGG16() # VGG16 model from keras
for layer in vgg16.layers:
    weights = layer.get_weights()
    if layer.name=='fc1':
        weights[0] = np.reshape(weights[0], (7,7,512,4096))
    elif layer.name=='fc2':
        weights[0] = np.reshape(weights[0], (1,1,4096,4096))
    elif layer.name=='predictions':
        layer.name='predictions_1000'
        weights[0] = np.reshape(weights[0], (1,1,4096,1000))
    if index.has_key(layer.name):
        index[layer.name].set_weights(weights)
```
It load `VGG16` weights first, then reshapes last three layers from fully connected weights. 

I am using `VOC2012` dataset, which contains 11120 images for training and 904 for validation.
The loss function is pixel wised cross entropy loss function.

The Adam optimizer is slower than SGD optimizer with `0.9` momentum in my experiments setups,
using same `learning_rate=0.001`. Following are training logs for two optimizer 
respectively.  

```
# Adam
Epoch 1/100
694/695 [============================>.] - ETA: 0s - loss: 1.6255 - accuracy_ignoring_last_label: 0.5834
Epoch 00000: val_loss improved from inf to 1.59723, saving model to seg_checkpoint_weights.hdf5
695/695 [==============================] - 592s - loss: 1.6251 - accuracy_ignoring_last_label: 0.5835 
- val_loss: 1.5972 - val_accuracy_ignoring_last_label: 0.6216
Epoch 2/100
694/695 [============================>.] - ETA: 0s - loss: 1.3828 - accuracy_ignoring_last_label: 0.5864
Epoch 00001: val_loss improved from 1.59723 to 1.40242, saving model to seg_checkpoint_weights.hdf5
695/695 [==============================] - 585s - loss: 1.3830 - accuracy_ignoring_last_label: 0.5863 
- val_loss: 1.4024 - val_accuracy_ignoring_last_label: 0.6234

# SGD
Epoch 1/100
694/695 [============================>.] - ETA: 0s - loss: 0.7014 - sparse_accuracy_ignoring_last_label: 0.7464
Epoch 00000: val_loss improved from inf to 0.51831, saving model to seg_checkpoint_weights.hdf5
695/695 [==============================] - 554s - loss: 0.7014 - sparse_accuracy_ignoring_last_label: 0.7464 
- val_loss: 0.5183 - val_sparse_accuracy_ignoring_last_label: 0.8191
Epoch 2/100
694/695 [============================>.] - ETA: 0s - loss: 0.4930 - sparse_accuracy_ignoring_last_label: 0.8082
Epoch 00001: val_loss improved from 0.51831 to 0.48101, saving model to seg_checkpoint_weights.hdf5
695/695 [==============================] - 551s - loss: 0.4928 - sparse_accuracy_ignoring_last_label: 0.8083 
- val_loss: 0.4810 - val_sparse_accuracy_ignoring_last_label: 0.8308

```
We can see `Adam` optimizer is not panacea. 

With large learning rate will easily boost `loss` value to a very large number.
Following use `learning_rate=0.01` with SGD.
```
Epoch 1/100
33/695 [>.............................] - ETA: 575s - loss: 2292.9412 - accuracy_ignoring_last_label: 0.3691
```

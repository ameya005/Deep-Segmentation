#!/usr/bin/env python
'''
Implentation of 
Pinheiro, P. and Collobert, R.; From Image-level to Pixel-level Labeling with Convolutional Networks
http://arxiv.org/pdf/1411.6228v3.pdf

The original paper uses Overfeat as it's convNet. I will be using VGG with CIFAR since that is easily available with keras

'''

import os,sys
from keras.models import Sequential
from keras.layers import * ##TODO: Fix layers to be imported
from keras.optimizers import RMSprop
from keras.activations import relu,sigmoid
import keras.backend as K
from keras.objectives import categorical_crossentropy
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,TensorBoard
from keras.datasets import cifar10
from pretrained import imagenet_utils,vgg16
from keras.models import model_from_json

#required for custom layer
from keras.engine.topology import Layer
from keras.layers.core import Lambda
from keras.utils.data_utils import print_function
#Refer to listed paper for implementation of the Log Sum Exp Layer

import argparse

def LogSumExp(x,r, num_classes ):
    #print_function('x shape:',x.shape)
    shape = K.int_shape(x)
    print(shape)
    #assuming tf weights
    y_out = (1./r) *  K.log( (1./(shape[1] * shape[2])) * K.sum( K.exp(r*x), axis=[1,2] ) )
    #y_out = K.exp(r*x)
    return y_out
    
    
def log_sum_out(input_shape):
    shape = list(input_shape)
    return tuple([shape[0],shape[-1]])
    


class DeconvSegmenter():

    def __init__(self, model,  num_classes=2,r=5):
        #self.model = vgg16.VGG16()
        ##Freezing the initial convolutional layers
        for layer in self.model.layers:
            layer.trainable=False

        #Adding 4 convolutional layers as given in the paper : (May need to modify the #filters
        self.model.add(Conv2D(1024, 3, 3, activation='relu'))
        self.model.add(Conv2D(512,3,3,activation='relu'))
        self.model.add(Conv2D(256,3,3,activation='relu'))
        #The following layer outputs probability segmentation maps. 
        self.model.add(Conv2D(num_classes,3,3,activation='linear'))
        #New layer to be added here for the softmax aggregation.
        #The Log Sum Exp layer follows the follwoing aggregation rule:
        # s^k = (1/r) * log ( (1/(h_0*w_0) ) * \sigma(exp(r*(s_ij)^k)))
        # r = 5 for smoothness for now.
        self.model.add(Lambda(LogSumExp, output_shape=logsum_out,
                              arguments={'r':r,'num_classes'=num_classes}))
	self.model.add(Activation('softmax'))
	#This should give us a softmax vector to get a class prediction label

    def train(self, model_file, datagen, cifar_flag = True, 
              train_path='.',validate_path='.', nb_epoch=100,batch_size=32):

        if cifar_flag:
            self.model.compile(optimizer='rmsprop',loss='categorical_crossentropy',
                               metrics=['accuracy'])
            (X_train,y_train),(X_test,y_test) = cifar10.load_data()
            datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
            

            self.model.fit_generator(datagen.flow(X_train,y_train,batch_size=batch_size),
                                     samples_per_epoch = X_train.shape[0],
                                     nb_epoch = nb_epoch,
                                     validation_data = (X_test,y_test),
                                     callbacks = 
                                     [
                                         ModelCheckpoint('weights.h5py',monitor='val_acc',
                                                         verbose=1,mode='max',
                                                         save_best_only='true'),
                                         TensorBoard(log_dir='./logs',
                                                     histogram_freq=10,write_graph=True),
                                     ])
            print('Saving model at %s'%model_path)
        return self.model
       
    def predict(self,model,X):
       #Removing the final aggregation and softmax layers. 
       #Now, we only need the segmentation maps
       self.model.pop()
       self.model.pop()
       self.model.add(Activation('softmax'))
       self.model.compile()
       seg_maps = self.model.predict(X,batch_size,verbose=1)
       return seg_maps

 


if __name___ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path',help='Path to model file')
    parser.add_argument('weights_path',help='Path to weights file')
    parser.add_argument('-b','--batchsize',help='Batch Size',type=int)
    parser.add_argument('-n','--nb_epoch',help='Number of epochs',type=int,default=32)

    args = parser.parse_args()
    model_path = args.model_path
    weights_path = args.weights_path
    batch_size = args.batch_size
    nb_epoch = args.nb_epoch
    
    (X_train,y_train),(X_test,y_test) = cifar10.load_data()
    
    ##Reading model
    with open(model_path,'r') as f:
        model_json = f.read()

    model = model_from_json(model_json)

    model.load_weights(weights_path)
    print('Loaded Model from disk')
    
    ##Removing the MLP layer
    for i in xrange(6):
        model.pop()
    print(model.layer[-1])

    segmenter = DeconvSegmenter(model,num_classes=10,r=5)
    new_model = segmenter.train()

    json_out = new_model.to_json()
    with open('SegModel.json','w') as f:
        f.write(json_out)

    for i in xrange(10):
        segmaps = segmenter.predict(new_model,X[i])
        plt.imshow(np.hstack(segmaps))

     




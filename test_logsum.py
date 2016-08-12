import os,sys
from keras.models import Sequential
from keras.layers import * ##TODO: Fix layers to be imported
from keras.optimizers import RMSprop
from keras.activations import relu,sigmoid
import keras.backend as K
from keras.objectives import categorical_crossentropy
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,TensorBoard
from pretrained import imagenet_utils,vgg16
from keras.utils import np_utils
#required for custom layer
from keras.engine.topology import Layer
from keras.layers.core import Lambda
from keras.utils.data_utils import print_function
from keras.datasets import cifar10
from keras.activations import softmax
def LogSumExp(x,r, num_classes ):
    #print_function('x shape:',x.shape)
    shape = K.int_shape(x)
    print(shape)
    #assuming tf weights
    y_out = (1./r) *  K.log( (1./(shape[2] * shape[3])) * K.sum( K.exp(r*x), axis=[2,3] ) )
    #y_out = K.exp(r*x)
    return y_out


def log_sum_out(input_shape):
    shape = list(input_shape)
    return tuple([shape[0],shape[-1]])



if __name__ == '__main__':
    nb_classes=10
    (X_train,y_train),(X_test,y_test) = cifar10.load_data()
    Y_train = np_utils.to_categorical(y_train, nb_classes)  
    print(X_train[0].shape)
    model = Sequential()
    #model.add(ZeroPadding2D((0,2,2),input_shape=X_train[0].shape))
    #X_train = X_train.transpose(0,2,3,1)
    model.add(Convolution2D(8,3,3,activation='relu',input_shape=(3,32,32)))
    model.add(MaxPooling2D())
    model.add(Convolution2D(10,3,3,activation='linear'))
    model.add(Lambda(LogSumExp,output_shape=log_sum_out,arguments={'r':5,'num_classes':2}))
    model.add(Activation('softmax'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    num = 15
    input_1 = np.float32(X_train)
    for i in xrange(X_train.shape[0]):
        input_1[i] = ( input_1[i] - input_1[i].mean() )/ input_1[i].std()
    
    model.fit(input_1,Y_train,batch_size=10,nb_epoch=10,validation_split=0.2)
    
    a = model.predict(input_1,batch_size=3,verbose=1)

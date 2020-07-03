import keras
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *


def get_model():
	aliases = {}
	Input_0 = Input(shape=(3, 56, 56), name='Input_0')
	convolution2d_889 = Convolution2D(name='convolution2d_889',nb_filter= 64,activation= 'relu' ,dim_ordering= 'th' ,nb_row= 2,border_mode= 'same' ,nb_col= 2)(Input_0)
	convolution2d_890 = Convolution2D(name='convolution2d_890',nb_filter= 64,activation= 'linear' ,dim_ordering= 'th' ,nb_row= 2,border_mode= 'same' ,nb_col= 2)(convolution2d_889)
	maxpooling2d_219 = MaxPooling2D(name='maxpooling2d_219',strides= (2, 2),dim_ordering= 'th' )(convolution2d_890)
	convolution2d_891 = Convolution2D(name='convolution2d_891',nb_filter= 32,activation= 'relu' ,dim_ordering= 'th' ,nb_row= 2,border_mode= 'same' ,nb_col= 2)(maxpooling2d_219)
	convolution2d_892 = Convolution2D(name='convolution2d_892',nb_filter= 32,activation= 'linear' ,dim_ordering= 'th' ,nb_row= 2,border_mode= 'same' ,nb_col= 2)(convolution2d_891)
	batchnormalization_290 = BatchNormalization(name='batchnormalization_290')(convolution2d_892)
	maxpooling2d_220 = MaxPooling2D(name='maxpooling2d_220',strides= (2, 2),dim_ordering= 'th' )(batchnormalization_290)
	convolution2d_893 = Convolution2D(name='convolution2d_893',nb_filter= 32,activation= 'relu' ,dim_ordering= 'th' ,nb_row= 2,border_mode= 'same' ,nb_col= 2)(maxpooling2d_220)
	convolution2d_894 = Convolution2D(name='convolution2d_894',nb_filter= 32,activation= 'linear' ,dim_ordering= 'th' ,nb_row= 2,border_mode= 'same' ,nb_col= 2)(convolution2d_893)
	convolution2d_895 = Convolution2D(name='convolution2d_895',nb_filter= 32,activation= 'relu' ,dim_ordering= 'th' ,nb_row= 2,border_mode= 'same' ,nb_col= 2)(convolution2d_894)
	convolution2d_896 = Convolution2D(name='convolution2d_896',nb_filter= 32,activation= 'relu' ,dim_ordering= 'th' ,nb_row= 2,border_mode= 'same' ,nb_col= 2)(convolution2d_895)
	maxpooling2d_221 = MaxPooling2D(name='maxpooling2d_221',strides= (2, 2),dim_ordering= 'th' )(convolution2d_896)
	flatten = Flatten(name='flatten')(maxpooling2d_221)
	dense_125 = Dense(name='dense_125',output_dim= 2048,activation= 'linear' )(flatten)
	activation_237 = Activation(name='activation_237',activation= 'relu' )(dense_125)
	dense_126 = Dense(name='dense_126',output_dim= 1024,activation= 'linear' )(activation_237)
	batchnormalization_291 = BatchNormalization(name='batchnormalization_291')(dense_126)
	activation_238 = Activation(name='activation_238',activation= 'relu' )(batchnormalization_291)
	dropout_32 = Dropout(name='dropout_32',p= 0.4)(activation_238)
	dense_127 = Dense(name='dense_127',output_dim= 1024,activation= 'linear' )(dropout_32)
	activation_239 = Activation(name='activation_239',activation= 'relu' )(dense_127)
	dense_128 = Dense(name='dense_128',output_dim= 3,activation= 'softmax' )(activation_239)

	model = Model([Input_0],[dense_128])
	return aliases, model


from keras.optimizers import *

def get_optimizer():
	return Adadelta()

def is_custom_loss_function():
	return False

def get_loss_function():
	return 'categorical_crossentropy'

def get_batch_size():
	return 32

def get_num_epoch():
	return 10

def get_data_config():
	return '{"dataset": {"name": "back_data_3T", "type": "private", "samples": 1643}, "mapping": {"Images": {"shape": "", "options": {"vertical_flip": false, "pretrained": "None", "Normalization": true, "width_shift_range": 0, "Resize": true, "shear_range": 0, "horizontal_flip": false, "Augmentation": false, "Height": "56", "Scaling": 1, "rotation_range": 0, "height_shift_range": 0, "Width": "56"}, "port": "InputPort0", "type": "Image"}, "Target": {"shape": "", "options": {}, "port": "OutputPort0", "type": "Categorical"}}, "numPorts": 1, "samples": {"validation": 771, "test": 100, "split": -1, "training": 771}, "kfold": 2, "datasetLoadOption": "batch", "shuffle": true}'
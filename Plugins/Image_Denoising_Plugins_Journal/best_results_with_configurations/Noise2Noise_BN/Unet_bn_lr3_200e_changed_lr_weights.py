import numpy as np
import os
import sys
import time
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential

import random
import pandas as pd
#import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
plt.style.use("ggplot")
#matplotlib inline

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow.keras as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, UpSampling2D, LeakyReLU, ReLU, Subtract, Add
from tensorflow.keras.layers import Lambda, RepeatVector, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.layers import Add, Subtract, add, subtract
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#devices list
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import CSVLogger #added logger file
csv_logger = CSVLogger('log.csv', append=True, separator=';')

#import pickle
import pickle
MODEL_SAVE_NAME = 0
MODEL_SAVE_WEIGHTS = 1
import warnings
from tensorflow.keras.callbacks import Callback
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"#",1,2,3"  # specify which GPU(s) to be used
import json
from pprint import pprint
import scipy.io as io
from PIL import Image
import pandas
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
import random

l_rate = 5e-4 #lr value is here changed to 1e-3
CSV_path = '/afs/crc.nd.edu/user/g/ganantha/Test/Balayya/training_code/sample_data/'

#modifications  for the random seed in diff places
np.random.seed(113) #same seed like in pytorch #numpy
seed_val = 1
random.seed(seed_val) #rand seed is always 1 #random generator
from tensorflow import set_random_seed #tf seed setting
set_random_seed(seed_val) #seed is 1 for the tensorflow
os.environ['PYTHONHASHSEED']=str(seed_val) #seed for os files of python hashseed: 
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, num_exps=100, batch_size=4, dim=(512,512), n_channels=1, shuffle=True, train = False, validation = False, test = False, test_set =1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.num_exps = num_exps
        self.train = train
        self.validation = validation
        #if self.train == False and self.validation == False:
        self.test = test
        #print('>>>>>>',self.test)
        self.test_set = test_set
        self.captures = 50
        self.on_epoch_end()
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_exps / self.batch_size))

    def __getitem__(self, index): #index = batch num
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = indexes#[self.list_IDs[k] for k in indexes]
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_exps)
        if self.train == True:
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty(((self.batch_size)*4, *self.dim, self.n_channels)) #total  batch size here includes slices
        Y = np.empty(((self.batch_size)*4, *self.dim, self.n_channels))
        if self.train == True:
            df = pandas.read_csv(CSV_path + 'samples_train.csv')
        if self.validation == True:
            df = pandas.read_csv(CSV_path + 'samples_test.csv')
        if self.test == True:
            if self.test_set==1:
                df2 = pandas.read_csv(CSV_path + 'test_mix_samples_raw.csv')
            elif self.test_set==2:
                df2 = pandas.read_csv(CSV_path + 'test_mix_samples_avg2.csv')
            elif self.test_set==3:
                df2 = pandas.read_csv(CSV_path + 'test_mix_samples_avg4.csv')
            elif self.test_set==4:
                df2 = pandas.read_csv(CSV_path + 'test_mix_samples_avg8.csv')
            elif self.test_set==5:
                df2 = pandas.read_csv(CSV_path + 'test_mix_samples_avg16.csv')
            else:
                df2 = pandas.read_csv(CSV_path + 'test_mix_samples_all_grouped.csv')
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #print('index is here in data gen is >>>>>',ID)
            if self.train == True or self.validation == True:
                files1 = df['Images'][ID*self.captures:(ID+1)*self.captures]
                indx1 = range(ID*self.captures,(ID+1)*self.captures)
                indx1 = np.array(indx1) #to array
                #take input from id*50 to (id+1)*50, shuffle the data and take input0 and input1 as input and target
                if self.shuffle == True:
                    np.random.shuffle(indx1) #shuffle array
                fx1 = indx1[0] #input 
                fx2 = indx1[1] #target
                
                img = load_img(df['Images'][fx1], color_mode = "grayscale")
                label = load_img(df['Images'][fx2] , color_mode = "grayscale")

            if self.test == True: #test no shuffle
                img = load_img(df2['Images'][ID], color_mode = "grayscale")
                label = load_img(df2['Labels'][ID] , color_mode = "grayscale")
                img1 = img_to_array(img)
                label1 = img_to_array(label)
            
            x_img = img_to_array(img)
            y_label = img_to_array(label)
            
            x_img = resize(x_img, (512, 512, 1), mode='constant', preserve_range=True) #actual  image and label
            y_label = resize(y_label, (512, 512, 1), mode='constant', preserve_range=True)#actual  image and label
            
            imx1 = np.array([x_img[x:x+self.dim[0],y:y+self.dim[1]] for x in range(0,x_img.shape[0],self.dim[0]) for y in range(0,x_img.shape[1],self.dim[1])])
            lbx1 = np.array([y_label[x:x+self.dim[0],y:y+self.dim[1]] for x in range(0,y_label.shape[0],self.dim[0]) for y in range(0,y_label.shape[1],self.dim[1])])
            
            X[i*4:(i+1)*4, ..., 0] = imx1.squeeze() / 255 - 0.5
            Y[i*4:(i+1)*4, ..., 0] = lbx1.squeeze() / 255 - 0.5

        return X, Y
        
def save_weight_light(model, filename):
	layers = model.layers
	pickle_list = []
	for layerId in range(len(layers)):
		weigths = layers[layerId].get_weights()
		pickle_list.append([layers[layerId].name, weigths])

	with open(filename, 'wb') as f:
		pickle.dump(pickle_list, f, -1)

def load_weight_light(model, filename):
	layers = model.layers
	with open(filename, 'rb') as f:
		pickle_list = pickle.load(f)

	for layerId in range(len(layers)):
		assert(layers[layerId].name == pickle_list[layerId][MODEL_SAVE_NAME])
		layers[layerId].set_weights(pickle_list[layerId][MODEL_SAVE_WEIGHTS])

class ModelCheckpointLight(Callback):
    def __init__(self, filepath, monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        super(ModelCheckpointLight, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpointLight mode %s is unknown, ''fallback to auto mode.' % (mode), RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        if mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        if mode == 'auto':
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, ''skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'' saving model to %s'
                                    % (epoch + 1, self.monitor, self.best, current, filepath))
                        self.best = current
                        save_weight_light(self.model, filepath)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                    (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                save_weight_light(self.model, filepath)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
get_gpus =  get_available_gpus()
print('>>>>>>>> Available GPU devices list is here: >>>>>>>>>>',get_gpus)
   
#with open("args.txt", 'w') as args_file:
#    json.dump(get_gpus, args_file, indent=4)


#multiple GPUS training
#from multi_gpu_utils import multi_gpu_models ##so much dependance -> use latest TF version here
from tensorflow.keras.utils import multi_gpu_model #-> use for next keras release version
# or use this pip install --user git+git://github.com/fchollet/keras.git --upgrade in job file/in terminal that upgrades the keras version
import multiprocessing #for CPU count

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.backend import set_session
import tensorflow.keras.backend as Kb
#Kb._get_available_gpus()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #either true or per fraction need to be set
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True

t1= time.time()
for d in ['/device:CPU:0','/device:GPU:0']:#, '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']:
    with tf.device(d):
        path_save_model = '/afs/crc.nd.edu/user/g/ganantha/Test/Balayya/training_code/Unet_bn/unet_bn_lr3/Saved_model/'
        batch_size = 4
        im_height = 256 #M
        im_width = 256  #N
        num_exps_train = 1140 #total train examples = 57000
        num_exps_validation = 60  #total test examples = 3000
        num_exps_test = 48 #test dataset with raw data
        
        params_train = {'dim': (im_height,im_width),'num_exps':num_exps_train,'batch_size': batch_size,'n_channels': 1,'shuffle': True, 'train': True, 'validation': False, 'test': False,'test_set':1}
        params_validation= {'dim': (im_height,im_width),'num_exps':num_exps_validation,'batch_size': batch_size,'n_channels': 1,'shuffle': False,'train': False, 'validation': True, 'test': False,'test_set':1}
        #test_set =1 raw data, 2 -> avg2 , 3-> avg4, 4 -> avg8, 5-> avg16, 6-> all together noise levels (1,2,4,8,16)
        params_test = {'dim': (im_height,im_width),'num_exps':num_exps_test,'batch_size': 1,'n_channels': 1,'shuffle': False, 'train': False,  'validation': False, 'test': True,'test_set':1}

        training_generator = DataGenerator( **params_train)
        validation_generator = DataGenerator(**params_validation)
        test_generator = DataGenerator(**params_test)

init_ortho = tf.keras.initializers.Orthogonal() #orthogonal weights initialization

def get_unet_bn(input_img):
    # contracting path
    #256
    xc0 = Conv2D(filters=48, kernel_size=(3,3), kernel_initializer=init_ortho,padding="same",use_bias=False)(input_img)
    xa0 = LeakyReLU(alpha=0.1)(xc0)
    xc1 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer=init_ortho,padding="same",use_bias=False)(xa0)
    xcbn1 = BatchNormalization()(xc1)
    xa1 = LeakyReLU(alpha=0.1)(xcbn1)
    xp1 = MaxPooling2D((2, 2)) (xa1)
    
    #128
    xc2 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer=init_ortho,padding="same",use_bias=False)(xp1)
    xcbn2 = BatchNormalization()(xc2)
    xa2 = LeakyReLU(alpha=0.1)(xcbn2)
    xp2 = MaxPooling2D((2, 2)) (xa2)
    
    #64
    xc3 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer=init_ortho,padding="same",use_bias=False)(xp2)
    xcbn3 = BatchNormalization()(xc3)
    xa3 = LeakyReLU(alpha=0.1)(xcbn3)
    xp3 = MaxPooling2D((2, 2)) (xa3)

    #32
    xc4 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer=init_ortho,padding="same",use_bias=False)(xp3)
    xcbn4 = BatchNormalization()(xc4)
    xa4 = LeakyReLU(alpha=0.1)(xcbn4)
    xp4 = MaxPooling2D((2, 2)) (xa4)
    
    #16
    xc5 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer=init_ortho, padding="same",use_bias=False)(xp4)
    xcbn5 = BatchNormalization()(xc5)
    xa5 = LeakyReLU(alpha=0.1)(xcbn5)
    xp5 = MaxPooling2D((2, 2)) (xa5)
    
    #8
    xc6 = Conv2D(filters=48, kernel_size=(3, 3), kernel_initializer=init_ortho, padding="same",use_bias=False)(xp5)
    xcbn6 = BatchNormalization()(xc6)
    xa6 = LeakyReLU(alpha=0.1)(xcbn6)
    xup5 = UpSampling2D((2, 2))(xa6)
    
    #16
    cat1 = concatenate([xup5, xp4])
    xdc5a = Conv2D(96, (3, 3), kernel_initializer=init_ortho, padding='same',use_bias=False) (cat1)
    xdbn5a = BatchNormalization()(xdc5a)
    xad5a = LeakyReLU(alpha=0.1)(xdbn5a)
    xdc5b =  Conv2D(96, (3, 3), kernel_initializer=init_ortho, padding='same',use_bias=False) (xad5a)
    xdbn5b = BatchNormalization()(xdc5b)
    xad5b = LeakyReLU(alpha=0.1)(xdbn5b)
    xup4 = UpSampling2D((2, 2))(xad5b)
    
    #32
    cat2 = concatenate([xup4, xp3])
    xdc4a =  Conv2D(96, (3, 3), kernel_initializer=init_ortho, padding='same',use_bias=False) (cat2) #op channels (96) not input channels (144)
    xdbn4a = BatchNormalization()(xdc4a)
    xad4a = LeakyReLU(alpha=0.1)(xdbn4a)
    xdc4b =  Conv2D(96, (3, 3), kernel_initializer=init_ortho, padding='same',use_bias=False) (xad4a)
    xdbn4b = BatchNormalization()(xdc4b)
    xad4b = LeakyReLU(alpha=0.1)(xdbn4b)
    xup3 = UpSampling2D((2, 2))(xad4b)
    
    #64
    cat3 = concatenate([xup3, xp2])
    xdc3a =  Conv2D(96, (3, 3), kernel_initializer=init_ortho, padding='same',use_bias=False) (cat3) #op channels (96) not input channels (144)
    xdbn3a = BatchNormalization()(xdc3a)
    xad3a = LeakyReLU(alpha=0.1)(xdbn3a)
    xdc3b =  Conv2D(96, (3, 3), kernel_initializer=init_ortho, padding='same',use_bias=False) (xad3a)
    xdbn3b = BatchNormalization()(xdc3b)
    xad3b = LeakyReLU(alpha=0.1)(xdbn3b)
    xup2 = UpSampling2D((2, 2))(xad3b)
    
    #128
    cat4 = concatenate([xup2, xp1])
    xdc2a =  Conv2D(96, (3, 3), kernel_initializer=init_ortho, padding='same',use_bias=False) (cat4) #op channels (96) not input channels (144)
    xdbn2a = BatchNormalization()(xdc2a)
    xad2a = LeakyReLU(alpha=0.1)(xdbn2a)
    xdc2b =  Conv2D(96, (3, 3), kernel_initializer=init_ortho, padding='same',use_bias=False) (xad2a)
    xdbn2b = BatchNormalization()(xdc2b)
    xad2b = LeakyReLU(alpha=0.1)(xdbn2b)
    xup1 = UpSampling2D((2, 2))(xad2b)
    
    #256
    cat5 = concatenate([xup1, input_img])
    xdc1a =  Conv2D(64, (3, 3), kernel_initializer=init_ortho, padding='same',use_bias=False) (cat5) #op channels (64) not input channels 
    xdbn1a = BatchNormalization()(xdc1a)
    xad1a = LeakyReLU(alpha=0.1)(xdbn1a)
    xdc1b =  Conv2D(32, (3, 3), kernel_initializer=init_ortho, padding='same',use_bias=False) (xad1a) #input 64, output=32
    xdbn1b = BatchNormalization()(xdc1b)
    xad1b = LeakyReLU(alpha=0.1)(xdbn1b)
    xdc1c =  Conv2D(1, (3, 3), kernel_initializer=init_ortho, padding='same',use_bias=False) (xad1b) #input 32, output=1
    
    outputs = Activation("tanh")(xdc1c)

    #model = Model(inputs=[input_img], outputs=[outputs])
    #return model
    return outputs


#lr updates : 0106
def annealing_linear(start, end, pct):
    return start + pct * (end-start)

def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out

epochs = 200 #this epochs for the lr function adjustment
def scheduler(epoch, lr):
    lr_low = l_rate/10
    lr_min = lr_low/1e4
    pct = epoch/epochs
    pct_start = 0.3
    print('>>>> pct here <<<<<< ',pct)
    if pct <= pct_start:
        lr_mod = annealing_linear(lr_low, l_rate, pct/pct_start)
    else:
        lr_mod = annealing_cos(l_rate, lr_min, (pct-pct_start)/(1-pct_start))
    print('>>>>>> updated lr is here', lr_mod)
    return lr_mod

scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
#callbacks = [EarlyStopping(patience=10, verbose=1),ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    #ModelCheckpointLight('model-dncnn_multiple_gpus.h5', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)] 
#old reduced lr call back
# callbacks = [ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=3, min_lr=0.00000001, verbose=1),
#         ModelCheckpointLight('model_dncnn_multiple_gpus_lr_change.h5', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1),csv_logger] #removed early stop
#new stepwise reduce lr call back
callbacks = [scheduler_callback, csv_logger,
        ModelCheckpointLight('model_multiple_gpus_unet_bn_lr3.h5', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)] #removed early stop
    #always save old: save_best_only=True

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from tensorflow.keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def train(epochs):
    
    input_img = Input((256, 256, 1), name='img')
    output_img = Input((256, 256, 1), name='label')
    outputs = get_unet_bn(input_img)
    model = Model(inputs=[input_img],outputs=[outputs])
    
    if len(get_gpus) ==0:
        print(" >>>>>> training with 0 GPU >>>>>> ")
        model.build(input_img)
    elif len(get_gpus) ==1:
        print(" >>>>>> training with 1 GPU >>>>>> ")
        model.build(input_img)
    else:
        print(" >>>>>>>> training with {} GPUs >>>>>>".format(len(get_gpus)))
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/device:CPU:0"):
            # initialize the model
            model.build(input_img)
            #print('<<<< Model is here <<<<<',model)
            # make the model parallel
        model = multi_gpu_model(model, gpus=len(get_gpus))
        
    #with open("args.txt", 'w') as args_file:
    #    json.dump(len(get_gpus), args_file, indent=4)
    adam = Adam(lr=l_rate,beta_1=0.9,beta_2=0.999,epsilon=1e-08,amsgrad=False) #adam optimizer   
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=["mse"])
    #print('<<<< Model summary is here <<<<<',model.summary())
    #gbytes_model = get_model_memory_usage (batch_size=batch_size, model=model)
    #print('>>>>>>> Memory used in the Model: (GBytes) >>>>>>>>>>',gbytes_model)
    #with open("args.txt", 'w') as args_file:
    #    json.dump(gbytes_model, args_file, indent=4)
        
    #added validation loss as well
    results = model.fit_generator(generator=training_generator, validation_data=validation_generator, steps_per_epoch=num_exps_train//batch_size, validation_steps=num_exps_validation//batch_size, epochs=epochs, callbacks=callbacks, use_multiprocessing=True, workers=4, max_queue_size=10)
    #old one
    #results = model.fit_generator(generator=training_generator, steps_per_epoch=num_exps_train//batch_size, epochs=epochs, callbacks=callbacks, use_multiprocessing=True, workers=4, max_queue_size=10) #multiprocessing.cpu_count() - 1)#6)
    
    test_results = model.predict_generator(generator=test_generator, steps=None, use_multiprocessing=False, workers=4, max_queue_size=10)
    import scipy.io as io
    print('result type ', test_results.shape)
    io.savemat('Estimated_result_unet_bn_lr3.mat', dict([('Estimated_result_unet_bn_lr3',test_results)]))
    #print('>>>> test_results2 are here >>>> ', test_results.shape)
    test_results_final_epoch = model.evaluate_generator(generator=test_generator, steps=None, max_queue_size=10, workers=4, use_multiprocessing=False)
    
    signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={'input_img': model.inputs[0]}, outputs={'outputs': model.outputs[0]})  #this is a list here  
    timestr = time.strftime("%Y%m%d_%H%M%S")                                                                     
    builder = tf.saved_model.builder.SavedModelBuilder(path_save_model + timestr)                                                                    
    builder.add_meta_graph_and_variables( sess=Kb.get_session(), tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:signature})
    builder.save()
    return results, test_results_final_epoch

      
if __name__ == "__main__":
    Epochs =  200
    
    results, test_results_final_epoch = train(epochs=Epochs)
    #save results
    loss_mse = results.history["loss"]
    val_loss_mse = results.history["val_loss"]
    #rmse calculation:
    loss_rmse = np.sqrt(np.array(loss_mse)/16) #batch_size*multiplier #4*4
    print('>>>> test_results (loss and MSE on final epoch)are here >>>> ', test_results_final_epoch)
    
    #plot
    plt.figure(figsize=(8, 8))
    plt.title("MSE Loss")
    plt.plot(results.history["loss"],'b*--', label="loss")
    plt.plot(results.history["val_loss"],'r-d', label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.title('MSE Loss vs Epochs')
    plt.legend(loc='best', frameon=False)
    timestr_plt = time.strftime("%Y%m%d_%H%M%S")   
    path_for_images = '/afs/crc.nd.edu/user/g/ganantha/Test/Balayya/training_code/Unet_bn/unet_bn_lr3/plots/'
    path_for_results = '/afs/crc.nd.edu/user/g/ganantha/Test/Balayya/training_code/Unet_bn/unet_bn_lr3/results/'
    np.savetxt(path_for_results+ 'train_loss_unet_bn_denoising_'+ timestr_plt +'.txt', np.array(loss_mse))
    np.savetxt(path_for_results+ 'validation_loss_unet_bn_denoising_'+ timestr_plt +'.txt', np.array(val_loss_mse))
    np.savetxt(path_for_results+ 'rmse_loss_unet_bn_denoising_'+ timestr_plt +'.txt', np.array(loss_rmse)) #rmse
    plt.savefig(path_for_images + 'Unet_bn_all_images_'+ timestr_plt +'.png')

    t2 = time.time()

    print('Execution time is: \n',t2-t1)
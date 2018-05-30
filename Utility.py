# Hidden 2 domains no constrained
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, argparse, glob, time, six, shutil 

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 

# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation
import malis

import PIL
from PIL import Image
# Tensorpack toolbox
import tensorpack.tfutils.symbolic_functions as symbf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.tfutils import optimizer, gradproc
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_tensor_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils import logger

# Tensorflow 
import tensorflow as tf

# Tensorlayer
from tensorlayer.cost import binary_cross_entropy, absolute_difference_error, dice_coe

# Sklearn
from sklearn.metrics.cluster import adjusted_rand_score

# Augmentor
import Augmentor
###############################################################################
EPOCH_SIZE = 500
NB_FILTERS = 32   # channel size

DIMX  = 640
DIMY  = 640
DIMZ  = 1
DIMC  = 1


###############################################################################
def INReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.leaky_relu(x, name=name)
    
def BNLReLU(x, name=None):
    x = BatchNorm('bn', x)
    return tf.nn.leaky_relu(x, name=name)

###############################################################################
# Utility function for scaling 
def tf_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
    with tf.variable_scope(name):

        return (x / maxVal - 0.5) * 2.0
        # x = tf.divide(x, tf.convert_to_tensor(maxVal))
        # x = tf.subtract(x, tf.convert_to_tensor(0.5))
        # x = tf.multiply(x, tf.convert_to_tensor(2.0))
        # return x
###############################################################################
def tf_2imag(x, maxVal = 255.0, name='ToRangeImag'):
    with tf.variable_scope(name):

        return (x / 2.0 + 0.5) * maxVal
        # x = tf.divide(x, tf.convert_to_tensor(2.0))
        # x = tf.add(x, tf.convert_to_tensor(0.5))
        # x = tf.multiply(x, tf.convert_to_tensor(maxVal))
        # return x

# Utility function for scaling 
def np_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
    return (x / maxVal - 0.5) * 2.0
###############################################################################
def np_2imag(x, maxVal = 255.0, name='ToRangeImag'):
    return (x / 2.0 + 0.5) * maxVal

###############################################################################
# FusionNet
@layer_register(log_shape=True)
def residual(x, chan, first=False, kernel_shape=3):
    with argscope([Conv2D], nl=INLReLU, stride=1, kernel_shape=kernel_shape):
        inputs = x
        # x = tf.pad(x, name='pad1', mode='REFLECT', paddings=[[0,0], [1*(kernel_shape//2),1*(kernel_shape//2)], [1*(kernel_shape//2),1*(kernel_shape//2)], [0,0]])
        # x = Conv2D('conv1', x, chan, padding='VALID', dilation_rate=1)
        # x = tf.pad(x, name='pad2', mode='REFLECT', paddings=[[0,0], [2*(kernel_shape//2),2*(kernel_shape//2)], [2*(kernel_shape//2),2*(kernel_shape//2)], [0,0]])
        # x = Conv2D('conv2', x, chan, padding='VALID', dilation_rate=2)
        # x = tf.pad(x, name='pad3', mode='REFLECT', paddings=[[0,0], [4*(kernel_shape//2),4*(kernel_shape//2)], [4*(kernel_shape//2),4*(kernel_shape//2)], [0,0]])
        # x = Conv2D('conv3', x, chan, padding='VALID', dilation_rate=4)             
        # x = tf.pad(x, name='pad4', mode='REFLECT', paddings=[[0,0], [8*(kernel_shape//2),8*(kernel_shape//2)], [8*(kernel_shape//2),8*(kernel_shape//2)], [0,0]])
        # x = Conv2D('conv4', x, chan, padding='VALID', dilation_rate=8)
        # x = tf.pad(x, name='pad0', mode='REFLECT', paddings=[[0,0], [kernel_shape//2,kernel_shape//2], [kernel_shape//2,kernel_shape//2], [0,0]])
        # x = Conv2D('conv0', x, chan, padding='VALID', nl=tf.identity)
        x = tf.pad(x, name='pad1', mode='REFLECT', paddings=[[0,0], [1*(kernel_shape//2),1*(kernel_shape//2)], [1*(kernel_shape//2),1*(kernel_shape//2)], [0,0]])
        x = Conv2D('conv1', x, chan, padding='VALID', dilation_rate=1)
        x = tf.pad(x, name='pad2', mode='REFLECT', paddings=[[0,0], [2*(kernel_shape//2),2*(kernel_shape//2)], [2*(kernel_shape//2),2*(kernel_shape//2)], [0,0]])
        x = Conv2D('conv2', x, chan, padding='VALID', dilation_rate=2)
        x = tf.pad(x, name='pad3', mode='REFLECT', paddings=[[0,0], [4*(kernel_shape//2),4*(kernel_shape//2)], [4*(kernel_shape//2),4*(kernel_shape//2)], [0,0]])
        x = Conv2D('conv3', x, chan, padding='VALID', dilation_rate=4)             
        # x = tf.pad(x, name='pad4', mode='REFLECT', paddings=[[0,0], [8*(kernel_shape//2),8*(kernel_shape//2)], [8*(kernel_shape//2),8*(kernel_shape//2)], [0,0]])
        # x = Conv2D('conv4', x, chan, padding='VALID', dilation_rate=8) 
        x = InstanceNorm('inorm', x) + inputs
        return x

###############################################################################
@layer_register(log_shape=True)
def residual_enc(x, chan, first=False, kernel_shape=3):
    with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=kernel_shape):
        
        x = tf.pad(x, name='pad_i', mode='REFLECT', paddings=[[0,0], [kernel_shape//2,kernel_shape//2], [kernel_shape//2,kernel_shape//2], [0,0]])
        x = Conv2D('conv_i', x, chan, stride=2) 
        x = residual('res_', x, chan, first=True)
        x = tf.pad(x, name='pad_o', mode='REFLECT', paddings=[[0,0], [kernel_shape//2,kernel_shape//2], [kernel_shape//2,kernel_shape//2], [0,0]])
        x = Conv2D('conv_o', x, chan, stride=1) 
        #x = Dropout('dr', x, 0.5)
        return x

###############################################################################
@layer_register(log_shape=True)
def Subpix2D(inputs, chan, scale=2, stride=1, kernel_shape=3):
    with argscope([Conv2D], nl=INLReLU, stride=stride, kernel_shape=kernel_shape):
        padded = tf.pad(inputs, paddings=[[0,0], [kernel_shape//2,kernel_shape//2], [kernel_shape//2,kernel_shape//2], [0,0]], 
            mode='REFLECT', name='padded')
        result = Conv2D('conv0', padded, chan* scale**2, padding='VALID')
        old_shape = inputs.get_shape().as_list()
        if scale>1:
            result = tf.depth_to_space(result, scale, name='depth2space', data_format='NHWC')
        return result

@layer_register(log_shape=True)
def residual_dec(x, chan, first=False, kernel_shape=3):
    with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=kernel_shape):
        x = Deconv2D('deconv_i', x, chan, stride=1) 
        x = residual('res2_', x, chan, first=True)
        x1 = Deconv2D('deconv_o', x, chan, stride=2) 
        x2 = BilinearUpSample('upsample', x, 2)
        # x3 = Subpix2D('subpix', x, chan, scale=2)
        return (x1+x2)/2.0
        # return x2

###############################################################################
@auto_reuse_variable_scope
def arch_fusionnet_2d(img, last_dim=1, nl=INLReLU, nb_filters=32):
    assert img is not None
    with argscope([Conv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
            e0 = residual_enc('e0', img, nb_filters*1)
            e1 = residual_enc('e1',  e0, nb_filters*2)
            e2 = residual_enc('e2',  e1, nb_filters*4)

            e3 = residual_enc('e3',  e2, nb_filters*8)
            # e3 = Dropout('dr', e3, 0.5)

            d3 = residual_dec('d3',    e3, nb_filters*4)
            d2 = residual_dec('d2', d3+e2, nb_filters*2)
            d1 = residual_dec('d1', d2+e1, nb_filters*1)
            d0 = residual_dec('d0', d1+e0, nb_filters*1) 
            dp = tf.pad( d0, name='pad_o', mode='REFLECT', paddings=[[0,0], [3//2,3//2], [3//2,3//2], [0,0]])
            dd = Conv2D('convlast', dp, last_dim, kernel_shape=3, stride=1, padding='VALID', nl=nl, use_bias=True) 
            return dd


@auto_reuse_variable_scope
def arch_fusionnet_encoder_2d(img, feats=[None, None, None], last_dim=1, nl=INLReLU, nb_filters=32):
    assert img is not None
    with argscope([Conv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
            e0 = residual_enc('e0', img, nb_filters*1)
            e1 = residual_enc('e1',  e0, nb_filters*2)
            e2 = residual_enc('e2',  e1, nb_filters*4)

            e3 = residual_enc('e3',  e2, nb_filters*8)
            # e3 = Dropout('dr', e3, 0.5)
            return e3, [e2, e1, e0]
            # d3 = residual_dec('d3',    e3, nb_filters*4)
            # d2 = residual_dec('d2', d3+e2, nb_filters*2)
            # d1 = residual_dec('d1', d2+e1, nb_filters*1)
            # d0 = residual_dec('d0', d1+e0, nb_filters*1) 
            # dp = tf.pad( d0, name='pad_o', mode='REFLECT', paddings=[[0,0], [3//2,3//2], [3//2,3//2], [0,0]])
            # dd = Conv2D('convlast', dp, last_dim, kernel_shape=3, stride=1, padding='VALID', nl=nl, use_bias=True) 
            # return dd

@auto_reuse_variable_scope
def arch_fusionnet_decoder_2d(img, feats=[None, None, None], last_dim=1, nl=tf.nn.tanh, nb_filters=32):
    assert img is not None
    with argscope([Conv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='VALID'):
        with argscope([Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
            e3 = img
            e2, e1, e0 = feats 
            d4 = e3 
            d3 = residual_dec('d3', d4, nb_filters*4)
            d3 = d3+e2 if e2 is not None else d3 
            d2 = residual_dec('d2', d3, nb_filters*2)
            d2 = d2+e1 if e1 is not None else d2 
            d1 = residual_dec('d1', d2, nb_filters*1)
            d1 = d1+e0 if e0 is not None else d1  
            d0 = residual_dec('d0', d1, nb_filters*1) 
            dp = tf.pad( d0, name='pad_o', mode='REFLECT', paddings=[[0,0], [3//2,3//2], [3//2,3//2], [0,0]])
            dd = Conv2D('convlast', dp, last_dim, kernel_shape=3, stride=1, padding='VALID', nl=nl, use_bias=True) 
            return dd, [d1, d2, d3]










def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

###############################################################################
class ImageDataFlow(RNGDataFlow):
    def __init__(self, imageDir, labelDir, noiseDir, size, dtype='float32', isTrain=False, isValid=False, isTest=False):
        self.dtype      = dtype
        self.imageDir   = imageDir
        self.labelDir   = labelDir
        self.noiseDir   = noiseDir
        self._size      = size
        self.isTrain    = isTrain
        self.isValid    = isValid

        
        images = natsorted (glob.glob(self.imageDir + '/*.tif'))
        labels = natsorted (glob.glob(self.labelDir + '/*.tif'))
        noises = natsorted (glob.glob(self.noiseDir + '/*.tif'))
        self.images = []
        self.labels = []
        self.noises = []
        self.data_seed = time_seed ()
        self.data_rand = np.random.RandomState(self.data_seed)
        self.rng = np.random.RandomState(999)
        for i in range (len (images)):
            image = images[i]
            self.images.append (skimage.io.imread (image))
        for i in range (len (labels)):
            label = labels[i]
            self.labels.append (skimage.io.imread (label))
        for i in range (len (noises)):
            noise = noises[i]
            self.noises.append (skimage.io.imread (noise))

        # print(self.imageDir)
        # print(self.labelDir)
        # print(self.noiseDir)
        
        # print(self.images)
        # print(self.labels)
        # print(self.noises)

        # print(self.images[0].shape)
        # print(self.labels[0].shape)
        # print(self.noises[0].shape)

        #self._size = 0
        #for i in range (len (self.images)):
        #   self._size += self.images[i].shape[0] * self.images[i].shape[1] * self.images[i].shape[2] \
        #           / (input_shape[0] * input_shape[1] * input_shape[2])

    def size(self):
        return self._size

    ###############################################################################
    # def AugmentPair(self, src_image, src_label, pipeline, seed=None, verbose=False):
    #   np.random.seed(seed) if seed else np.random.seed(2015)
        
    #   # Create the result
    #   aug_image = np.zeros_like(src_image)
    #   aug_label = np.zeros_like(src_label)
    #   # print(src_image.shape, src_label.shape, aug_image.shape, aug_label.shape) if verbose else ''
    #   for z in range(src_image.shape[0]):
    #       #Image and numpy has different matrix order
    #       pipeline.set_seed(seed)
    #       aug_image[z,...] = pipeline._execute_with_array(src_image[z,...]) 
    #       pipeline.set_seed(seed)
    #       aug_label[z,...] = pipeline._execute_with_array(src_label[z,...])        
    #   return aug_image, aug_label


    ###############################################################################
    def get_data(self):
        for k in range(self._size):
            #
            # Pick randomly a tuple of training instance
            #
            rand_image = self.data_rand.randint(0, len(self.images))
            rand_label = self.data_rand.randint(0, len(self.labels))
            rand_noise = self.data_rand.randint(0, len(self.noises))
            image = self.images[rand_image].copy ()
            label = self.labels[rand_label].copy ()
            noise = self.noises[rand_label].copy ()



            

            seed = time_seed () #self.rng.randint(0, 20152015)

            if self.isTrain:

                p = Augmentor.Pipeline()
                p.crop_by_size(probability=1, width=DIMX, height=DIMY, centre=False)
                p.rotate_random_90(probability=0.75, resample_filter=Image.NEAREST)
                p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20, resample_filter=Image.NEAREST)
                #p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=10)
                p.zoom_random(probability=0.5, percentage_area=0.8)
                p.flip_random(probability=0.75)

                image = p._execute_with_array(image) 
                label = p._execute_with_array(label) 
                noise = p._execute_with_array(noise) 
            else:
                p = Augmentor.Pipeline()
                p.crop_by_size(probability=1, width=DIMX, height=DIMY, centre=True)
                image = p._execute_with_array(image) 
                label = p._execute_with_array(label) 
                noise = p._execute_with_array(noise) 
                # pass
                

            #Expand dim to make single channel
            image = np.expand_dims(image, axis=-1)
            label = np.expand_dims(label, axis=-1)
            noise = np.expand_dims(noise, axis=-1)

            image = np.expand_dims(image, axis=0)
            label = np.expand_dims(label, axis=0)
            noise = np.expand_dims(noise, axis=0)

            yield [image.astype(np.float32), 
                   label.astype(np.float32), 
                   noise.astype(np.float32), 
                   ] 

    
###############################################################################
class ClipCallback(Callback):
    def _setup_graph(self):
        vars = tf.trainable_variables()
        ops = []
        for v in vars:
            n = v.op.name
            if not n.startswith('discrim/'):
                continue
            logger.info("Clip {}".format(n))
            ops.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))
        self._op = tf.group(*ops, name='clip')

    def _trigger_step(self):
        self._op.run()




###############################################################################
def sample(dataDir, model_path, model=None, prefix='.'):
    print("Starting...")
    
    print("Ending...")

    imageFiles = glob.glob(os.path.join(dataDir, 'image/*.tif'))
    labelFiles = glob.glob(os.path.join(dataDir, 'label/*.tif'))
    noiseFiles = glob.glob(os.path.join(dataDir, 'noise/*.tif'))
    
    # Load the model 
    predict_func = OfflinePredictor(PredictConfig(
        model=model,
        session_init=get_model_loader(model_path),
        input_names=['image', 'label', 'noise'],
        output_names=['viz']))

    for k in range(len(imageFiles)):
        image = skimage.io.imread(imageFiles[k])
        label = skimage.io.imread(labelFiles[k])
        noise = skimage.io.imread(noiseFiles[k])
        # label = image.copy()
        # noise = image.copy()
        # group the input to form one datapoint
        instance = []
        instance.append(image)
        instance.append(label)
        instance.append(noise)
        instance = np.array(instance).astype(np.float32)
        import dask.array as da 

        #print(instance)
        da_instance = da.from_array(instance, chunks=(instance.shape[0], 512, 512)) 
        #print(da_instance)
        gp_instance = da.ghost.ghost(da_instance, depth={0:0, 1:64, 2:64}, boundary = {0:0, 1:'reflect', 2:'reflect'})
        def func(block, predict_func):
            #print(block.shape)
            bl_image = block[0,...]
            bl_label = block[1,...]
            bl_noise = block[2,...]


            bl_image = np.expand_dims(bl_image, axis=0)
            bl_label = np.expand_dims(bl_label, axis=0)
            bl_noise = np.expand_dims(bl_noise, axis=0)
            
            bl_image = np.expand_dims(bl_image, axis=-1)
            bl_label = np.expand_dims(bl_label, axis=-1)
            bl_noise = np.expand_dims(bl_noise, axis=-1)
            

            pred = predict_func(bl_image, 
                                bl_label, 
                                bl_noise 
                                )

            d = pred[0] # First output
            d = np.squeeze(d)
            # print(d.shape)
            # Crop to the clean version
            # d = d[640:1280, 640:1280]
            # d = np.expand_dims(d, axis=0) # concatenation
            
            # ppc_clean  = d[0640:1280, (0640+0640*0):(1280+0640*0)]
            # ppc_noise  = d[0640:1280, (0640+0640*1):(1280+0640*1)]
            # wfly_clean = d[0640:1280, (0640+0640*3):(1280+0640*3)]
            # wfly_noise = d[0640:1280, (0640+0640*4):(1280+0640*4)]
            ppc_clean  = d[640:1280,  640:1280]
            ppc_noise  = d[640:1280, 1280:1920]
            wfly_bline = d[640:1280, 1920:2560]
            wfly_clean = d[640:1280, 2560:3200]
            wfly_noise = d[640:1280, 3200:3840]

            # print(ppc_clean.shape)
            # print(ppc_noise.shape)
            # print(wfly_clean.shape)
            # print(wfly_noise.shape)

            ppc_clean  = np.expand_dims(ppc_clean, axis=0)
            ppc_noise  = np.expand_dims(ppc_noise, axis=0)
            wfly_bline = np.expand_dims(wfly_bline, axis=0)
            wfly_clean = np.expand_dims(wfly_clean, axis=0)
            wfly_noise = np.expand_dims(wfly_noise, axis=0)

            # print(ppc_clean.shape)
            # print(ppc_noise.shape)
            # print(wfly_clean.shape)
            # print(wfly_noise.shape)

            group = np.concatenate([ppc_clean, ppc_noise, wfly_bline, wfly_clean, wfly_noise], axis=0)
            return group
            
        gp_deployment = gp_instance.map_blocks(func, predict_func, dtype=np.float32)
        gp_deployment = da.ghost.trim_internal(gp_deployment, {0:0, 1:64, 2:64})

        np_deployment = np.array(gp_deployment).astype(np.uint8)
        skimage.io.imsave(os.path.join(prefix, "ppc_clean_{}.png".format(k+1)), np.squeeze(np_deployment[0,...]))
        skimage.io.imsave(os.path.join(prefix, "ppc_noise_{}.jpg".format(k+1)), np.squeeze(np_deployment[1,...]))
        skimage.io.imsave(os.path.join(prefix, "wfly_bline_{}.pgm".format(k+1)), np.squeeze(np_deployment[2,...]))
        skimage.io.imsave(os.path.join(prefix, "wfly_clean_{}.tif".format(k+1)), np.squeeze(np_deployment[3,...]))
        skimage.io.imsave(os.path.join(prefix, "wfly_noise_{}.jpg".format(k+1)), np.squeeze(np_deployment[4,...]))
        
    
    return None

###############################################################################
def get_data(dataDir, isTrain=False, isValid=False, isTest=False):
    # Process the directories 
    if isTrain:
        num=500
    if isValid:
        num=1
    if isTest:
        num=1

    
    dset  = ImageDataFlow(os.path.join(dataDir, 'image'),
                          os.path.join(dataDir, 'label'),
                          os.path.join(dataDir, 'noise'),
                          num, 
                          isTrain=isTrain, 
                          isValid=isValid, 
                          isTest =isTest)
    dset.reset_state()
    return dset
"""
Build the dataset for train and validate.
The hyperparameters was config by a powerful tool named sacred.
The github url of sacred was "https://github.com/IDSIA/sacred"
Author thu-xd (xied15@mails.tsinghua.edu.cn)
"""

import tensorflow as tf
import os
import numpy as np
from experiment import ex

IMAGE_RESIZE_SIZE=256
IMAGE_SIZE=227

def resize_image(input_image,resize_size):
    '''Resize the input_image, so the shorter side is resize_size,
    the aspect_ratio was not changed.

    args:
        input_image:the input image with shape [height,width,channels]
        resize_size:the size of the shorter side after resize_image
        '''
    height_original=tf.shape(input_image)[0]
    width_original=tf.shape(input_image)[1]
    height_greater_than_width=tf.greater(height_original,width_original)

    height=tf.where(height_greater_than_width,tf.cast(resize_size*height_original/width_original,tf.int32),
                                                      resize_size)
    width=tf.where(height_greater_than_width,resize_size,tf.cast(resize_size*width_original/height_original,
                                                                 tf.int32))
    image=tf.image.resize_images(input_image,[height,width])
    return image

def read_image_for_train(file_name,label):
    image=tf.read_file(file_name)
    image=tf.image.decode_jpeg(image)
    image=resize_image(image,IMAGE_RESIZE_SIZE)
    
    #crop the center area of the image
    image_shape=tf.shape(image)
    h_offset=tf.cast((image_shape[0]-IMAGE_RESIZE_SIZE)/2,tf.int32)
    w_offset=tf.cast((image_shape[1]-IMAGE_RESIZE_SIZE)/2,tf.int32)
    image=tf.slice(image,[h_offset,w_offset,0],[IMAGE_RESIZE_SIZE,IMAGE_RESIZE_SIZE,3])
    
    image=tf.random_crop(image,[IMAGE_SIZE,IMAGE_SIZE,3])
    image=tf.image.random_hue(image,0.1)
    image=tf.image.random_contrast(image,lower=0.5,upper=1.5)
    image=tf.image.random_flip_left_right(image)
    image=tf.image.random_flip_up_down(image)
    image=tf.image.random_saturation(image,lower=0.5,upper=1.5)
    image=tf.image.per_image_standardization(image)
    
    image.set_shape([IMAGE_SIZE,IMAGE_SIZE,3])
    label.set_shape([])
    
    return image,label

def read_image_for_validate(file_name,label):
    image=tf.read_file(file_name)
    image=tf.image.decode_jpeg(image)    
    image=resize_image(image,IMAGE_RESIZE_SIZE)

    #crop the center area of the image
    image_shape=tf.shape(image)
    h_offset=tf.cast((image_shape[0]-IMAGE_RESIZE_SIZE)/2,tf.int32)
    w_offset=tf.cast((image_shape[1]-IMAGE_RESIZE_SIZE)/2,tf.int32)
    image=tf.slice(image,[h_offset,w_offset,0],[IMAGE_RESIZE_SIZE,IMAGE_RESIZE_SIZE,3])

    '''#crop the left-top, right-top,left-bottom,right-bottom and center patch of
    #size 224x224x3
    offset=IMAGE_RESIZE_SIZE-IMAGE_SIZE
    image_list=[]
    image_left_top=tf.slice(image,[0,0,0],[IMAGE_SIZE,IMAGE_SIZE,3])
    image_list.append(image_left_top)
    image_list.append(tf.image.flip_up_down(image_left_top))
    image_right_top=tf.slice(image,[0,offset,0],[IMAGE_SIZE,IMAGE_SIZE,3])
    image_list.append(image_right_top)
    image_list.append(tf.image.flip_up_down(image_right_top))
    image_left_bottom=tf.slice(image,[offset,0,0],[IMAGE_SIZE,IMAGE_SIZE,3])
    image_list.append(image_left_bottom)
    image_list.append(tf.image.flip_up_down(image_left_bottom))
    image_right_bottom=tf.slice(image,[offset,offset,0],[IMAGE_SIZE,IMAGE_SIZE,3])
    image_list.append(image_right_bottom)
    image_list.append(tf.image.flip_up_down(image_right_bottom))
    offset_half=int(offset/2)
    image_center=tf.slice(image,[offset_half,offset_half,0],[IMAGE_SIZE,IMAGE_SIZE,3])
    image_list.append(image_center)
    image_list.append(tf.image.flip_up_down(image_center))
    
    images=[tf.image.per_image_standardization(image) for image in image_list]
    
    images=tf.stack(image_list)
    images.set_shape([10,IMAGE_SIZE,IMAGE_SIZE,3])
    label.set_shape([])'''
    
    image=tf.slice(image,[15,15,0],[IMAGE_SIZE,IMAGE_SIZE,3])
    image.set_shape([IMAGE_SIZE,IMAGE_SIZE,3])
    label.set_shape([])
    
    return image,label

@ex.capture
def build_dataset(config):
    '''Build dataset for given image datasets.This function return two
    tf.data.Dataset:train_dataset and validate_dataset and a float
    NUM_EXAMPLES_PER_EPOCH which is the examples per epoch when training.

    args:
        config:The config parameter for building dataset'''

    config=config['data_fetch_config']

    categorys=os.listdir(config['data_dir'])
    labels=[string.lstrip('label') for string in categorys]
    labels=[int(label) for label in labels]
    
    images_train=[]
    label_train=[]
    images_validate=[]
    label_validate=[]
    NUM_EXAMPLES_PER_EPOCH=0
    for category,label in zip(categorys,labels):
        category_dir=os.path.join(config['data_dir'],category)
        for root,dirs,files in os.walk(category_dir):
            for file in files:
                flag=np.random.rand()
                if flag<=config['train_valid_split_ratio']:
                    images_train.append(os.path.join(category_dir,file))
                    label_train.append(label)
                    NUM_EXAMPLES_PER_EPOCH+=1
                else:
                    images_validate.append(os.path.join(category_dir,file))
                    label_validate.append(label)

    dataset_train=tf.data.Dataset.from_tensor_slices(
            (images_train,label_train))
    dataset_validate=tf.data.Dataset.from_tensor_slices(
            (images_validate,label_validate))      

    return [dataset_train.shuffle(10000).repeat().map(read_image_for_train).
            batch(config['batch_size']),
            dataset_validate.shuffle(10000).repeat().map(read_image_for_validate).
            batch(config['batch_size']),
            NUM_EXAMPLES_PER_EPOCH] 

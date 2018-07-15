'''This file was used to test the data_fetch functions in the data_fetch.py.

You can test the data_fetch.py using the following commandline:
    python data_fetch_test.py -p with dataset.config.batch_size=9

Author :thu-xd(xied15@mails.tsinghua.edu.cn)
'''

import tensorflow as tf
from sacred import Experiment
from data_fetch import data_ingredient,build_dataset
import matplotlib.pyplot as plt
import numpy as np

ex=Experiment('data_fetch_test',ingredients=[data_ingredient])

@ex.automain
def main():
    train_dataset,valid_dataset=build_dataset()
    train_images,train_labels=train_dataset.make_one_shot_iterator().get_next()
    valid_images,valid_labels=valid_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        # test the output of the train_dataset
        images,labels=sess.run([train_images,train_labels])
        plt.figure(figsize=(12,12))
        images_num=images.shape[0]
        rows=images_num/3
        for i in range(images_num):
            plt.subplot(rows+1,3,i+1)
            img=images[i,:,:,:]
            max_num=np.amax(img)
            min_num=np.amin(img)
            img=(img-min_num)/(max_num-min_num)
            plt.imshow(img)
            plt.title(str(labels[i]))
        plt.show()
        
        #test the output of the test_dataset
        images,labels=sess.run([valid_images,valid_labels])
        images_num=images.shape[0]
        rows=images_num/3
        for i in range(images_num):
            plt.figure(figsize=(12,12))
            imgs=images[i,:,:,:,:]
            for j in range(10):
                plt.subplot(4,3,j+1)
                img=imgs[j,:,:,:]
                max_num=np.amax(img)
                min_num=np.amin(img)
                img=(img-min_num)/(max_num-min_num)
                plt.imshow(img)
            plt.show()


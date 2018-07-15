'''This was an tensorflow implementation of the AlexNet in the great paper
"ImageNet Classification with Deep Convolutional Neural Networks"

The alexnet weights are from www.cs.toronto.edu/~guerzhoy/tf_alexnet/

This script enables finetuning alexnet on giving dataset.This script was
inspired by the code from
https://github.com/kratzert/finetune_alexnet_with_tensorflow


Author:thu-xd (xied15@mails.tsinghua.edu.cn)
'''

import tensorflow as tf
import numpy as np
import os
from sacred.stflow import LogFileWriter
import sys
sys.path.append('../')
from experiment import ex

class AlexNet(object):
    '''Class AlexNet was used to define the Architecture of AlexNet'''

    @ex.capture
    def __init__(self,train_dataset,validate_dataset,num_examples_per_epoch,config):
        '''Create the computation graph of the Alexnet

        Args:
            train_dataset:The dataset for finetune train
            validate_dataset:The dataset for finetune validate
            num_examples_per_epoch:number examples per epoch when training
            config:config parameters,we only need config['AlexNet_config'] which including
                --num_class:number of classes
                --stop_gradient_layer:in which layer gradient stop
                backpropagation,it could be
                fc7,fc6,conv5,conv4,conv3,conv2,conv2 conv1 or None.
                --dropout_prob:the dropout probability for full connection
                layers
                --stddev:The stddev of the weights initializer
                --weight_decay:The weight_decay parameter
                --NUM_EPOCHS_PER_DECAY:number epochs per decay
                --INITIAL_LEARNING_RATE:initial Learning_rate
                --LEARNING_RATE_DECAY_FACTOR:learning rate decay factor
                --MAX_BATCHS:maxmium batch for finetune
                --VALID_STEPS: validate steps
                --batch_size:batch_size
            '''

        self.train_dataset=train_dataset
        self.validate_dataset=validate_dataset
        self.num_examples_per_epoch=num_examples_per_epoch
        self.config=config['AlexNet_config']
        self.x=tf.placeholder(tf.float32,[None,227,227,3])
        self.y=tf.placeholder(tf.float32,[None])
        self.training=tf.placeholder(tf.bool,[])
        #Call build_graph function to create the computation graph for AlexNet
        self.build_graph()

    def _trainable_variable(self,scope_name):
        '''This function was used to decide the variables in the scope was
        trainable or not.

        args:
            scope_name:the scope_name of the variable_scope,can be
            fc7,fc6,conv5,conv4,conv3,conv2,conv1 or None
            '''
        if self.config['stop_gradient_layer']==None:
            return True
        if int(scope_name[-1])>int(self.config['stop_gradient_layer'][-1]):
            return True
        return False
    
    def _group_conv(self,x,filter_height,filter_width,filter_num,
                       stride_y,stride_x,name,padding='SAME',groups=1):
        '''This function was used to implement conv for the two-gpu algorithm in the
        original paper.

        args:
            x:the input with shape [batch_size,input_height,input_width,input_channels]
            filter_height:filter_height
            filter_width:filter_width
            filter_num:filter_num
            stride_y:stride_y
            stride_x:stride_x
            name:the name of the scope
            padding: padding stratagy for the input
            groups:How many groups of the filter
            '''
        input_channels=x.get_shape().as_list()[-1]
        convolve=lambda i,f:tf.nn.conv2d(i,f,strides=[1,stride_y,stride_x,1],
                                         padding=padding)

        with tf.variable_scope(name) as scope:
            weights=tf.get_variable("weights",shape=[filter_height,filter_width,
                                                     input_channels/groups,filter_num],
                                    trainable=self._trainable_variable(name))

            biases=tf.get_variable("biases",shape=[filter_num],
                                   trainable=self._trainable_variable(name))
        
        if groups==1:
            conv=convolve(x,weights)
        else:
            x_groups=tf.split(x,groups,3)
            weights_groups=tf.split(weights,groups,3)
            output_groups=[convolve(i,f) for i,f in
                           zip(x_groups,weights_groups)]
            conv=tf.concat(axis=3,values=output_groups)
        
        return tf.nn.relu(tf.nn.bias_add(conv,biases),name=scope.name)


    def _variable_with_weight_decay(self,name,shape,stddev,weight_decay,scope_name):
        '''Get_variable with weight_decay used in full connection layers

        args:
            name:the name of the variable
            shape:the shape of the variable
            stddev:the stddev of the tf.truncated_normal_initializer
            weight_decay:the weight_decay parameter defined in config.weight_decay
            scope_name:The name of the scope
            '''

        var=tf.get_variable(name,shape,dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(mean=0,stddev=stddev),
                            trainable=self._trainable_variable(scope_name))

        if weight_decay is not None:
            weight_decay_loss=tf.multiply(tf.nn.l2_loss(var),weight_decay,name='weight_loss')
            tf.add_to_collection('losses',weight_decay_loss)
        return var


    def build_graph(self):
        '''Create the computation  graph'''
        
        #1st layer  conv(relu) >>  lrn >> pool
        conv1=self._group_conv(self.x,11,11,96,4,4,name='conv1',padding='VALID')
        # The dimension of the conv1 was [55,55,96],55=(227-11)/4+1
        norm1=tf.nn.local_response_normalization(conv1,depth_radius=2,
                                                 alpha=1e-4,beta=0.75,name='norm1')
        pool1=tf.nn.max_pool(norm1,[1,3,3,1],[1,2,2,1],padding='VALID',name='pool1')
        # The dimension of the pool1 was [27,27,96],27=(55-3)/2+1

        #2st layer  conv(relu,groups=2) >> lrn >> pool
        conv2=self._group_conv(pool1,5,5,256,1,1,groups=2,name='conv2')
        # The dimension of the conv2 was [27,27,256],27=(27-5+2*2)/1+1
        norm2=tf.nn.local_response_normalization(conv2,depth_radius=2,
                                                 alpha=1e-4,beta=0.75,name='norm2')
        pool2=tf.nn.max_pool(norm2,[1,3,3,1],[1,2,2,1],padding='VALID',name='pool2')
        # The dimension of the pool2 was [13,13,256],13=(27-3)/2+1

        #3st layer  conv(relu)
        conv3=self._group_conv(pool2,3,3,384,1,1,name='conv3')
        # The dimension of the conv3 was [13,13,384]

        #4st layer  conv(relu,groups=2)
        conv4=self._group_conv(conv3,3,3,384,1,1,groups=2,name='conv4')
        # The dimension of the conv4 was [13,13,384]

        #5st layer  conv(relu) >> pool
        conv5=self._group_conv(conv4,3,3,256,1,1,groups=2,name='conv5')
        # The dimension of the conv5 was [13,13,256]
        pool5=tf.nn.max_pool(conv5,[1,3,3,1],[1,2,2,1],padding='VALID',name='pool5')
        # The dimension of the pool5 was [6,6,256],6=(13-3)/2+1

        #6th Layer: Flatten >> FC >> dropout
        flattened=tf.reshape(pool5,[-1,6*6*256])
        with tf.variable_scope('fc6') as scope:
            weights=self._variable_with_weight_decay('weights',shape=[6*6*256,4096],
                                                   stddev=self.config['stddev'],
                                                   weight_decay=self.config['weight_decay'],
                                                   scope_name='fc6')

            biases=tf.get_variable('biases',[4096],dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1),
                                   trainable=self._trainable_variable('fc6'))

            fc6=tf.nn.relu(tf.matmul(flattened,weights)+biases,name=scope.name)
            dropout6=tf.layers.dropout(fc6,rate=self.config['dropout_prob'],training=self.training)

        #7th layer  FC >> dropout
        with tf.variable_scope('fc7') as scope:
            weights=self._variable_with_weight_decay('weights',shape=[4096,4096],
                                                   stddev=self.config['stddev'],
                                                   weight_decay=self.config['weight_decay'],
                                                   scope_name='fc7')

            biases=tf.get_variable('biases',[4096],dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1),
                                   trainable=self._trainable_variable('fc7'))

            fc7=tf.nn.relu(tf.matmul(dropout6,weights)+biases,name=scope.name)
            dropout7=tf.layers.dropout(fc7,rate=self.config['dropout_prob'],training=self.training)
        
        #8th layer  FC output
        with tf.variable_scope('fc8') as scope:
            weights=self._variable_with_weight_decay('weights',
                                                     shape=[4096,self.config['num_class']],
                                                     stddev=self.config['stddev'],
                                                     weight_decay=self.config['weight_decay'],
                                                     scope_name='fc8')

            biases=tf.get_variable('biases',[self.config['num_class']],dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.1),
                                   trainable=self._trainable_variable('fc8'))

            fc8=tf.matmul(dropout7,weights)+biases
        
        self.linear_output=fc8 
        self.calculate_loss()
        print(tf.trainable_variables())

    def calculate_loss(self):
        '''Get the total loss=cross_entropy_loss+weight_decay_loss'''

        labels=tf.cast(self.y,tf.int64)
        correct_prediction=tf.equal(tf.argmax(self.linear_output,1),labels)
        self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        cross_entropy_loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                          logits=self.linear_output,
                                                                          name='cross_entropy_loss')
        cross_entropy_loss_mean=tf.reduce_mean(cross_entropy_loss,name='cross_entropy_loss_mean')
        tf.add_to_collection('losses',cross_entropy_loss_mean)

        self.loss=tf.add_n(tf.get_collection('losses'),name='total_loss')

    def _load_initial_weights(self,session):
        '''load weights from file into network
        the weights and biases in fc8 was not required to load'''

        weights_dict=np.load("AlexNet/bvlc_alexnet.npy",encoding='bytes').item()
        for op_name in weights_dict:
            if op_name != 'fc8':
                with tf.variable_scope(op_name,reuse=True):
                    for data in weights_dict[op_name]:
                        if len(data.shape)==1:
                            var=tf.get_variable('biases')
                            session.run(var.assign(data))
                        else:
                            var=tf.get_variable("weights")
                            session.run(var.assign(data))

    @ex.capture
    @LogFileWriter(ex)
    def train(self,_run):
        '''Finetune AlexNet on new dataset'''
        
        config=self.config
        global_step=tf.train.get_or_create_global_step()
        #Define the learning rate and train_op
        num_batches_per_epoch=int(self.num_examples_per_epoch/config['batch_size'])
        decay_steps=int(num_batches_per_epoch*config['NUM_EPOCHES_PER_DECAY'])
        lr=tf.train.exponential_decay(config['INITIAL_LEARNING_RATE'],
                                      global_step,
                                      decay_steps,
                                      config['LEARNING_RATE_DECAY_FACTOR'],
                                      staircase=True)
        opt=tf.train.GradientDescentOptimizer(lr)
        train_op=opt.minimize(self.loss,global_step)

        #Define the train and validate examples
        train_img,train_lab=self.train_dataset.make_one_shot_iterator().get_next()
        valid_img,valid_lab=self.validate_dataset.make_one_shot_iterator().get_next()

        with tf.Session() as sess:
            #Save the graph
            tensorboard_graph=tf.summary.FileWriter("/tmp/file_writer",sess.graph)
            #Load the parameters
            self._load_initial_weights(sess)
            #Initial the variables in fc8
            fc8_variables_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='fc8')
            fc8_variables_list.append(global_step)
            init=tf.variables_initializer(fc8_variables_list)
            sess.run(init)
            
            #Finetune Alexnet for config.MAX_BATCHS steps
            for i in range(config['MAX_EPOCHES']*num_batches_per_epoch):
                if i%config['VALID_STEPS']==0:
                    images,labels=sess.run([valid_img,valid_lab])
                    acc=sess.run(self.accuracy,feed_dict={self.x:images,self.y:labels,
                                                          self.training:False})
                    _run.log_scalar('valid_accuracy',acc,i)
                    epoch=int(i/ num_batches_per_epoch)
                    print('Accuracy of valid at epoch%d step%d:%.2f'%(epoch,i,acc))
                else:
                    images,labels=sess.run([train_img,train_lab])
                    _,losses,acc=sess.run([train_op,self.loss,self.accuracy],
                                          feed_dict={self.x:images,self.y:labels,self.training:True})
                    _run.log_scalar('training_loss',losses,i)
                    _run.log_scalar('training_accuracy',acc,i)
                    epoch=int(i/ num_batches_per_epoch)
                    print('Training epoch%d step%d,loss:%.2f,accuracy:%.2f'%(epoch,i,losses,acc))

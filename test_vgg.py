#!/usr/bin/env python
from __future__ import division
import argparse
import os
import tensorflow as tf
import datetime
import numpy as np
import time
from vgg_config import InvasiveSpeciesConfigVgg
from tensorflow.contrib import slim
import vgg
from tf_record import vgg_preprocessing, tfrecord2metafilename, read_and_decode
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import csv


def val_vgg16_with_labels(device):
    """
    Computes accuracy for the validation dataset
    Input: gpu device 
    Output: None
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    config = InvasiveSpeciesConfigVgg()

    val_meta = np.load(tfrecord2metafilename(config.val_fn))
    print ('Using test tfrecords: %s | %s images' % (config.val_fn, len(val_meta['labels'])))
    

    val_filename_queue = tf.train.string_input_producer([config.val_fn]
    , num_epochs=1) # 1 epoch, passing through the the dataset once

    val_image, val_label, _ = read_and_decode(filename_queue = val_filename_queue,
                                             img_dims = config.input_image_size,
                                             resize_to = config.resize_to,
                                             model_dims = config.output_image_size,
                                             size_of_batch = config.test_batch_size,
                                             num_of_threads = 1,
                                             scale_jitter = False,
                                             rand_flip = False,
                                             rand_crop = False,
                                             shuffle = False) # do not shuffle while validating 

    processed_image = vgg_preprocessing(val_image)
    with tf.variable_scope('vgg_16') as vgg16_scope:
        with slim.arg_scope(vgg.vgg_arg_scope()):
            test_logits, _ = vgg.vgg_16(inputs = processed_image, 
                                    num_classes=config.output_shape,
                                    scope=vgg16_scope,
                                    batch_norm=config.batch_norm, 
                                    is_training=False) # important to set to False for validation to use bath norm 
                                                       # in inference mode (normalized with moving statistics)
                                                       # see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization


        probabilities = tf.nn.softmax(test_logits)

    
    prob_and_label = [probabilities, val_label]
    restorer = tf.train.Saver()
    print "Variables stored in checpoint:"
    print_tensors_in_checkpoint_file(file_name=config.test_checkpoint, tensor_name='',all_tensors='')
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:


      
        sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
        restorer.restore(sess, config.test_checkpoint)
        

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        count = 0
        correct_count = 0

        try:

            while not coord.should_stop():
               
                prob_and_label_val = sess.run(prob_and_label)

                prob = prob_and_label_val[0]
                label = prob_and_label_val[1] 
                
                label = int(label[0])
                pred_label = np.argmax(prob[0])
                if pred_label == label:
                    correct_count +=1
                msg = "image number: {0} | prediction : {1} | True label is: {2}"
                print(msg.format((count + 1), experiment_labels[pred_label], experiment_labels[label]))
                count += 1

        except tf.errors.OutOfRangeError:
            print "Total accuracy: {0}".format(correct_count/count)
            print 'Done Testing model :)'
        finally:
            coord.request_stop()  
        coord.join(threads)

def test_vgg16_no_labels(device):
    """
    Creates a csv with predictions for the test data set
    Input: gpu device 
    Output: None
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    config = InvasiveSpeciesConfigVgg()

    test_filename_queue = tf.train.string_input_producer([config.test_fn]
    , num_epochs=1)

    test_image, file_name = read_and_decode(filename_queue = test_filename_queue,
                                             img_dims = config.input_image_size,
                                             model_dims = config.output_image_size,
                                             resize_to = config.resize_to,
                                             size_of_batch = config.test_batch_size,
                                             num_of_threads = 1,
                                             labels = False,
                                            scale_jitter = False,
                                             rand_flip = False,
                                             rand_crop = False,
                                             shuffle = False) # do not shuffle while testing 

    processed_image = vgg_preprocessing(test_image)
    with tf.variable_scope('vgg_16') as vgg16_scope:
        with slim.arg_scope(vgg.vgg_arg_scope()):
            test_logits, _ = vgg.vgg_16(inputs = processed_image, 
                                    num_classes=config.output_shape,
                                    scope=vgg16_scope,
                                    batch_norm=config.batch_norm, 
                                    is_training=False) # important to set to False for validation to use bath norm 
                                                       # in inference mode (normalized with moving statistics)
                                                       # see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization

        probabilities = tf.nn.softmax(test_logits)

    
    prob_file_name_logits = [probabilities, file_name, test_logits]
    restorer = tf.train.Saver()
    print "Variables stored in checpoint:"
    print_tensors_in_checkpoint_file(file_name=config.test_checkpoint, tensor_name='',all_tensors='')
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
        restorer.restore(sess,config.test_checkpoint)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        file_name_prob = []

        try:

            while not coord.should_stop():
               
                prob_file_logits = sess.run(prob_file_name_logits)

                prob = prob_file_logits[0][0]
                file_name = prob_file_logits[1][0]
                logits = prob_file_logits[-1][0]

                pred_label = np.argmax(prob)


                file_num = int(file_name.split('/')[-1].split('.')[0])
                file_name_prob.append((file_num, prob))

               
                msg = "file name: {0} | prediction : {1} "
                print msg.format(file_name, experiment_labels[pred_label])

        except tf.errors.OutOfRangeError:
            print 'Done Testing model :)'
        finally:
            coord.request_stop()  
        coord.join(threads)

        sorted_file_name_prob = sorted(file_name_prob, key = lambda x : x[0])

        fn = 'vgg_results.csv' 

        with open(fn, 'wb') as f: # creating csv with prediction results
            writer = csv.writer(f)
            writer.writerow(['name', 'invasive'])
            for name, probs in sorted_file_name_prob:
                writer.writerow([name,probs[1]])


if __name__ == '__main__':
    experiment_labels = {0:'non-invasive', 1:'invasive'}
    parser = argparse.ArgumentParser()
    parser.add_argument("set")
    parser.add_argument("device")
    args = parser.parse_args()

    if args.set == 'test':
        test_vgg16_no_labels(args.device)
    elif args.set == 'val':
         val_vgg16_with_labels(args.device)
    else:
        raise Exception('options for set arg are val or test!')





    
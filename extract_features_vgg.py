#!/usr/bin/env python
# Train or fine-tune a VGG16 model
from __future__ import division
import os
import tensorflow as tf
import numpy as np
import random
import argparse
from tensorflow.contrib import slim
import vgg
from vgg_config import InvasiveSpeciesConfigVgg
from tf_record import vgg_preprocessing, tfrecord2metafilename, read_and_decode
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def vgg16_feature_extract(device):
    """
    Loads test tf records and stores fc7 features vectors and image filepaths to numpy arrays.
    Input: gpu device number 
    Output None
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi to see available options '0' means first gpu
    config = InvasiveSpeciesConfigVgg() # loads pathology configuration defined in vgg_config

    with tf.Graph().as_default():
        # loading test data
        val_meta = np.load(tfrecord2metafilename(config.val_fn))
        print 'Using test tfrecords: {0} | {1} images'.format(config.val_fn, len(val_meta['labels']))
        val_filename_queue = tf.train.string_input_producer([config.val_fn] , num_epochs=1) # 1 epoch, passing through the 
                                                                                              # the dataset once

        val_images, _, val_file_path = read_and_decode(filename_queue = val_filename_queue,
                                             img_dims = config.input_image_size,
                                             resize_to = config.resize_to,
                                             model_dims = config.output_image_size,
                                             size_of_batch = config.test_batch_size,
                                             num_of_threads = 1,
                                             scale_jitter = False,
                                             rand_flip = False,
                                             rand_crop = False,
                                             shuffle = False) # do not shuffle while validating 

        processed_images = vgg_preprocessing(val_images)
        with slim.arg_scope(vgg.vgg_arg_scope()):
            _, end_points = vgg.vgg_16(inputs = processed_images, 
                                        num_classes=config.output_shape,
                                        batch_norm=config.batch_norm, 
                                        is_training=False) # important to set to False for test to use bath norm 
                                                           # in inference mode (normalized with moving statistics)
                                                           # see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization

        restorer = tf.train.Saver()
        print "Variables stored in checpoint:"
        print_tensors_in_checkpoint_file(file_name=config.test_checkpoint, tensor_name='',all_tensors='')

        # Initialize the graph
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            restorer.restore(sess, config.test_checkpoint) 

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            feature_array = []
            file_paths = []
            batch_num = 1

            try:

                while not coord.should_stop():
                    fc7, file_path = sess.run([end_points['vgg_16/fc7'], val_file_path])
                    feature_array += list(fc7) # storing batch fc7 features
                    file_paths += list(file_path) # sotring batch image filepaths later used in embedding_visalization.py
                    print "appending current features batch number: {0}".format(batch_num)
                    batch_num += 1

            except tf.errors.OutOfRangeError:

                print 
                print "Saving file ..."
                np.save(os.path.join(config.features, 'val_file_path.npy'), file_paths)
                np.save(os.path.join(config.features, 'val_fc7.npy'),feature_array)
                print "Done saving file!"
                print 
                print('Done Creating Features :)')
            finally:
                coord.request_stop()  
            coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    args = parser.parse_args()
    vgg16_feature_extract(args.device)
#!/usr/bin/env python
from __future__ import division
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy import misc
from tqdm import tqdm
from scipy import misc
from random import randint
# from models.slim.preprocessing import inception_preprocessing

VGG_MEAN = [103.939, 116.779, 123.68]

def tfrecord2metafilename(tfrecord_filename):
    """
    Derive associated meta filename for a tfrecord filename
    Input: /path/to/foo.tfrecord
    Output: /path/to/foo.meta
    """
    base, ext = os.path.splitext(tfrecord_filename)
    return base + '_meta.npz'

def iterate_files(img_and_labels, imgs, file_pointers, labels, delimiter):
    """
    Reads data from img_and_labels and imgs and loads the file pointers
    and labels into their corresponding arrays
    Input: text_file_array, image_array 
    Output: None
    """
    labels_dic = {}
    for line in img_and_labels:
        split_line = line.strip('\r\n').split(delimiter) # text file is seperated by tabs
        img = split_line[0]
        label = split_line[1]

        labels_dic[img] = int(label)
        
    for f in imgs:
        file_name = f.split('/')[-1]
        image_name = file_name.split('.')[0]

        if image_name in labels_dic:
            file_pointers += [f]
            labels += [labels_dic[image_name]]

def vgg_preprocessing(image_rgb):
    """
    Preprocssing the given image for evalutaiton with vgg16 model 
    Input: A Tensor representing an image of size [224, 224, 3]
    Output: A processed image 
    """

    image_rgb_scaled = image_rgb * 255.0
    red, green, blue = tf.split(num_or_size_splits=3, axis=3, value=image_rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    image_bgr = tf.concat(values = [
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
        ], axis=3)
    assert image_bgr.get_shape().as_list()[1:] == [224, 224, 3], image_bgr.get_shape().as_list()
    return image_bgr

def inception_preprocessing(image):
    red, green, blue = tf.split(num_or_size_splits=3, axis=3, value=image)
    assert red.get_shape().as_list()[1:] == [299, 299, 1]
    assert green.get_shape().as_list()[1:] == [299, 299,  1]
    assert blue.get_shape().as_list()[1:] == [299, 299, 1]
    image_adjusted = tf.concat(values = [
        2*(red - 0.5),
        2*(green - 0.5),
        2*(blue - 0.5),
        ], axis=3)
    assert image_adjusted.get_shape().as_list()[1:] == [299, 299, 3], image_adjusted.get_shape().as_list()
    return image_adjusted

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def create_tf_record(tfrecords_filename, file_pointers, labels=None, is_grayscale=False, resize=True):
    """
    Creates tf records by writing image data to binary files (allows for faster 
    reading of data). Meta data for the tf records is stored as well
    Input:  tfrecords_filename - directory to store tfrecords
            file_pointers - list of filepointers to where the images are located
            labels - list of labels to the corresponding images
            is_grayscale - boolean for whether the images are grayscale or not
    Output: None
    """
    
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    if labels == None:

        print '%d files' % (len(np.unique(file_pointers)))
    
        for img_path in tqdm(file_pointers):
           
            img = np.array(Image.open(img_path))

            if is_grayscale:
                img = np.expand_dims(img,-1)
                img = np.repeat(img, 3, -1)

            if resize:
                # print "resizing"
                img = misc.imresize(img, (512, 512)) 
        
            img_raw = img.tostring()
            path_raw = img_path.encode('utf-8')

            example = tf.train.Example(features=tf.train.Features(feature={
        
                    'image_raw': _bytes_feature(img_raw),
                    'file_path': _bytes_feature(path_raw),

                   }))

            writer.write(example.SerializeToString())

        writer.close()

       
    else:

        print '%d files in %d categories' % (len(np.unique(file_pointers)), len(np.unique(labels)))

        for img_path, l in tqdm(zip(file_pointers, labels)):

                img = np.array(Image.open(img_path))

                if is_grayscale:
                    img = np.expand_dims(img,-1)
                    img = np.repeat(img, 3, -1)

                if resize:
                    img = misc.imresize(img, (512, 512)) 
            
                img_raw = img.tostring()
                path_raw = img_path.encode('utf-8')

                example = tf.train.Example(features=tf.train.Features(feature={
            
                        'image_raw': _bytes_feature(img_raw),
                        'file_path': _bytes_feature(path_raw),
                        'label':_int64_feature(int(l)),

                       }))

                writer.write(example.SerializeToString())

        writer.close()

        meta = tfrecord2metafilename(tfrecords_filename)
        np.savez(meta, file_pointers=file_pointers, labels=labels, output_pointer=tfrecords_filename)

    print '-' * 100
    print 'Generated tfrecord at %s' % tfrecords_filename
    print '-' * 100


def read_and_decode(filename_queue, img_dims, resize_to, model_dims, size_of_batch,\
                     num_of_threads,labels=True, scale_jitter=True, rand_flip = True, rand_crop = True, shuffle=True):

    """
    Reads in tf records and decodes the features of the image 
    Input: filename_queue - a node in a TensorFlow Graph used for asynchronous computations
           img_dims - dimensions of the tensor image, example: [256, 256, 3] 
           resize_to - dimensions to reisze image to,  example: [224, 224] 
           model_dims - dimensions of the tensor image that the model accepts, example: [224, 224, 3] 
           size_of_batch - size of the batch that will be fed into the model, example: 30
           num_threads - number of threads that execute a training op that dequeues mini-batches from the queue 
           labels - boolean wheter the tfrecords to be decoded contains labels
           scale_jitter - boolean whether to randomly resize image within a range of 256 and 512
           rand_flip - boolean wheter to randomly filp an image left or right
           rand_crop - boolean whether to random_crop is going to be set to true 
           shuffle - boolean wheter to randomly shuffle images while feeding them to the graph 
    Output: tensor image, label of the image and filepath to the image
    """
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    if not labels:
        features = tf.parse_single_example(
          serialized_example,
        
          features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'file_path': tf.FixedLenFeature([], tf.string),
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)

        file_path = tf.cast(features['file_path'], tf.string)
        
        
        image = tf.reshape(image, img_dims)

        image = tf.cast(image, tf.float32)


        image = tf.image.resize_images(image, resize_to)

        if scale_jitter:
            random_size = randint(256,512)
            image = tf.image.resize_images(image, [random_size, random_size])
        else:
            image = tf.image.resize_images(image,resize_to)

        if rand_crop:
            image = tf.random_crop(image, model_dims)

        else:
            image = tf.image.resize_image_with_crop_or_pad(image, model_dims[0],\
                                                         model_dims[1])

        if rand_flip:
            image = tf.image.random_flip_left_right(image)

        image = tf.to_float(image)


        image = image/255

        if shuffle:
      
            img, f = tf.train.shuffle_batch([image, file_path],
                                                         batch_size=size_of_batch,
                                                         capacity=5000,
                                                         min_after_dequeue=500,
                                                         num_threads=num_of_threads)
        else:
            img, f = tf.train.batch([image, file_path],
                                                         batch_size=size_of_batch,
                                                         capacity=5000,
                                                         num_threads=num_of_threads)

        
        return img, f

    else:
        features = tf.parse_single_example(
          serialized_example,
        
          features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'file_path': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)

        label = tf.cast(features['label'], tf.int32)

        file_path = tf.cast(features['file_path'], tf.string)
        
        
        image = tf.reshape(image, img_dims)

        image = tf.cast(image, tf.float32)
        
        if scale_jitter:
            random_size = randint(256,512)
            image = tf.image.resize_images(image, [random_size, random_size])
        else:
            image = tf.image.resize_images(image,resize_to)


        if rand_crop:
            image = tf.random_crop(image, model_dims)

        else:
            image = tf.image.resize_image_with_crop_or_pad(image, model_dims[0],\
                                                         model_dims[1])

        if rand_flip:
            image = tf.image.random_flip_left_right(image)

        image = tf.to_float(image)


        image = image/255

        if shuffle:
      
            img, l, f = tf.train.shuffle_batch([image, label, file_path],
                                                         batch_size=size_of_batch,
                                                         capacity=5000,
                                                         min_after_dequeue=500,
                                                         num_threads=num_of_threads)
        else:
            img, l, f = tf.train.batch([image, label, file_path],
                                                         batch_size=size_of_batch,
                                                         capacity=5000,
                                                         num_threads=num_of_threads)        
        return img, l, f



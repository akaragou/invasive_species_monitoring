import os
import cv2
import csv
import numpy as np
import tensorflow as tf
import glob
from tqdm import tqdm
import argparse
from vgg_config import InvasiveSpeciesConfigVgg
from tensorflow.contrib.tensorboard.plugins import projector

def images_to_sprite(data):
    """
    Source: https://github.com/tensorflow/tensorflow/issues/6322
    Creates the sprite image along with any necessary padding
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data

def creating_embedding(device):
    """
    Loads test images and fc7 features. Creates sprite images, metadata class file 
    and configurates tensorflow embeddings to be used with tensorboard. 
    Input: device 
    Output None.
    """
    name_label_dic = {}
    with open('labels.csv', 'rb') as f:
        f.next()
        for row in f:
            line = row.strip('\n').split(',')
            name_label_dic[int(line[0])] = int(line[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    species_config = InvasiveSpeciesConfigVgg()
    print "loading features..."
    feature_vectors = np.squeeze(np.load(os.path.join(species_config.features, 'val_fc7.npy')))
    print "feature_vectors_shape:",feature_vectors.shape 
    print "num of images:",feature_vectors.shape[0]
    print "size of individual feature vector:",feature_vectors.shape[1]
    features = tf.Variable(feature_vectors, name='features')
    print "done loading feautes!"
    print "creating metadata tsv..."
    file_paths = np.load(os.path.join(species_config.features, 'val_file_path.npy'))
    with open(os.path.join(species_config.embedding, 'metadata_2_classes.tsv'), 'w') as f:
        f.write('Class' + '\t' + 'Name' + '\n')
        for file_path in file_paths:
            num = int(file_path.split('/')[-1].split('.')[0])
            if name_label_dic[num] == 1:
                f.write('invasive' + '\t' + file_path.split('/')[-1] + '\n')
            else:
                f.write('non-invasive' + '\t' + file_path.split('/')[-1] + '\n')
    print "done creating metadata tsv"
    print "loading image data..."
    img_data = []
    for file_path in tqdm(file_paths):
        input_img=cv2.imread(file_path)
        input_img_resize=cv2.resize(input_img,(128,128))
        img_data.append(input_img_resize)
    img_data = np.array(img_data)
    sprite = images_to_sprite(img_data)
    cv2.imwrite(os.path.join(species_config.embedding, '2_classes.png'), sprite)
    print "done loading image data!"

    with tf.Session() as sess:
        saver = tf.train.Saver([features])

        sess.run(features.initializer)
        saver.save(sess, os.path.join(species_config.embedding, '2_classes.ckpt'))
        
        tf_embedding_config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        embedding = tf_embedding_config.embeddings.add()
        embedding.tensor_name = features.name
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = 'metadata_2_classes.tsv'
        # Comment out if you don't want sprites
        embedding.sprite.image_path = os.path.join(species_config.embedding, '2_classes.png')
        embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
        # Saves a config file that TensorBoard will read during startup.
        projector.visualize_embeddings(tf.summary.FileWriter(species_config.embedding), tf_embedding_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    args = parser.parse_args()
    creating_embedding(args.device)
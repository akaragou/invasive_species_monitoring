#!/usr/bin/env python
from __future__ import division
import os
import tensorflow as tf
import argparse
import datetime
import numpy as np
import time
import vgg
from tensorflow.contrib import slim
from vgg_config import InvasiveSpeciesConfigVgg
from tf_record import vgg_preprocessing, tfrecord2metafilename, read_and_decode
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

def train_vgg(device):
    """
    Loads training and validations tf records and trains vgg model and validates every number of fixed steps.
    Input: gpu device number 
    Output None
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device) # use nvidia-smi linux command to see which device are available
    config = InvasiveSpeciesConfigVgg()

    with tf.Graph().as_default():
        # load training data
        train_meta = np.load(tfrecord2metafilename(config.train_fn))
        print 'Using train tfrecords: {0} | {1} images'.format(config.train_fn, len(train_meta['labels']))
        train_filename_queue = tf.train.string_input_producer(
        [config.train_fn], num_epochs=config.num_train_epochs)
        # load val data
        val_meta = np.load(tfrecord2metafilename(config.val_fn))
        print 'Using test tfrecords: {0} | {1} images'.format(config.val_fn, len(val_meta['labels']))
        val_filename_queue = tf.train.string_input_producer(
        [config.val_fn], num_epochs=config.num_train_epochs)

         # defining model names and setting output and summary directories
        model_train_name = 'vgg16_train_scale_jitter_' + str(config.resize_to[0])
        dt_stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = config.get_results_path(model_train_name, dt_stamp)
        summary_dir = config.get_summaries_path(model_train_name, dt_stamp)
        print '-'*60
        print 'Training model: {0}'.format(dt_stamp)
        print '-'*60

        train_images, train_labels, _ = read_and_decode(filename_queue = train_filename_queue,
                                                         img_dims = config.input_image_size,
                                                         resize_to = config.resize_to,
                                                         model_dims = config.output_image_size,
                                                         size_of_batch = config.train_batch_size,
                                                         num_of_threads = 2,
                                                         scale_jitter=True,
                                                         rand_flip = True,
                                                         rand_crop = True,
                                                         shuffle = True)

        val_images, val_labels, _ = read_and_decode(filename_queue = val_filename_queue,
                                                     img_dims = config.input_image_size,
                                                     resize_to = config.resize_to,
                                                     model_dims = config.output_image_size,
                                                     size_of_batch = config.val_batch_size,
                                                     num_of_threads = 2,
                                                     scale_jitter=False,
                                                     rand_flip = False,
                                                     rand_crop = False,
                                                     shuffle = False)

        # summaries to use with tensorboard check https://www.tensorflow.org/get_started/summaries_and_tensorboard
        tf.summary.image('train images', train_images)
        tf.summary.image('validation images', val_images)

        # creating step op that counts the number of training steps
        step = get_or_create_global_step()
        step_op = tf.assign(step, step+1)

        with tf.variable_scope('vgg_16') as vgg16_scope:
            with tf.name_scope('train') as train_scope:


                train_processed_images = vgg_preprocessing(train_images) # subtracting vgg means to center data 
                regularizer = tf.contrib.layers.l2_regularizer(config.l2_regularization) # using l2 regularization for weight decay
                with slim.arg_scope(vgg.vgg_arg_scope()):
                    

                    train_logits, _ = vgg.vgg_16(inputs = train_processed_images, 
                                            num_classes = config.output_shape,
                                            scope = vgg16_scope,
                                            regularizer = regularizer,
                                            batch_norm = config.batch_norm,
                                            is_training = True) 
                

                one_hot_lables = tf.one_hot(train_labels, config.output_shape)
                batch_loss = tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_lables, logits = train_logits)

                loss = tf.reduce_mean(batch_loss)
                tf.summary.scalar("loss", loss)


                lr = tf.train.exponential_decay(
                        learning_rate = config.initial_learning_rate,
                        global_step = step_op,
                        decay_steps = config.decay_steps,
                        decay_rate = config.learning_rate_decay_factor,
                        staircase = True) # if staircase is True decay the learning rate at discrete intervals

                if config.optimizer == "adam":
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # used to update batch norm params. see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
                    with tf.control_dependencies(update_ops):
                        train_op =  tf.train.AdamOptimizer(lr).minimize(loss)
                elif config.optimizer == "sgd":
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        train_op =  tf.train.GradientDescentOptimizer(lr).minimize(loss)
                elif config.optimizer == "nestrov":
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        train_op =  tf.train.MomentumOptimizer(lr, config.momentum, use_nesterov=True).minimize(loss)
                else:
                    raise Exception("Not known optimizer! options are adam, sgd or nestrov")

                train_prob = tf.nn.softmax(train_logits)
               
                train_accuracy = config.class_accuracy(predictions=train_prob, labels=train_labels)
                tf.summary.scalar("training accuracy", train_accuracy)

            vgg16_scope.reuse_variables() # training variables are reused in validation graph 

            with tf.name_scope('val') as val_scope:
                
                val_processed_images = vgg_preprocessing(val_images)
                with slim.arg_scope(vgg.vgg_arg_scope()):
                    val_logits, _ = vgg.vgg_16(inputs = val_processed_images, 
                                            num_classes=config.output_shape,
                                            scope=vgg16_scope,
                                            batch_norm = config.batch_norm,
                                            is_training=False) # important to set to False for validation to use bath norm in inference mode (normalized with moving statistics)
                                                               # see https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
      
                val_prob = tf.nn.softmax(val_logits)

                val_accuracy = config.class_accuracy(predictions=val_prob, labels=val_labels)
                tf.summary.scalar("validation accuracy", val_accuracy)

                
        # adjusting variables to keep in the model
        # variables that are exluded will allow for transfer learning (normally fully connected layers are excluded)
        exclusions = [scope.strip() for scope in config.checkpoint_exclude_scopes]
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        print "Restroing variables:"
        for var in variables_to_restore:
            print var
        restorer = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver(slim.get_model_variables(), max_to_keep=100)

        summary_op = tf.summary.merge_all()
        # Initialize the graph
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Need to initialize both of these if supplying num_epochs to inputs
            sess.run(tf.group(tf.global_variables_initializer(),
                 tf.local_variables_initializer()))
            restorer.restore(sess, config.model_path)
            summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            np.save(os.path.join(out_dir, 'training_config_file'), config)
            val_acc_max, losses = 0, []

            try:

                while not coord.should_stop():

                    start_time = time.time()
                    step_count, loss_value, train_acc,lr_value, _ = sess.run([step_op, loss, train_accuracy,lr, train_op])
                    losses.append(loss_value)
                    duration = time.time() - start_time
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    step_count = step_count - 1 

                    if step_count % config.validate_every_num_steps == 0:
                        it_val_acc = np.asarray([])
                        for num_vals in range(config.num_batches_to_validate_over):
                            # Validation accuracy as the average of n batches
                            it_val_acc = np.append(
                                it_val_acc, sess.run(val_accuracy))
                        
                        val_acc_total = it_val_acc.mean()
                        # Summaries
                        summary_str= sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step_count)

                        # Training status and validation accuracy
                        msg = '{0}: step {1}, loss = {2:.2f} ({3:.2f} examples/sec; '\
                            + '{4:.2f} sec/batch) | Training accuracy = {5:.4f} '\
                            + '| Validation accuracy = {6:.4f} | logdir = {7}'
                        print msg.format(
                              datetime.datetime.now(), step_count, loss_value,
                              (config.train_batch_size / duration), float(duration),
                              train_acc, val_acc_total, summary_dir)
                        print "learning rate: ", lr_value

                        # Save the model checkpoint if it's the best yet
                        if val_acc_total >= val_acc_max:
                            file_name = 'vgg16_{0}_{1}'.format(dt_stamp, step_count)
                            saver.save(
                                sess,
                                config.get_checkpoint_filename(model_train_name, file_name))
                            # Store the new max validation accuracy
                            val_acc_max = val_acc_total

                    else:
                        # Training status
                        msg = '{0}: step {1}, loss = {2:.2f} ({3:.2f} examples/sec; '\
                            + '{4:.2f} sec/batch) | Training accuracy = {5:.4f}'
                        print msg.format(datetime.datetime.now(), step_count, loss_value,
                              (config.train_batch_size / duration),
                              float(duration), train_acc)
                    # End iteration

            except tf.errors.OutOfRangeError:
                print 'Done training for {0} epochs, {1} steps.'.format(config.num_train_epochs, step_count)
            finally:
                coord.request_stop()
                np.save(os.path.join(out_dir, 'training_loss'), losses)
            coord.join(threads)
            sess.close()
                



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("device")
    args = parser.parse_args()
    train_vgg(args.device)
  
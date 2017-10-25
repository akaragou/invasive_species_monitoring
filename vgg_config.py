import os
import tensorflow as tf

class InvasiveSpeciesConfigVgg():
    def __init__(self, **kwargs):

        self.main_dir = '/media/data_cifs/x8/andreas/kaggle/invasive_species_monitoring/'
        self.checkpoint_path = os.path.join(self.main_dir, 'checkpoints')
        self.summary_path = os.path.join(self.main_dir, 'summaries')
        self.results_path = os.path.join(self.main_dir, 'results')
        self.features = os.path.join(self.main_dir, 'features')
        self.embedding = os.path.join(self.main_dir, 'embedding')
        self.model_path = os.path.join(self.main_dir, 'model_checkpoints/vgg_16.ckpt')
        self.train_fn =  os.path.join(self.main_dir, 'tfrecords/train.tfrecords')
        self.val_fn =  os.path.join(self.main_dir, 'tfrecords/val.tfrecords')
        self.test_fn =  os.path.join(self.main_dir, 'tfrecords/test.tfrecords')
        self.test_checkpoint = os.path.join(self.checkpoint_path, 'vgg16_train_256/vgg16_20170813_145713_400.ckpt')

        self.optimizer = "nestrov"
        self.initial_learning_rate = 1e-03
        self.momentum = 0.95 # if optimizer is nestrov
        self.decay_steps = 300
        self.learning_rate_decay_factor = 0.5
        self.train_batch_size = 90
        self.val_batch_size = 50
        self.test_batch_size = 1
        self.num_batches_to_validate_over = 10
        self.validate_every_num_steps = 50
        self.num_train_epochs = 100
        self.output_shape = 2
        self.input_image_size = [512, 512, 3]
        self.resize_to = [256, 256]
        self.output_image_size = [224,224, 3]
        self.l2_regularization = 0.0005
        self.batch_norm = True


        self.checkpoint_exclude_scopes = ["vgg_16/fc6", "vgg_16/fc7", "vgg_16/fc8", "vgg_16/batchnormfc6", "vgg_16/batchnormfc7", "vgg_16/batchnormfc8"]
    
    def class_accuracy(self, predictions, labels):
        return tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(predictions, 1), tf.cast(labels, dtype=tf.int64))))

    def get_checkpoint_filename(self, model_name, run_name):
            """ Return filename for a checkpoint file. Ensure path exists. """
            pth = os.path.join(self.checkpoint_path, model_name)
            if not os.path.isdir(pth): os.makedirs(pth)
            return os.path.join(pth, run_name + '.ckpt')

    def get_summaries_path(self, model_name, run_name):
        """ Return filename for files. Ensure path exists. """
        pth = os.path.join(self.summary_path, model_name)
        if not os.path.isdir(pth): os.makedirs(pth)
        return os.path.join(pth, run_name)

    def get_results_path(self, model_name, run_name):
        """ Return filename for files. Ensure path exists. """
        pth = os.path.join(self.results_path, model_name, run_name)
        if not os.path.isdir(pth): os.makedirs(pth)
        return pth
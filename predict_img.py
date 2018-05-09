#!/usr/bin/env python

from __future__ import print_function

import tensorflow as tf

tf.app.flags.DEFINE_integer('num_classes', 2, 'The number of classes.')
tf.app.flags.DEFINE_string('infile','None', 'Image file, one image per line.')
tf.app.flags.DEFINE_boolean('tfrecord',False, 'Input file is formatted as TFRecord.')
tf.app.flags.DEFINE_string('outfile','None', 'Output file for prediction probabilities.')
tf.app.flags.DEFINE_string('model_name', 'resnet_v1_152', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_string('checkpoint_path', 'Result','The directory where the model was written to or an absolute path to a checkpoint file.')
tf.app.flags.DEFINE_integer('eval_image_size', None, 'Eval image size.')
FLAGS = tf.app.flags.FLAGS

import numpy as np
import os
import sys
from datasets import imagenet
from nets import inception
from nets import resnet_v1
from nets import inception_utils
from nets import resnet_utils
from preprocessing import inception_preprocessing
from nets import nets_factory
from preprocessing import preprocessing_factory
from scipy.misc import imread
slim = tf.contrib.slim




def predict(fl):
	  
                tf.reset_default_graph()
		image_name = None
		model_name_to_variables = {'inception_v3':'InceptionV3','inception_v4':'InceptionV4','resnet_v1_152':'resnet_v1_152','inception_resnet_v2':'InceptionResnetV2'}

		preprocessing_name = 'resnet_v1_152'
		
		
		model_variables = model_name_to_variables.get('resnet_v1_152')
		if model_variables is None:
		  tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
		  sys.exit(-1)

		
		if tf.gfile.IsDirectory('Result'):
		  checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
		else:
		  checkpoint_path = FLAGS.checkpoint_path

		image_string = tf.placeholder(tf.string) # Entry to the computational graph, e.g. image_string = tf.gfile.FastGFile(image_file).read()


		
		image = tf.image.decode_jpeg(image_string, channels=3) ## To process corrupted image files
		

		image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training=False)

		network_fn = nets_factory.get_network_fn(FLAGS.model_name, FLAGS.num_classes, is_training=False)

		if FLAGS.eval_image_size is None:
		  eval_image_size = network_fn.default_image_size

		processed_image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

		processed_images  = tf.expand_dims(processed_image, 0) 
		

		logits, _ = network_fn(processed_images)

		probabilities = tf.nn.softmax(logits)

		init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))

		with tf.Session() as sess:

			init_fn(sess)
		       
			x = tf.gfile.FastGFile(fl).read() # You can also use x = open(fl).read()
			
		       
			probs = sess.run(probabilities, feed_dict={image_string:x})
		      

			probs = probs[0, 0:]
			index = np.argmax(probs)
		        
	        return index
	  

	       
	

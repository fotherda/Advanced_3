from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import sys
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import Advanced_1.confusionMatrix as cm

from tensorflow.examples.tutorials.mnist import input_data
from Advanced_1.convergenceTester import ConvergenceTester
from Advanced_1.learningRateScheduler import LearningRateScheduler
from sklearn.metrics import confusion_matrix

root_dir = 'C:/Users/Dave/Documents/GI13-Advanced/Assignment1';
summaries_dir = root_dir + '/Summaries';
# save_dir = root_dir + '/SavedVariables';
save_dir = root_dir;


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def print_confusion_matrix_command_line(x, y_, X, Y_, argm_y, argm_y_, sess, name, keep_prob):
    pred_label = sess.run([argm_y, argm_y_], feed_dict={x: X, y_: Y_, keep_prob: 1.0})   
    
    cm = tf.contrib.metrics.confusion_matrix(pred_label[0], pred_label[1], 
                num_classes=None, dtype=tf.int32, name=None, weights=None)   
    
    print( name, ' confusion matrix')
    cm_str = np.array2string(cm.eval(), separator=', ')
    print(str(cm_str).replace('[','').replace(']',''))
    
def print_confusion_matrix(x, y_, X, Y_, argm_y, argm_y_, sess, keep_prob, model_type):

    pred_label = sess.run([argm_y, argm_y_], feed_dict={x: X, y_: Y_, keep_prob: 1.0})   
    
    y_pred = pred_label[0]
    y_true = pred_label[1]
    
    cnf_matrix = confusion_matrix(y_true, y_pred)
    cm_filename = model_type.replace(' ','_') + '.p'
    pickle.dump( cnf_matrix, open( cm_filename, "wb" ) )    
    
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Plot non-normalized confusion matrix
    plt.figure()
    cm.plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title=('Confusion matrix - ' + model_type))    
    plt.show()

def part_a(x):
    W = weight_variable([784, 10])
    b = weight_variable([10])
    y = tf.matmul(x, W) + b 
    
    return y
   
def part_b(x):    
    W_1 = weight_variable([784, 128])
    b_1 = bias_variable([128])   
    h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)    
    
    W_2 = weight_variable([128, 10])
    b_2 = bias_variable([10])
    y = tf.matmul(h_1, W_2) + b_2
    
    return y
   
def part_c(x):    
    W_1 = weight_variable([784, 256])
    b_1 = bias_variable([256])    
    h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)    
    
    W_2 = weight_variable([256, 256])
    b_2 = bias_variable([256])   
    h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)    
    
    W_3 = weight_variable([256, 10])
    b_3 = bias_variable([10])
    y = tf.matmul(h_2, W_3) + b_3
    
    return y

def part_d(x):    
    
    x_image = tf.reshape(x, [-1,28,28,1])
    
    W_conv1 = weight_variable([3, 3, 1, 16])
    b_conv1 = bias_variable([16])
    
    h_conv1 = conv2d(x_image, W_conv1) + b_conv1
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([3, 3, 16, 16])
    b_conv2 = bias_variable([16])
    
    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 16, 256])
    b_fc1 = bias_variable([256])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)    
     
    W_fc2 = weight_variable([256, 10])
    b_fc2 = bias_variable([10])
    
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y, keep_prob
        
   
def run_part1_models(FLAGS):
    print('Tensorflow version: ', tf.VERSION)
    print(sys.path)
    
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    if FLAGS.model=='P1_a':
        y = part_a(x) 
    elif FLAGS.model=='P1_b':
        y = part_b(x)
    elif FLAGS.model=='P1_c':
        y = part_c(x) 
    elif FLAGS.model=='P1_d':
        y, keep_prob  = part_d(x)
    
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar('CrossEntropy', cross_entropy)
    
    learning_rate = tf.placeholder(tf.float32, shape=[])
    lr = 0.1;
    decay = lr / 2e6
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # Test trained model
    argm_y = tf.argmax(y, 1)
    argm_y_ = tf.argmax(y_, 1)
    correct_prediction = tf.equal(argm_y, argm_y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    X_train = mnist.train.images
    y_train = mnist.train.labels
    X_test = mnist.test.images
    y_test = mnist.test.labels   

    with tf.Session() as sess:    

        file_name = FLAGS.saved_model_dir + '/' + FLAGS.model + '_' + str(lr) + '.ckpt';    
                   
        if FLAGS.use_saved: #Restore saved model       
            saver2restore = tf.train.Saver(write_version=tf.train.SaverDef.V1)
            tf.global_variables_initializer().run()    
            saver2restore.restore(sess, file_name)
            
        else: #Train new model
            # Merge all the summaries and write them out to file
            merged = tf.summary.merge_all()
            
            dir_name = summaries_dir + '/' + FLAGS.model + '/lRate_' + str(lr);
            train_writer = tf.summary.FileWriter(dir_name + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(dir_name + '/test')
            
            # init operation
            tf.global_variables_initializer().run()    
            
            conv_tester = ConvergenceTester(0.0005, lookback_window=25) #stop if converged to within 0.05%
            lrs = LearningRateScheduler(decay)

            # Train
            for i in range(30000):
                lr = lrs.get_learning_rate(i, lr)
                batch_xs, batch_ys = mnist.train.next_batch(50)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, learning_rate: lr, keep_prob: 0.5})
         
                if i % 100 == 0: #calc intermediate results
                    train_accuracy, train_summary = sess.run([accuracy, merged], feed_dict={x: X_train, y_: y_train, keep_prob: 1.0})                                      
                    test_accuracy, test_summary = sess.run([accuracy, merged], feed_dict={x: X_test, y_: y_test, keep_prob: 1.0})
                    print("step %d, training : test accuracy %g : %g learning rate %f" % (i, train_accuracy, test_accuracy, lr))
                                   
                    train_writer.add_summary(train_summary, i)
                    test_writer.add_summary(test_summary, i)
                    
                    if conv_tester.has_converged(test_accuracy):
                        print('converged after ', i, ' epochs')
                        break
                    
            #save trained model
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
            save_path = saver.save(sess, file_name)
            print("Model saved in file: %s" % save_path)
        
            
        #print final results        
        print("Training Error rate:", 1-accuracy.eval({x: X_train, y_: y_train, 
                                                              keep_prob: 1.0}))
        print("Test Error rate:", 1-accuracy.eval({x: X_test, y_: y_test, 
                                                              keep_prob: 1.0}))
        print_confusion_matrix(x, y_, X_train, y_train, argm_y, argm_y_, 
                               sess, keep_prob, FLAGS.model + ' training data')
        print_confusion_matrix(x, y_, X_test, y_test, argm_y, argm_y_, 
                               sess, keep_prob, FLAGS.model + ' test data')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--use_saved', action='store_true', default=True, help='Use saved data')
    parser.add_argument('--model', type=str, default='a', help='which model to run, one of [a, b, c, d]')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

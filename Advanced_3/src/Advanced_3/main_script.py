from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput


# import Advanced_3.demo as p1        
import Advanced_3.qA3 as qA3      
import Advanced_3.qA4 as qA4      
import Advanced_3.qA5 as qA5      
import Advanced_3.qA6 as qA6      
import Advanced_3.qA7 as qA7      
import Advanced_3.qA8 as qA8      
import Advanced_3.qB1 as qB1      
        

def main(_): 
    
    if FLAGS.model=='qA3':
        qA3.run_net(1e-4, FLAGS)
    elif FLAGS.model=='qA4':
        qA4.run_net(1e-3, FLAGS)
    elif FLAGS.model=='qA5':
        qA5.run_net(1e-3, FLAGS)
    elif FLAGS.model=='qA6':
        qA6.run_net(1e-3, FLAGS)
    elif FLAGS.model=='qA7':
        qA7.run_net(1e-3, FLAGS)
    elif FLAGS.model=='qA8':
        qA8.run_net(1e-3, FLAGS)
    elif FLAGS.model=='qB1':
            qB1.run_net(FLAGS)

#     p1.run_models(FLAGS)
    
   
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('-saved_model_dir', type=str, default='C:/Users/Dave/Documents/GI13-Advanced/Assignment2/model',
                        help='Directory where trained models are saved')
    parser.add_argument('-lr', type=str, default='1e-4', help='learning rate')
    parser.add_argument('-sm', type=str, default=None, help='saved model to evaluate')
    parser.add_argument('-eval', action='store_true', help='just evaluate with saved model')
    parser.add_argument('-bn', action='store_true', help='use batch normalization')
    parser.add_argument('--model', type=str, default='P1_a', 
        help='which model to run, one of [P1_a, P1_b, P1_c, P1_d, P2_a, P2_b, P3_c, P4_d]')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

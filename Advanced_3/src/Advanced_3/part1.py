from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import datetime#, time
import pickle as pi

from timeit import default_timer as timer

from tensorflow.examples.tutorials.mnist import input_data
from Advanced_1.convergenceTester import ConvergenceTester
from Advanced_1.learningRateScheduler import LearningRateScheduler
from Advanced_1.dataBatcher import DataBatcher
from scipy.stats import ttest_ind
from scipy.misc import toimage



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial)
  
def build_network_task1(x, nrecurrent_units, cell, y_, use_batch_norm):
    
    raw_rnn_outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    last_rnn_output = tf.slice(raw_rnn_outputs, [0, raw_rnn_outputs.get_shape()[1].value - 1, 0], 
                               [-1, 1, raw_rnn_outputs.get_shape()[2].value] )
    last_rnn_output = tf.squeeze(last_rnn_output, 1, name='sliced_rnn_outputs')

    W_2 = weight_variable([nrecurrent_units, 100])
    b_2 = bias_variable([100])   
    lin_1 = tf.matmul(last_rnn_output, W_2) + b_2
    
    if use_batch_norm:
        # Calculate batch mean and variance
        batch_mean1, batch_var1 = tf.nn.moments(lin_1,[0])
    
        # Apply the initial batch normalizing transform
        lin_1_hat = (lin_1 - batch_mean1) / tf.sqrt(batch_var1 + 1e-3)
        
        # Create two new parameters, scale and beta (shift)
        scale1 = tf.Variable(tf.ones([100]))
        beta1 = tf.Variable(tf.zeros([100]))
        
        # Scale and shift to obtain the final output of the batch normalization
        # this value is fed into the activation function (here a sigmoid)
        BN1 = scale1 * lin_1_hat + beta1
        h_2 = tf.nn.relu(BN1)    
    else:
        h_2 = tf.nn.relu(lin_1)
        
    W_3 = weight_variable([100, 10])
    b_3 = bias_variable([10])
    y = tf.matmul(h_2, W_3) + b_3
    
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y))
    
    return y, cross_entropy

def build_network_task2(x, nrecurrent_units, cell, y_):
    
    raw_rnn_outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32) # batch_size x 783 x 32
    
    W_1 = weight_variable([nrecurrent_units, 1])
    b_1 = bias_variable([1])   
         
    reshaped_outputs = tf.reshape(raw_rnn_outputs, [-1, nrecurrent_units])
    logits = tf.matmul(reshaped_outputs, W_1) + b_1
    logits = tf.reshape(logits, [-1, raw_rnn_outputs.get_shape()[1].value, 1])
    logits = tf.squeeze(logits, 2, name='logits')
    epsilon = tf.constant(value=0.0000001)
    logits = logits + epsilon
    
    tf.check_numerics(logits,'numerical problem with logits')
    
    y = tf.nn.sigmoid(logits, name='sigmoid_outputs')
    target_float = tf.to_float(y_)
    cross_entropy_all = tf.nn.sigmoid_cross_entropy_with_logits(logits, target_float)
    cross_entropy = tf.reduce_mean(cross_entropy_all)
    
    return y, cross_entropy
          
def show_all_variables():
    total_count = 0
    for idx, op in enumerate(tf.trainable_variables()):
        shape = op.get_shape()
        count = np.prod(shape)
        print("[%2d] %s %s = %s" % (idx, op.name, shape, count))
        total_count += int(count)
    print("[Total] variable size: %s" % "{:,}".format(total_count))
 
def binarize(images, threshold=0.1):
    return (threshold < images).astype('float32')

def add_dimension(images):
    return np.expand_dims(images, -1)

def save_model(session, model_name, root_dir):
    if not os.path.exists(root_dir + '/model/'):
        os.mkdir(root_dir + '/model/')
    saver = tf.train.Saver(write_version=1)
    save_path = saver.save(session, root_dir + '/model/' + model_name +'.ckpt')
#     print("Model saved in file: %s" % save_path)

def import_data(FLAGS, num_train_examples):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    X_train_bin = binarize(mnist.train.images)
    X_test_bin = binarize(mnist.test.images)
    X_train = add_dimension( X_train_bin )
    X_test = add_dimension( X_test_bin )
    
    if FLAGS.model[:2]=='P1':
        y_train = mnist.train.labels
        y_test = mnist.test.labels   
        embedding = np.array([0,1,2,3,4,5,6,7,8,9])
        y_train = (np.dot(y_train, embedding)).astype(int)  
        y_test = (np.dot(y_test, embedding)).astype(int)  
    elif FLAGS.model[:2]=='P2':
        y_train = X_train_bin
        y_test = X_test_bin   
        y_train = np.delete(y_train, 0, 1) #remove first pixel
        y_test = np.delete(y_test, 0, 1)

        npixels = X_train_bin.shape[1]
        X_train = X_train_bin.astype(int)
        X_test = X_test_bin.astype(int)
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
        
    if num_train_examples > 0:
        X_train = X_train[:num_train_examples]
        y_train = y_train[:num_train_examples]    

#     rs = np.reshape(mnist.train.images[0], (28,28))
#     toimage(rs).show()

    print('Imported data')
    return X_train, y_train, X_test, y_test

class Parser():

    def __init__(self, model):
        self._model = model
        self._task = model[:2]
        self._num_layers = int(model[3])
        self._num_units = int(model[5:8])
        self._cell_type = model[9]


def run_models(FLAGS):
    print('Tensorflow version: ', tf.VERSION)
    print('PYTHONPATH: ',sys.path)
    print('model: ', FLAGS.model )
    root_dir = os.getcwd()
    summaries_dir = root_dir + '/Summaries';
    fn = ''

    
############## Hyperparameters ################################################
    max_num_epochs = 20000
    dropout_val = 0.5
    learning_rate_val = float(FLAGS.lr)
    use_batch_norm = FLAGS.bn
    
    decay = learning_rate_val / 2e4
    use_peepholes = False; peep_str='' #only for LSTM
    BATCH_SIZE = 64
    num_train_examples = 0
    
    
    image_size = 784
    # Import data
    X_train, y_train, X_test, y_test = import_data(FLAGS, num_train_examples)

    #Build different model types
    ps = Parser(FLAGS.model)
    
    if ps._cell_type=='L':
        cell = tf.nn.rnn_cell.LSTMCell(ps._num_units, use_peepholes=use_peepholes)
    elif ps._cell_type=='G':
        cell = tf.nn.rnn_cell.GRUCell(ps._num_units)

    if ps._num_layers>1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * ps._num_layers)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    if ps._task=='P1':
        x = tf.placeholder(tf.float32, [None, image_size, 1], name='x')
        y_ = tf.placeholder(tf.int32, [None], name='y_')
        y, cross_entropy = build_network_task1(x, ps._num_units, cell, y_, use_batch_norm ) 
    elif ps._task=='P2':
        x = tf.placeholder(tf.float32, [None, image_size-1, 1], name='x')
        y_ = tf.placeholder(tf.int32, [None, image_size-1], name='y_')
        y, cross_entropy = build_network_task2(x, ps._num_units, cell, y_ )     

        
#     train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#     train_step = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(cross_entropy)
    

    # Test trained model
    if ps._task=='P1':
        argm_y = tf.to_int32( tf.argmax(y, 1) )
        correct_prediction = tf.equal(argm_y, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('CrossEntropy', cross_entropy)
    path_arr = [FLAGS.model, "drop{:.1f}".format(dropout_val), peep_str, 'bs' + str(BATCH_SIZE),
                "lr{:.2g}".format(learning_rate_val)]
    if use_batch_norm:
        path_arr.append('bn')

#     show_all_variables()

    if ps._task=='P1':
        db = DataBatcher(X_train, y_train)
    elif ps._task=='P2':
        db = DataBatcher(X_train[:,:783], y_train)

            
    with tf.Session() as sess:  
        tf.global_variables_initializer().run()    
                       
        if FLAGS.sm is not None or FLAGS.eval: #Restore saved model   
#             fn='P1_1x128_L_drop0.5__bs64_lr2e-06_nt0_47'
            if FLAGS.sm is not None:
                fn=FLAGS.sm
            else:
                fn= FLAGS.model
#             print("\n".join([n.name for n in tf.get_default_graph().as_graph_def().node]))

#             model_file_name = root_dir + '/model/' + fn + '.ckpt'    
            model_file_name = root_dir + '/final_models/' + fn + '.ckpt'  
            print('loading model from: ' + model_file_name)  
            saver2restore = tf.train.Saver(write_version=1)
            saver2restore.restore(sess, model_file_name)
            
        if not FLAGS.eval: #Train new model
            # Merge all the summaries and write them out to file
            merged = tf.summary.merge_all()
            
            summary_file_name = '/'.join(path_arr)
            dir_name = summaries_dir + '/' + summary_file_name;
            train_writer = tf.summary.FileWriter(dir_name + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(dir_name + '/test')
            
            conv_tester = ConvergenceTester(0.0001, lookback_window=5, decreasing=True) #stop if converged to within 0.05%
            lrs = LearningRateScheduler(decay)
            ntrain = X_train.shape[0]
            
            
            print('Starting Training.........')
            
            # Train
            for epoch in range(max_num_epochs):
                start = timer()
           
                for i in range(ntrain // BATCH_SIZE):
                    learning_rate_val = lrs.get_learning_rate(epoch, learning_rate_val)
                    batch_xs, batch_ys = db.next_batch(BATCH_SIZE)
                    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, 
                                                    learning_rate: learning_rate_val, keep_prob: dropout_val})
                end = timer()
             
                if epoch % 1 == 0: #calc intermediate results
     
                    if ps._task=='P1':
                        train_accuracy, train_loss, train_summary = sess.run([accuracy, cross_entropy, merged], feed_dict={x: X_train[:5000], y_: y_train[:5000], learning_rate: learning_rate_val, keep_prob: 1.0})                                      
                        test_accuracy, test_loss, test_summary = sess.run([accuracy, cross_entropy, merged], feed_dict={x: X_test, y_: y_test, learning_rate: learning_rate_val, keep_prob: 1.0})
                        print("epoch %d, tr:te accuracy %g : %g loss %g : %g lr %g et %s" % (epoch, train_accuracy, test_accuracy, train_loss, test_loss, learning_rate_val, str(datetime.timedelta(seconds=end-start))))
                                   
                    elif ps._task=='P2':
                        train_loss, train_summary = sess.run([cross_entropy, merged], feed_dict={x: X_train[:1000,:783], y_: y_train[:1000], learning_rate: learning_rate_val, keep_prob: 1.0})                                      

                        nsplits = 20
                        n = int(X_test.shape[0] / nsplits)
                        test_losses = np.zeros((nsplits))
                        for i in range(nsplits):
                            s = i * n
                            e = (i+1) * n
                            test_losses[i] = sess.run(cross_entropy, feed_dict={x: X_test[s:e,:783], y_: y_test[s:e], keep_prob: 1.0})
                        test_loss = np.mean(test_losses)                              
                        test_summary = sess.run(merged, feed_dict={x: X_test[:,:783], y_: y_test, learning_rate: learning_rate_val, keep_prob: 1.0})
                        print("epoch %d, loss %g : %g lr %g et %s" % (epoch, train_loss, test_loss, learning_rate_val, str(datetime.timedelta(seconds=end-start))))
                                   
                    if np.isnan(train_loss) or np.isnan(test_loss):
                        exit()
                    
                    train_writer.add_summary(train_summary, i)
                    test_writer.add_summary(test_summary, i)
                    
                    if conv_tester.has_converged(test_loss):
                        print('converged after ', epoch, ' epochs')
                        break
                        
                #save trained model
                model_file_name = '_'.join(path_arr)+'_'+ str(epoch) #write every epoch
                save_model(sess, model_file_name, root_dir)
#             exit()
            
        if FLAGS.eval and True:
            #print final results        
            nsplits = 10
            n = int(X_train.shape[0] / nsplits)
            train_losses = np.zeros((nsplits))
            if ps._task=='P1':
                train_accuracies = np.zeros((nsplits))
                for i in range(nsplits):
                    start = i * n
                    end = (i+1) * n
                    train_losses[i], train_accuracies[i] = sess.run([cross_entropy, accuracy], feed_dict={x: X_train[start:end], y_: y_train[start:end], keep_prob: 1.0})                                      
                test_loss, test_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: X_test, y_: y_test, keep_prob: 1.0})                                      
                print("\ntrain loss %.6f train accuracy %.6f" % (np.mean(train_losses), np.mean(train_accuracies)))
                print("\ntest loss %.6f test accuracy %.6f" % (test_loss, test_accuracy))
            elif ps._task=='P2':
                for i in range(nsplits):
                    start = i * n
                    end = (i+1) * n
                    train_losses[i] = sess.run(cross_entropy, feed_dict={x: X_train[start:end,:783], y_: y_train[start:end], keep_prob: 1.0})                                      
                test_loss = sess.run(cross_entropy, feed_dict={x: X_test[:,:783], y_: y_test, keep_prob: 1.0})                                      
                print("\ntrain loss %.6f" % np.mean(train_losses))
                print("\ntest loss %.6f" % (test_loss))
            exit()

#         task_2(sess, x, y, y_, X_train, y_train, X_test, y_test, keep_prob, fn, root_dir)

        task_3(y, x, y_, keep_prob, sess, root_dir, fn)

    
# Task 2: pixel prediction
def task_2(sess, x, y, y_, X_train, y_train, X_test, y_test, keep_prob, fn, root_dir):

    inpaintings_data__filename = root_dir + '/inpaint_data/' + fn + '.p'
                
    calc_inpaintings = False
    if calc_inpaintings:
        nsamples = 100
        mask_length = 300
        db2 = DataBatcher(X_test, y_test)
        ground_truth_images, batch_ys = db2.next_batch(nsamples)
                    
        pixel_preds = np.zeros((nsamples, mask_length), dtype='float32')            
        pixel_gt = np.zeros((nsamples, mask_length), dtype='float32')            
        pixel_idx = ground_truth_images.shape[1] - mask_length - 1                  
        for i in range(mask_length):
            pred = sess.run(y, feed_dict={x: ground_truth_images[:,:783], y_: batch_ys, keep_prob: 1.0})
    
            pixel_preds[:, i] = pred[:, pixel_idx]
            pixel_idx += 1
            pixel_gt[:, i] = ground_truth_images[:, pixel_idx, 0]
#             ground_truth_images[:, pixel_idx, 0] = pixel_preds[:, i]
        
        samples = generate_in_paintings(pixel_preds, 10) # 10 x 100 x 300

        pi.dump( (ground_truth_images, pixel_preds, pixel_gt, samples), open( inpaintings_data__filename, "wb" ) )
        exit()
    else:
        (ground_truth_images, pixel_preds, pixel_gt, samples) = pi.load( open( inpaintings_data__filename, "rb" ) )

    saved_images_filename = root_dir + '/images/' + fn
    get_cross_entropy(samples, ground_truth_images, pixel_preds, pixel_gt, saved_images_filename)    
        
def generate_in_paintings(model_probs, nsamples):
    samples = np.random.uniform(size=(nsamples,model_probs.shape[0],model_probs.shape[1])) < model_probs
    return samples.astype('float32')
        
def generate_possible_ips(m, missing_pixels, images_ip, i, num_missing_pixels):

    mp = missing_pixels[m]
    for a in range(2):
        images_ip[i,a,mp] = a
        if m < num_missing_pixels:
            generate_possible_ips(m+1, missing_pixels, images_ip, i)
            
        
def get_in_paintings(images, gt_images, num_in_paintings):
    #generate 2 or 16 (all possible) in-paintings
    images_ip = np.zeros( (images.shape[0], num_in_paintings, images.shape[1]) ) 
    num_missing_pixels = int(np.log2(num_in_paintings))
    missing_pixels = np.zeros( (gt_images.shape[0], num_missing_pixels), dtype=np.int)
    
    #build LUT to hold all possible inpaintings
    lut = np.zeros((num_in_paintings, num_missing_pixels))
    for s in range(num_in_paintings):
        for m in range(num_missing_pixels):
            lut[s, m] = min(1, s & 2**m) 
    
    for i, img in enumerate(images):
        mps = np.where(img<0) #find the missing pixels in this image
        missing_pixels[i,:] = mps[0] #find the missing pixels in this image
        
        for s in range(num_in_paintings):
            images_ip[i,s,:] = np.copy(img) #make 1 copy for each possible in-painting
            
            for m, mp_idx in enumerate(missing_pixels[i,:]): 
                images_ip[i,s,mp_idx] = lut[s, m]
        
    return images_ip, missing_pixels    
        
def mean_xent_missing_pixels(mean_xent, missing_pixels):
    
    missing_pixels = missing_pixels.flatten()
    xent = 0;
    for mp in missing_pixels:
        xent += mean_xent[mp]
    return xent/len(missing_pixels)

def idx_of_gt(images_ip, gt_image, num_in_paintings):
    for s in range(num_in_paintings):
        if np.array_equal(images_ip[s,:], gt_image):
            return s
    return None
        
def calc_log_p(preds, images_ip, gt_images, missing_pixels):
    
    num_in_paintings = preds.shape[1] #2 or 16
    num_missing_pixels = np.log2(num_in_paintings)
    nimgs = preds.shape[0]              #1000
    npixels = preds.shape[2]            #783
    num_correct = 0
    
    xent_gt = np.zeros((nimgs, npixels), dtype='float32')
    xent_ip = np.zeros((nimgs, num_in_paintings, npixels), dtype='float32')
    xent_ip_mp = np.zeros((nimgs, npixels), dtype='float32')
    
    for i in range(nimgs):
        for s in range(num_in_paintings):
            for p in range(npixels):
                log_prob = np.log(preds[i,s,p])
                log_one_minus_prob = np.log(1 - preds[i,s,p])
                
                xent_ip[i,s,p] -= images_ip[i,s,p+1] * log_prob + \
                                            (1-images_ip[i,s,p+1]) * log_one_minus_prob

        mean_arr = np.mean(xent_ip[i,:,:], axis=1)
        max_prob_idx = np.argmin( mean_arr )
        
        missing_pixels_in_most_probable_ip = images_ip[i, max_prob_idx, missing_pixels[i]]
        missing_pixels_in_gt = gt_images[i, missing_pixels[i]]
        idx_gt = idx_of_gt(images_ip[i], gt_images[i], num_in_paintings)

        for p in range(npixels):        
            xent_ip_mp[i,p] = xent_ip[i,max_prob_idx,p]
            xent_gt[i,p] = xent_ip[i,idx_gt,p]
    
        for mp_ip, mp_gt in zip(missing_pixels_in_most_probable_ip, missing_pixels_in_gt):
            if(mp_ip == mp_gt):
                num_correct += 1
                
                
    mean_xent = np.mean(xent_gt, axis=0) 
    mx_mp = mean_xent_missing_pixels(mean_xent, missing_pixels)
    print('mean_xent_missing_pixels=%g' %(mx_mp))

    fn = 'inpainting_' + str(int(num_missing_pixels)) + '_mps.npy'
    np.save(open( fn, "wb" ), xent_ip_mp )
    xent_ip_mp_test = np.load(open( fn, "rb" ) )
    
    fn = 'data_' + str(int(num_missing_pixels)) + '_mps.pi'
    pi.dump( (np.mean(xent_ip_mp, axis=1), np.mean(xent_gt, axis=1)), open( fn, "wb" ) )
#     (preds_rs, images_ip_784, gt_images, missing_pixels) = pi.load( open( inpaintings_data__filename, "rb" ) )

#     mean_xent = np.append(mean_xent, 0) 
#     mean_xent = np.reshape(mean_xent, (28,28))
#     plt.figimage(mean_xent)
#     plt.show()
#   
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     X = range(28)
#     Y = range(28)
#     X, Y = np.meshgrid(X, Y)
#     surf = ax.plot_surface(X, Y, mean_xent, cmap=cm.coolwarm,
#                            linewidth=0, antialiased=False)
#       
#     # Add a color bar which maps values to colors.
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#     ax.set_xlabel('horizontal pixel index')
#     ax.set_ylabel('vertical pixel index')
#     ax.set_zlabel('Cross entropy')
#     plt.show()

    print('num correct=%g, %%correct=%g' %(num_correct, 100*num_correct/(nimgs*num_missing_pixels)))
    print('xent_gt=%g, xent_ip_max_prob=%g' %(np.mean(xent_gt), np.mean(xent_ip_mp)))


def t_test_one_vs_2x2():
    
    num_missing_pixels = 1.0
    fn = 'data_' + str(num_missing_pixels) + '_mps.pi'
    (xent_ip_mp_1, xent_gt_1) = pi.load( open( fn, "rb" ) )

    num_missing_pixels = 4.0
    fn = 'data_' + str(num_missing_pixels) + '_mps.pi'
    (xent_ip_mp_4, xent_gt_4) = pi.load( open( fn, "rb" ) )
    
    _, p_value = ttest_ind(xent_gt_1, xent_gt_4, axis=None)    
    print('t-test p-value ground truth 1 vs 2x2=%g' %(p_value))

    _, p_value = ttest_ind(xent_ip_mp_1, xent_ip_mp_4, axis=None)
    print('t-test p-value most probable in-painting 1 vs 2x2=%g' %(p_value))
    

def task_3(y, x, y_, keep_prob, sess, root_dir, fn):   

#     t_test_one_vs_2x2()

    num_missing_pixels = 4; #1 or 4
    inpaintings_data__filename = root_dir + '/inpaint_data/' + fn + '_task3_' + str(num_missing_pixels) + '.p'

    calc_inpaintings = False
    if calc_inpaintings:
        if num_missing_pixels==1:
            dataset = np.load(root_dir + '/one_pixel_inpainting.npy');num_in_paintings = 2    
        elif num_missing_pixels==4:
            dataset = np.load(root_dir + '/2X2_pixels_inpainting.npy');num_in_paintings = 16
        
        images    = np.asarray(dataset[0])
        gt_images = np.asarray(dataset[1]) 
        
        images_ip_784, missing_pixels = get_in_paintings(images, gt_images, num_in_paintings)
        
        images_ip = np.delete(images_ip_784, images_ip_784.shape[2]-1, 2) #remove last pixel
        
        images_ip_rs = np.reshape(images_ip, [images_ip.shape[0] * images_ip.shape[1], images_ip.shape[2]])
        images_ip_rs = np.expand_dims(images_ip_rs, axis=2)
            
        preds = sess.run(y, feed_dict={x: images_ip_rs, keep_prob: 1.0})
    
        preds_rs = tf.reshape(preds, images_ip.shape).eval()
        
        pi.dump( (preds_rs, images_ip_784, gt_images, missing_pixels), open( inpaintings_data__filename, "wb" ) )
    else:
        (preds_rs, images_ip_784, gt_images, missing_pixels) = pi.load( open( inpaintings_data__filename, "rb" ) )
    
    calc_log_p(preds_rs, images_ip_784, gt_images, missing_pixels)
    
    
#     for SampleID in np.random.randint(nSamples,size=3):
#         plt.figure()
#         plt.subplot(1,2,1)
#         plt.imshow(np.reshape(gt_images[SampleID],(28,28)), interpolation='None',vmin=-1, vmax=1)
#         plt.title("GT image")
#         plt.subplot(1,2,2)
#         plt.imshow(np.reshape(images[SampleID],(28,28)), interpolation='None',vmin=-1, vmax=1)
#         plt.title("One pixel missing image")
#         plt.savefig("sample_"+str(SampleID)+"_one_pixel_inpainting.png")
    
    
    
    
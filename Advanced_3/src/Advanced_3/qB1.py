import gym
import os
import numpy as np
import random
import tensorflow as tf
import pickle as pi
import matplotlib.pyplot as plt
import datetime#, time
import sys
from scipy.misc import toimage, imshow
from skimage import color
from skimage import exposure
from skimage.transform import resize, downscale_local_mean, rescale
from scipy.misc import imresize
from timeit import default_timer as timer
from PIL import Image
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from pycallgraph import Config
from tensorflow.core.framework import summary_pb2

root_dir = 'C:/Users/Dave/Documents/GI13-Advanced/Assignment3';
summaries_dir = root_dir + '/Summaries';
save_dir = root_dir;

MAX_EPISODE_LENGTH = 1e6
MAX_AGENT_STEPS = 1e6
GAMMA = 0.99
NFRAMES = 4
QCHECK_NSTATES=1024
MAX_BUFFER_LENGTH = 1e5
BUFFER_DELETE_LENGTH = 50000

def save_model(session, model_name, root_dir):
    if not os.path.exists(root_dir + '/model/'):
        os.mkdir(root_dir + '/model/')
    saver = tf.train.Saver(write_version=1)
    saver.save(session, root_dir + '/model/' + model_name +'.ckpt')

class AllEpisodeData:
    
    def __init__(self):
        self._s_t = []
        self._a_t = []
        self._r_t1 = []
        self._s_t1 = []
        return
    
    def num_samples(self):
        return len(self._s_t)
    
    def get_first_n_samples(self, n):
        return self.get_data_from_idxs(range(n))
        
    def sample(self, batch_size):
        idxs = np.random.randint(self.num_samples(), size=batch_size)
        return self.get_data_from_idxs(idxs)
    
    def get_data_from_idxs(self, idxs):
        s_ts = []
        a_ts = []
        r_t1s = []
        s_t1s = []
        
        for i in idxs:
            s_ts.append(self._s_t[i])
            a_ts.append(self._a_t[i])
            r_t1s.append(self._r_t1[i])
            s_t1s.append(self._s_t1[i])
        return s_ts, a_ts, r_t1s, s_t1s
    
    def all_data(self):
        return self._s_t, self._a_t, self._r_t1, self._s_t1

    def add_data(self, s_t, frames, a_t, r_t1):
        s_t1 = np.transpose(frames[1:], axes=(1,2,0))
        
        self._s_t.append(s_t)
        self._a_t.append(a_t)
        self._r_t1.append(r_t1)
        self._s_t1.append(s_t1)
        
        if self.num_samples() > MAX_BUFFER_LENGTH:
            del self._s_t[:BUFFER_DELETE_LENGTH]
            del self._a_t[:BUFFER_DELETE_LENGTH]
            del self._r_t1[:BUFFER_DELETE_LENGTH]
            del self._s_t1[:BUFFER_DELETE_LENGTH]

        
def weight_variable(shape):
#     initial = tf.constant(0.0, shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, 'W')

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
#     initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, 'b')
 
def build_net(n_actions):
    
    x = tf.placeholder(tf.uint8, [None,28,28,4], 'x')
    x_flt = tf.cast(x, tf.float32)
    
    W_conv1 = weight_variable([6, 6, 4, 16])
    b_conv1 = bias_variable([16])
    
    h_conv1 = tf.nn.conv2d(x_flt, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1
    h_relu1 = tf.nn.relu(h_conv1)
    
    W_conv2 = weight_variable([14, 14, 16, 32])
    b_conv2 = bias_variable([32])
    
    h_conv2 = tf.nn.conv2d(h_relu1, W_conv2, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
    h_relu2 = tf.nn.relu(h_conv2)
    
    h_flattened = tf.reshape(h_relu2, [-1, 7 * 7 * 32])

    W_4 = weight_variable([7 * 7 * 32, 256])
    b_4 = bias_variable([256])
    
    h_4 = tf.nn.relu(tf.matmul(h_flattened, W_4) + b_4)
    
    W_5 = weight_variable([256, n_actions])
    b_5 = bias_variable([n_actions])

    y = tf.matmul(h_4, W_5) + b_5

#     l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
#     total_loss = loss + LAMBDA * l2_loss
#     tf.summary.scalar('residual', residual)

    return x, y

def pre_process_state(frame):
#     toimage(frame).show()
#     img_small = resize(frame, (28,28))
#     toimage(img_small).show()
    img = color.rgb2gray(frame)
#     toimage(img).show()

#     img_small = downscale_local_mean(img, (7,5))
    img_small = resize(img, (110,84), order=0)
    toimage(img_small).show()
    img_small = img_small[:84,:]
    toimage(img_small).show()
    
    
    img_small = resize(img, (28,28), order=0)
#     img_small = resize(img, (37,28), order=0)
#     v_min, v_max = np.percentile(img_small, (20, 80))
#     better_contrast = exposure.rescale_intensity(img_small, in_range=(v_min, v_max))
#     toimage(better_contrast).show()
#     toimage(img_small).show()
#     img_small = imresize(img, (28,28))
#     toimage(img_small).show()
    img_small = np.multiply(img_small, 256)
    img_uint = img_small.astype(np.uint8)
#     toimage(img_uint).show()
    return img_uint
    
def train_net(sess, replay_buffer, max_training_samples, 
              batch_size, Qfunc, Qfunc_tar, x, x_tar, residual, target,
              train_step, loss, train_writer, total_agent_steps):
    
    num_training_samples = replay_buffer.num_samples()
    if max_training_samples > num_training_samples:
        return None, None
    
    for _ in range(int( min(max_training_samples,num_training_samples) / batch_size)):

        s_ts, a_ts, r_t1s, s_t1s = replay_buffer.sample(batch_size=batch_size)
        Qfunc_s_t,Qfunc_s_t1 = sess.run([Qfunc,Qfunc_tar], feed_dict={x: s_ts, x_tar: s_t1s})
        max_Q = np.amax(Qfunc_s_t1, axis=1)
        target_vals = Qfunc_s_t
            
        for target_val, action, r, mQ, s_t1 in zip(target_vals, a_ts, r_t1s, max_Q, s_t1s):
            if np.sum(s_t1[:,:,NFRAMES-1]) == 0:
                target_val[action] = r
            else:
                target_val[action] = r + GAMMA * mQ

        residual_val, loss_val, _ = sess.run([residual, loss, train_step], 
                                    feed_dict={x: s_ts, target: target_vals})
        
    residual_val = np.asscalar(np.mean(np.fabs(residual_val)))
    loss_val = np.asscalar(np.mean(loss_val))

    res = summary_pb2.Summary.Value(tag="residual", simple_value=residual_val)
    lo = summary_pb2.Summary.Value(tag="loss", simple_value=loss_val)
    summary = summary_pb2.Summary(value=[lo, res])
    train_writer.add_summary(summary, total_agent_steps)
    return

def check_Q(sess, replay_buffer, Qfunc, Qfunc_tar, x, x_tar, residual):
    
    nsamples = replay_buffer.num_samples()
    if nsamples >= QCHECK_NSTATES:
        n = QCHECK_NSTATES
    else:
        n = nsamples
        
    s_ts, _,_,_ = replay_buffer.get_first_n_samples(n)

    Qfunc_s_t = sess.run(Qfunc, feed_dict={x: s_ts})
 
    max_Q = np.amax(Qfunc_s_t, axis=1)
    return np.asscalar(np.mean(max_Q))

def evaluate(sess, env, n_actions, Qfunc, x, train_writer, total_agent_steps):
    
    n_episodes = 100
    rewards = np.zeros((n_episodes))
    rewards_undiscounted = np.zeros((n_episodes))
    
    for episode in range(n_episodes):
        s_t = env.reset()
        discount_factor = 1
        frames = [] #holds 4 or 5 frames
        cum_rewards = 0
        cum_rewards_undiscounted = 0
    
        #initialize first 4 frames
        for t in range(NFRAMES):
            a_t = random.randint(0,n_actions-1)
            frame_t1, _, done, _ = env.step(a_t)
            frames.append(pre_process_state(frame_t1))
        
        while t < MAX_EPISODE_LENGTH:
            s_t = np.transpose(frames, axes=(1,2,0)) #4 entries
            
            #get greedy action
            s_t = np.expand_dims(s_t, axis=0)
            Qfunc_s_t = sess.run(Qfunc, feed_dict={x: s_t})
            max_a = np.argmax(Qfunc_s_t, axis=1)
            a_t = max_a[0]
            
            frame_t1, r_t1, done, _ = env.step(a_t)
            frames.append(pre_process_state(frame_t1)) #4->5 entries
#                 print('r {}'.format(r_t1))
            r_t1 = clipped_reward(r_t1)
    
            cum_rewards += discount_factor*r_t1
            cum_rewards_undiscounted += r_t1

            if done:
                break

            discount_factor *= GAMMA
            del frames[0] #5->4 entries
            t += 1

        rewards[episode] = cum_rewards    
        rewards_undiscounted[episode] = cum_rewards_undiscounted 

    reward_val = np.asscalar(np.mean(rewards))   
    rewarundisc_val = np.asscalar(np.mean(rewards_undiscounted))   
    
    etr = summary_pb2.Summary.Value(tag="eval test reward", simple_value=reward_val)
    etru = summary_pb2.Summary.Value(tag="eval test reward undiscounted", simple_value=rewarundisc_val)
    summary = summary_pb2.Summary(value=[etr, etru])
    train_writer.add_summary(summary, total_agent_steps)
    print('{} evaluation rewards {:.6f} rewards_undisc {:.6f}'.format(total_agent_steps, reward_val, rewarundisc_val))
    return
    

def run_net(FLAGS):
#     plot_data(MAX_EPISODES)
#     game = 'Boxing-v3'
#     game = 'Pong-v3'
    game = 'MsPacman-v3'
    print('Running ' + game)
    env = gym.make(game)

    n_actions = env.action_space.n
    print(env.action_space)
    print(env.observation_space)
#     print(env.observation_space.high)
#     print(env.observation_space.low)

    learning_rate=0.001
    batch_size = 32

    x, Qfunc = build_net(n_actions)
    network_params = tf.trainable_variables()
    x_tar, Qfunc_tar = build_net(n_actions)
    target_network_params = tf.trainable_variables()[len(network_params):]

    update_target_op = [target_network_params[i].assign(network_params[i]) 
                            for i in range(len(target_network_params))]

    target = tf.placeholder(tf.float32, [None, n_actions], 'target')
    residual = tf.sub(target, Qfunc)
    loss = 0.5 * tf.reduce_mean(tf.square(residual))

    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    max_training_samples = 0
    target_update_interval = 5000
    evaluation_interval = 50000
    train_step_interval = 10
    num_batches_per_episode = 1
    MAX_EPISODES = 10000
    EPSILON = 0.1
    save_model_interval = 100 #in epochs

    path_arr = [FLAGS.model, "tui{}".format(target_update_interval), game]

    runB1 = False
    runB2 = True
    
    if runB1 is True: #set parameters to get results for part B.1
        MAX_EPISODES = 100
        EPSILON = 1 #all actions random 
        evaluation_interval = sys.maxsize
        target_update_interval = sys.maxsize
        train_step_interval = sys.maxsize
        save_model_interval = sys.maxsize
    elif runB2 is True: #set parameters to get results for part B.2
        MAX_EPISODES = 100
        EPSILON = 0 #all actions greedy
        evaluation_interval = sys.maxsize
        target_update_interval = sys.maxsize
        train_step_interval = sys.maxsize
        save_model_interval = sys.maxsize

    with tf.Session() as sess:    
        episode_length = np.zeros((MAX_EPISODES))
        rewards = np.zeros((MAX_EPISODES))
#         rewards_undiscounted = np.zeros((n_episodes))
        merged = tf.summary.merge_all()
        dir_name = summaries_dir + '/B1/' + str(random.randint(0,99)) + '/lRate_' + str(learning_rate) ;
        train_writer = tf.summary.FileWriter(dir_name + '/train', sess.graph)

        replay_buffer = AllEpisodeData()
        
        tf.global_variables_initializer().run()    
        total_agent_steps = 0
        start = timer()

        for episode in range(MAX_EPISODES):
#             config = Config(max_depth=3)
#             graphviz = GraphvizOutput(output_file='filter_exclude.png')
#             with PyCallGraph(output=graphviz, config=config):

            if total_agent_steps >= MAX_AGENT_STEPS:
                break;

            s_t = env.reset()
            discount_factor = 1
            frames = [] #holds 4 or 5 frames
            cum_rewards = 0
            
            #initialize first 4 frames
            for t in range(NFRAMES):
                a_t = random.randint(0,n_actions-1)
                frame_t1, _, done, _ = env.step(a_t)
                frames.append(pre_process_state(frame_t1))
            
            while t < MAX_EPISODE_LENGTH:
#                 start = timer()
#                 env.render()
                    
                s_t = np.transpose(frames, axes=(1,2,0)) #4 entries
                a_t = get_epsilon_greedy_action(sess, Qfunc, x, s_t, EPSILON, n_actions)
                frame_t1, r_t1, done, _ = env.step(a_t)
                frames.append(pre_process_state(frame_t1)) #4->5 entries
#                 if r_t1 > 0 or r_t1 < 0:
#                     print('{} r {}'.format(t, r_t1))
                r_t1 = clipped_reward(r_t1)
        
                cum_rewards += discount_factor*r_t1
    
                if done:
                    frames[NFRAMES] = np.zeros((28,28))
                    replay_buffer.add_data(s_t, frames, a_t, r_t1)
                    break

                replay_buffer.add_data(s_t, frames, a_t, r_t1)
                
                discount_factor *= GAMMA
                del frames[0] #5->4 entries
                t += 1
                total_agent_steps += 1

                max_training_samples = batch_size * num_batches_per_episode
                
                #test for the different updates
                if t % train_step_interval == 0 and t>0:
                    train_net(sess, replay_buffer, max_training_samples, batch_size,
                              Qfunc, Qfunc_tar, x, x_tar, residual, target, 
                              train_step, loss, train_writer, total_agent_steps)
                
                if total_agent_steps % evaluation_interval == 0 and total_agent_steps>0: #
                    evaluate(sess, env, n_actions, Qfunc, x, train_writer, total_agent_steps)

                if total_agent_steps % target_update_interval==0 and total_agent_steps>0: #update the target net
                    print('{} updating target'.format(total_agent_steps))
                    sess.run(update_target_op)

#                 end = timer()
#                 duration.append(end-start)
            
            episode_length[episode] = t+1
            rewards[episode] = cum_rewards
            Qcheck_val = check_Q(sess, replay_buffer, Qfunc, Qfunc_tar, x, x_tar, residual)

            cr = summary_pb2.Summary.Value(tag="cum_rewards", simple_value=cum_rewards)
            l = summary_pb2.Summary.Value(tag="length", simple_value=t+1)
            qc = summary_pb2.Summary.Value(tag="Qcheck", simple_value=Qcheck_val)
            summary = summary_pb2.Summary(value=[cr, l, qc])
            train_writer.add_summary(summary, total_agent_steps)

            if episode % 5 ==0:
                end = timer()
                total_duration = end-start
                time_per_episode = total_duration / (episode+1)
                print('{} len {} cum_rewards {:.3f} Qc {:.2f} as {:g} t/ep {:.1f}'.format(episode, t+1, 
                        cum_rewards, Qcheck_val,total_agent_steps, time_per_episode))

            if episode % save_model_interval == 0 and episode>0:
                #save trained model
                model_file_name = '_'.join(path_arr)+'_'+ str(episode) #write every episode
                save_model(sess, model_file_name, root_dir)

        print('length {:.6f} std: {:.6f}'.format(
                                            np.asscalar(np.mean(episode_length)), 
                                            np.asscalar(np.std(episode_length))))
        print('cum reward {:.6f} std: {:.6f}'.format(
                                            np.asscalar(np.mean(rewards)), 
                                            np.asscalar(np.std(rewards))))

#         pi.dump( (episode_length, residuals, rewards), open( model_file_name+'.pi', "wb" ) )
        
#         (episode_length, losses, rewards) = pi.load( open( 'qA4_data', "rb" ) )
    
#         x_s = np.arange(0, MAX_EPISODES, 1)
#         
#         plt.figure(1, figsize=(12, 8))
#         plt.subplot(311)
#         plt.plot(x_s, np.mean(residuals, axis=0), 'r-')
#         plt.title("Double Q: " + 
#                   ' mean episode length: ' + str(np.asscalar(np.mean(episode_length))))
#         plt.ylabel('Absolute Residual')
#         
#         plt.subplot(312)
#         plt.plot(x_s, np.mean(episode_length, axis=0), 'b-')
#         plt.ylabel('Episode length')
#         
#         plt.subplot(313)
#         plt.plot(x_s, np.mean(rewards, axis=0), 'g-')
#         plt.xlabel('episode #')
#         plt.ylabel('Return')
#         plt.tight_layout()
#         plt.savefig('Fig_' + model_file_name)
#         plt.show() 


def plot_data(n_episodes):
    (episode_length_f, residuals_f, rewards_f) = pi.load( open( 'qA8_rep10_DQFalse_0.pi', "rb" ) )
    (episode_length_t, residuals_t, rewards_t) = pi.load( open( 'qA8_rep10_DQTrue_0.pi', "rb" ) )

    x_interval = 20    

    x_s = np.arange(0, n_episodes, x_interval)
    
    fig = plt.figure(1, figsize=(15, 5))
#     plt.title("No hidden units: "+str(n_hidden))
    plt.subplot(311)
    plt.plot(x_s, np.mean(residuals_t, axis=0)[0::x_interval], 'r-',label='Double Q learning')
    plt.plot(x_s, np.mean(residuals_f, axis=0)[0::x_interval], 'b--',label='no Double Q learning', linewidth=0.5)
    plt.ylabel('Absolute Residual')
    plt.legend(loc=0, borderaxespad=1.)
    
    print('residuals: {:.1f} {:.1f}'.format(np.asscalar(np.mean(residuals_t)),
                               np.asscalar(np.mean(residuals_f))))
    print('episode length: {:.1f} {:.1f}'.format(np.asscalar(np.mean(episode_length_t)),
                               np.asscalar(np.mean(episode_length_f))))
    print('returns: {:.3f} {:.3f}'.format(np.asscalar(np.mean(rewards_t)),
                               np.asscalar(np.mean(rewards_f))))
    plt.subplot(312)
    plt.plot(x_s, np.mean(episode_length_t, axis=0)[0::x_interval], 'r-',label='frozen target update')
    plt.plot(x_s, np.mean(episode_length_f, axis=0)[0::x_interval], 'b--',label='no frozen target', linewidth=0.5)
    plt.ylabel('Episode length')
    
    plt.subplot(313)
    plt.plot(x_s, np.mean(rewards_t, axis=0)[0::x_interval], 'r-',label='frozen target update')
    plt.plot(x_s, np.mean(rewards_f, axis=0)[0::x_interval], 'b--',label='no frozen target', linewidth=0.5)
    plt.xlabel('episode #')
    plt.ylabel('Return')
    
#     fig.legend((res20,res100,res1000), ('30 hidden units','100 hidden units','1000 hidden units'), 'lower right')
    plt.tight_layout()
    plt.show() 
    exit()
     
def clipped_reward(r_t1):
    if r_t1 == 0:
        cr = 0
    elif r_t1 < 0:
        cr = -1
    else:
        cr = 1
    return cr
        
def get_epsilon_greedy_action(sess, Qfunc, x, s, epsilon, n_actions):
    if random.random() < epsilon: #explore
        return random.randint(0,n_actions-1)
    else: #exploit
        s = np.expand_dims(s, axis=0)
        Qfunc_s_t = sess.run(Qfunc, feed_dict={x: s})
        max_a = np.argmax(Qfunc_s_t, axis=1)
        return max_a[0]

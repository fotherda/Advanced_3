import gym
import numpy as np
import random
import tensorflow as tf
import pickle as pi
import os

from Advanced_1.convergenceTester import ConvergenceTester

root_dir = 'C:/Users/Dave/Documents/GI13-Advanced/Assignment3';
summaries_dir = root_dir + '/Summaries';
# save_dir = root_dir + '/SavedVariables';
save_dir = root_dir;

MAX_EPISODES = 2000
MAX_EPISODE_LENGTH = 300
GAMMA = 0.99


episode_length = np.zeros(MAX_EPISODES)
returns = np.zeros(MAX_EPISODES)

# register(
#     id='CartPole-v0',
#     entry_point='gym.envs.classic_control:CartPoleEnv',
#     tags={'wrapper_config.TimeLimit.max_episode_steps': 500},
#     reward_threshold=1000.0,
# )
def save_model(session, model_name, root_dir):
    if not os.path.exists(root_dir + '/model/'):
        os.mkdir(root_dir + '/model/')
    saver = tf.train.Saver(write_version=1)
    save_path = saver.save(session, root_dir + '/model/' + model_name +'.ckpt')

class SingleEpisodeData:
    
    def __init__(self):
        self._s_ts = []
        self._a_ts = []
        self._sa_ts = []
        self._r_t1s = []
        self._s_t1s = []
        self._sa0_t1s = []
        self._sa1_t1s = []
        return
    
    def add_data(self, s_t, a_t, r_t1, s_t1):
        self._s_ts.append(s_t)
        self._a_ts.append(a_t)
        self._r_t1s.append(r_t1)
        self._s_t1s.append(s_t1)
        
    def size(self):
        return len(self._data)
    
    def get_data(self):
        return (self._sa_ts, self._r_t1s, self._sa0_t1s, self._sa1_t1s )
    
    def get_s_t(self):
        return np.asarray(self._s_ts)
        
    def get_R_t1_S_t1(self):
        return (self._r_t1s, self._s_t1s)
    
    def normalize(self, med):
        self._s_ts = self.normalize_s(self._s_ts, med)
        self._s_t1s = self.normalize_s(self._s_t1s, med)

    def normalize_s(self, s, med):
        diff = s - med
        ab = np.abs(diff)
        std = np.median(ab, axis=0)
        std[std == 0] = 1
        r_val =  diff/std
        return r_val
        
    def build_input(self):
        a_t = np.expand_dims( np.transpose(self._a_ts), axis=1)
        self._sa_ts = np.concatenate( (self._s_ts, a_t), axis=1)

        episode_length = len(self._s_ts)
        a0 = np.zeros((episode_length,1))
        self._sa0_t1s = np.concatenate( (self._s_t1s, a0), axis=1)
        a1 = np.ones((episode_length,1))
        self._sa1_t1s = np.concatenate( (self._s_t1s, a1), axis=1)
        
        
        
class AllEpisodeData:
    
    def __init__(self, num_episodes):
        self._data = []
        self._s_t = []
        self._a_t = []
        self._r_t1 = []
        self._s_t1 = []
        self._sa_t = []
        self._sa0_t1 = []
        self._sa1_t1 = []
        self._num_episodes = num_episodes
        self._maxQ = None
        self._num_samples = None
        
        return
    
    def save_maxQ(self, maxQ):
        self._maxQ = maxQ
        
        
#     def sample(self, idx):
#         idx = epoch % MAX_EPISODES
#         idx = random.randrange(0, self._num_episodes)
#         return self._data[idx]

    def sample(self, batch_size):
        idxs = np.random.randint(self._num_samples, size=batch_size)
        return self._sa_t[idxs], self._r_t1[idxs], self._sa0_t1[idxs], \
            self._sa1_t1[idxs], self._s_t[idxs], self._s_t1[idxs], self._a_t[idxs], \
            self._maxQ[idxs]

    def all_data(self):
        return self._sa_t, self._r_t1, self._sa0_t1, self._sa1_t1, self._s_t, self._s_t1, self._a_t

    
    def add_data(self, s_t, a_t, r_t1, s_t1):
        self._s_t.append(s_t)
        self._a_t.append(a_t)
        self._r_t1.append(r_t1)
        self._s_t1.append(s_t1)
    
    def collect_episodes(self, env, use_saved=True):
        if use_saved:
            self._data, self._s_t, self._a_t, self._r_t1, self._s_t1, self._sa_t, \
                self._sa0_t1, self._sa1_t1 = pi.load( open( 'A3data', "rb" ) )    
        else:
            for _ in range(self._num_episodes):
                episode_data = SingleEpisodeData()
                s_t = env.reset()
                s_t1 = None
                for _ in range(MAX_EPISODE_LENGTH):
                    a_t = env.action_space.sample()
                    s_t1, _, done, _ = env.step(a_t)
                    
                    if done:
                        r_t1 = -1
                        episode_data.add_data(s_t, a_t, r_t1, s_t1)
                        self.add_data(s_t, a_t, r_t1, s_t1)
        #                 print("Episode {} finished after {} timesteps. Return {}".format(i_episode, t+1, cum_reward))
                        break
                    else:
                        r_t1 = 0
                        episode_data.add_data(s_t, a_t, r_t1, s_t1)
                        self.add_data(s_t, a_t, r_t1, s_t1)
                        s_t = s_t1
                        
                self._data.append(episode_data)
            
            self._s_t = np.array(self._s_t) 
            self._a_t = np.array(self._a_t) 
            self._r_t1 = np.array(self._r_t1) 
            self._s_t1 = np.array(self._s_t1) 
            
#             self.normalize()
            self.build_input()
            
            pi.dump( (self._data, self._s_t, self._a_t, self._r_t1, self._s_t1, self._sa_t, \
                self._sa0_t1, self._sa1_t1), open( 'A3data', "wb" ) )    
    
        self._num_samples = len(self._s_t)
        return self._num_samples
    
    def build_input(self):
        for episode_data in self._data:
            episode_data.build_input()
                
        a_t = np.expand_dims( np.transpose(self._a_t), axis=1)
        self._sa_t = np.concatenate( (self._s_t, a_t), axis=1)

        n = len(self._s_t)
        a0 = np.zeros((n,1))
        self._sa0_t1 = np.concatenate( (self._s_t1, a0), axis=1)
        a1 = np.ones((n,1))
        self._sa1_t1 = np.concatenate( (self._s_t1, a1), axis=1)
    
    
    def normalize(self):
        med = np.median(self._s_t, axis=0)       
        
        for episode_data in self._data:
            episode_data.normalize(med)
    
        
def weight_variable(shape):
#     initial = tf.constant(0.0, shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
#     initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)
 
def weight_decay(var, wd):
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('weight_losses', weight_decay)#
 
def build_net(x, n_inputs, n_outputs):
    W = weight_variable([n_inputs, n_outputs])
    b = weight_variable([n_outputs])
    y = tf.matmul(x, W)
    return y, b, W

def build_net2(x, n_inputs, n_hidden, n_outputs, use_batch_norm=True):
    
    W_1 = weight_variable([n_inputs, n_hidden])
    b_1 = bias_variable([n_hidden])   
    lin = tf.matmul(x, W_1) + b_1    

    if use_batch_norm:
        batch_mean, batch_var = tf.nn.moments(lin,[0])
        h_hat = (lin - batch_mean) / tf.sqrt(batch_var + 1e-3)
        scale = tf.Variable(tf.ones([n_hidden]))
        beta = tf.Variable(tf.zeros([n_hidden]))
        BN = scale * h_hat + beta
        h_1 = tf.nn.relu(BN)    
    else:
        h_1 = tf.nn.relu(lin)

    W_2 = weight_variable([n_hidden, n_outputs])
    b_2 = bias_variable([n_outputs])
    y = tf.matmul(h_1, W_2) + b_2
    
    return y, b_2, W_2


def run_net(learning_rate, FLAGS):
    env = gym.make('CartPole-v0')

    if FLAGS.nhidden is not None:
        n_hidden = int(FLAGS.nhidden)
    else:
        n_hidden = 100

    batch_size = 32
    n_state_dims = 4
    use_sa_input = False
    LAMBDA = 0.000001
    
    if use_sa_input:
        n_inputs = n_state_dims + 1
        n_outputs = 1
    else:
        n_inputs = n_state_dims
        n_outputs = 2
        
    
    x = tf.placeholder(tf.float32, [None, n_inputs], 'x')
    target = tf.placeholder(tf.float32, [None, n_outputs], 'target')
    mean_episode_length = tf.Variable(0.0)
    tf.summary.scalar('mean_episode_length', mean_episode_length)
    mean_return = tf.Variable(0.0)
    tf.summary.scalar('mean_return', mean_return)


    if n_hidden==0:
        Qfunc, b, W = build_net(x, n_inputs, n_outputs)
    else:
        Qfunc, b, W = build_net2(x, n_inputs, n_hidden, n_outputs, use_batch_norm=False) 
    loss = tf.reduce_mean(tf.square(tf.sub(target, Qfunc)))
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    total_loss = loss + LAMBDA * l2_loss
    tf.summary.scalar('loss', loss)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
#     train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(total_loss)

    e_data = AllEpisodeData(MAX_EPISODES)     
    num_training_samples = e_data.collect_episodes(env, use_saved=False)
    path_arr = [FLAGS.model, "n_hidden{}".format(n_hidden), "lr{:g}".format(learning_rate)]

    with tf.Session() as sess:  
        if FLAGS.eval: #Restore saved model   
            fn= FLAGS.model + '_' + FLAGS.nhidden + '_' + FLAGS.lr
            model_file_name = root_dir + '/final_models/' + fn + '.ckpt'  
            print('loading model from: ' + model_file_name)  
            saver2restore = tf.train.Saver(write_version=1)
            saver2restore.restore(sess, model_file_name)
            mean_r, mean_episode_len = run_agent_on_env(sess, env, Qfunc, x, W, b, use_sa_input)
            print('mean episode length {} discounted reward {}'.format(mean_episode_len, mean_r))
            return
  

        merged = tf.summary.merge_all()
        
        dir_name = summaries_dir + '/A3/' + str(random.randint(0,99)) + '/lRate_' + str(learning_rate) ;
        train_writer = tf.summary.FileWriter(dir_name + '/train', sess.graph)
        
        # init operation
        tf.global_variables_initializer().run()    
        
        conv_tester = ConvergenceTester(1, lookback_window=5) #stop if converged to within 0.05%

        Qfunc_s_t1 = [0, 0]

#         batch_size = MAX_EPISODES
        # Train
        for epoch in range(10000):
            
            if epoch % 5 == 0: #update Q_beta value
                sa_t, r_t1, sa0_t1, sa1_t1, s_t, s_t1, a_t = e_data.all_data()
                Qfunc_s_t1, b_val, W_val = sess.run([Qfunc, b, W], feed_dict={x: s_t1})
                max_Q_beta = np.amax(Qfunc_s_t1, axis=1)
                e_data.save_maxQ(max_Q_beta)
                
            
#             for i in range(int( num_training_samples / batch_size)):
            for i in range(1000):
#                 sa_t, r_t1, sa0_t1, sa1_t1 = e_data.sample(i).get_data()
                sa_t, r_t1, sa0_t1, sa1_t1, s_t, s_t1, a_t, max_Q = e_data.sample(batch_size)
#                 sa_t, r_t1, sa0_t1, sa1_t1 = e_data.all_data()
                if use_sa_input:
                    Qfunc_s_t1[0] = sess.run(Qfunc, feed_dict={x: sa0_t1})
                    Qfunc_s_t1[1], b_val, W_val = sess.run([Qfunc, b, W], feed_dict={x: sa1_t1})
                    max_Q = np.maximum(Qfunc_s_t1[0], Qfunc_s_t1[1])
                    target_vals = r_t1 + GAMMA * np.squeeze(max_Q)
                    target_vals = np.expand_dims( target_vals, axis=1)
        
                    Qfunc_val, loss_val, _, train_summary = sess.run([Qfunc, loss, train_step, merged], 
                                            feed_dict={x: sa_t, target: target_vals})
                else:
                    Qfunc_s_t = sess.run(Qfunc, feed_dict={x: s_t})
#                     Qfunc_s_t1, b_val, W_val = sess.run([Qfunc, b, W], feed_dict={x: s_t1})
#                     max_Q = np.amax(Qfunc_s_t1, axis=1)
                    target_vals = Qfunc_s_t
                        
                    for target_val, action, r, mQ in zip(target_vals, a_t, r_t1, max_Q):
                        if r < 0:
                            target_val[action] = r
                        else:
                            target_val[action] = r + GAMMA * mQ
        
                    Qfunc_val, loss_val, _, train_summary = sess.run([Qfunc, loss, train_step, merged], 
                                            feed_dict={x: s_t, target: target_vals})
                    
         
            if epoch % 1 == 0: #calc intermediate results
                
                mean_r, mean_episode_len = run_agent_on_env(sess, env, Qfunc, x, W, b, use_sa_input)
                assign_op1 = mean_return.assign(mean_r)
                assign_op2 = mean_episode_length.assign(mean_episode_len)
                sess.run([assign_op1, assign_op2])
#                 _, _, loss_val = sess.run([assign_op1, assign_op2, loss],
                sa_t, r_t1, sa0_t1, sa1_t1, s_t, s_t1, a_t = e_data.all_data()
                if use_sa_input:
                    Qfunc_s_t1[0] = sess.run(Qfunc, feed_dict={x: sa0_t1})
                    Qfunc_s_t1[1], b_val, W_val = sess.run([Qfunc, b, W], feed_dict={x: sa1_t1})
                    
                    max_Q = np.maximum(Qfunc_s_t1[0], Qfunc_s_t1[1])
                    target_vals = r_t1 + GAMMA * np.squeeze(max_Q)
                    target_vals = np.expand_dims( target_vals, axis=1)
                    loss_val = sess.run(loss, feed_dict={x: sa_t, target: target_vals}) 
                else:
                    Qfunc_s_t = sess.run(Qfunc, feed_dict={x: s_t})
                    Qfunc_s_t1 = sess.run(Qfunc, feed_dict={x: s_t1})
                    
                    max_Q_test = np.amax(Qfunc_s_t1, axis=1)
                    target_vals = Qfunc_s_t
                    
                    for target_val, action, r, mQ in zip(target_vals, a_t, r_t1, max_Q_test):
                        if r < 0:
                            target_val[action] = r
                        else:
                            target_val[action] = r + GAMMA * mQ
                    Q_vals, loss_val, l2_loss_val, b_val, W_val,train_summary = sess.run([Qfunc, loss, l2_loss, b, W, merged], feed_dict={x: s_t, target: target_vals})

                train_writer.add_summary(train_summary, epoch)
                
                if epoch % 1 == 0: #calc intermediate results
                    print('{} loss {:.4f} l2_loss {:.4f} return {:.3f} len {:.0f} Q {:.3f} maxQ {:.3f} W {:.3f} b {:.3f}'.
                          format(epoch+1, loss_val, l2_loss_val*LAMBDA, mean_r, mean_episode_len, 
                        np.mean(Q_vals), np.mean(max_Q_beta), np.mean(W_val), np.mean(b_val)))
                               
                if mean_episode_len >199:
#                 if conv_tester.has_converged_to_constant(mean_episode_len, 200):
                    print('converged after ', epoch, ' epochs')
                    break

        #save trained model
        model_file_name = '_'.join(path_arr)
        save_model(sess, model_file_name, root_dir)

                    

def get_greedy_action(sess, Qfunc, x, s, W, b, use_sa_input):
    if use_sa_input:
        Qfunc_s_t1=[0,0]            
        batch_size = s.shape[0]
        a_t1 = np.zeros((batch_size,1))
        inputs_t1 = np.concatenate( (s, a_t1), axis=1)
        Qfunc_s_t1[0] = sess.run(Qfunc, feed_dict={x: inputs_t1})
        a_t1 = np.ones((batch_size,1))
        inputs_t1 = np.concatenate( (s, a_t1), axis=1)
        Qfunc_s_t1[1],W_val, b_val = sess.run([Qfunc,W,b], feed_dict={x: inputs_t1})
        
        max_Q = np.maximum(Qfunc_s_t1[0], Qfunc_s_t1[1])
        max_a = np.argmax(Qfunc_s_t1, axis=0)
        return max_Q[0,0], max_a[0,0]

    else:
        batch_size = s.shape[0]
        a_t1 = np.zeros((batch_size,1))
        Qfunc_s_t = sess.run(Qfunc, feed_dict={x: s})
        
        max_Q = np.amax(Qfunc_s_t)
        max_a = np.argmax(Qfunc_s_t, axis=1)
        return max_Q, max_a[0]


def run_agent_on_env(sess, env, Qfunc, x, W, b, use_sa_input):
    n_episodes = 100
    rewards = np.zeros(n_episodes)
    episode_length = np.zeros(n_episodes)
    
    for episode in range(n_episodes):
        discount_factor = GAMMA
        s_t = env.reset()
#         for t in range(100):
        for t in range(MAX_EPISODE_LENGTH):
            s_t = np.expand_dims( np.transpose(s_t), axis=0)
            _, max_a = get_greedy_action(sess, Qfunc, x, s_t, W, b, use_sa_input)
            s_t1, _, done, _ = env.step(max_a)
            
            if done:
                rewards[episode] -= 1.0 * discount_factor
                episode_length[episode] = t+1
                break
            
            discount_factor *= GAMMA
            s_t = s_t1
            
        if episode_length[episode]==0: #completed
            rewards[episode] = 0
            episode_length[episode] = t+1
                
    mean_r = np.mean(rewards)  
    mean_episode_len = np.mean(episode_length)  
#     print('mean reward {}'.format(mean_r))
    return mean_r, mean_episode_len


# run_net(1e-5)

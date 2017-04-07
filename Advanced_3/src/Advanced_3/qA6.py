import os
import gym
import numpy as np
import random
import tensorflow as tf
import pickle as pi
import matplotlib.pyplot as plt

root_dir = 'C:/Users/Dave/Documents/GI13-Advanced/Assignment3';
summaries_dir = root_dir + '/Summaries';
save_dir = root_dir;

MAX_EPISODES = 2000
MAX_EPISODE_LENGTH = 300
GAMMA = 0.99
EPSILON = 0.05

def save_model(session, model_name, root_dir):
    if not os.path.exists(root_dir + '/model/'):
        os.mkdir(root_dir + '/model/')
    saver = tf.train.Saver(write_version=1)
    save_path = saver.save(session, root_dir + '/model/' + model_name +'.ckpt')
        
class AllEpisodeData:
    
    def __init__(self):
        self._s_t = []
        self._a_t = []
        self._r_t1 = []
        self._s_t1 = []
        return
    
    def num_samples(self):
        return len(self._s_t)
    
    def sample(self, batch_size):
        idxs = np.random.randint(self.num_samples(), size=batch_size)
        
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

    def add_data(self, s_t, a_t, r_t1, s_t1):
        self._s_t.append(s_t)
        self._a_t.append(a_t)
        self._r_t1.append(r_t1)
        self._s_t1.append(s_t1)
    
    def normalize(self):
        med = np.median(self._s_t, axis=0)       
        
        for episode_data in self._data:
            episode_data.normalize(med)

        
def weight_variable(shape):
#     initial = tf.constant(0.0, shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, 'W')

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
#     initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, 'b')
 
def build_net(n_inputs, n_hidden, n_outputs, LAMBDA):
    x = tf.placeholder(tf.float32, [None,n_inputs], 'x')
    target = tf.placeholder(tf.float32, [None, n_outputs], 'target')
    
    W_1 = weight_variable([n_inputs, n_hidden])
    b_1 = bias_variable([n_hidden])   
    h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

    W_2 = weight_variable([n_hidden, n_outputs])
    b_2 = bias_variable([n_outputs])
    y = tf.matmul(h_1, W_2) + b_2
    
    residual = tf.sub(tf.stop_gradient(target), y)
    loss = 0.5 * tf.reduce_mean(tf.square(residual))
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    total_loss = loss + LAMBDA * l2_loss

    return x, y, target, total_loss, residual


def run_net(learning_rate, FLAGS):
    env = gym.make('CartPole-v0')

    batch_size = 16
    n_hidden = 100
    n_state_dims = 4
    LAMBDA = 0.00000001
    n_inputs = n_state_dims
    n_outputs = 2

    x, Qfunc, target, loss, residual = build_net(n_inputs, n_hidden, n_outputs, LAMBDA) 
    network_params = tf.trainable_variables()

    x_tar, Qfunc_tar, _,_,_ = build_net(n_inputs, n_hidden, n_outputs, LAMBDA) 
    target_network_params = tf.trainable_variables()[len(network_params):]
    
    update_target_op = [target_network_params[i].assign(network_params[i]) 
                            for i in range(len(target_network_params))]
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    path_arr = [FLAGS.model, "n_hidden{}".format(n_hidden), "lr{:g}".format(learning_rate)]

    n_episodes = 2
    n_repeats = 1
    trajectory_len = 5
    num_training_epochs = 1
    max_training_samples = 10000
    
    with tf.Session() as sess:   
        if FLAGS.eval: #Restore saved model   
            fn= FLAGS.model
            model_file_name = root_dir + '/final_models/' + fn + '.ckpt'  
            print('loading model from: ' + model_file_name)  
            saver2restore = tf.train.Saver(write_version=1)
            saver2restore.restore(sess, model_file_name)
            n_repeats=1
            n_episodes=100
            num_training_epochs=0
 
        episode_length = np.zeros((n_repeats, n_episodes))
        rewards = np.zeros((n_repeats, n_episodes))
        residuals = np.zeros((n_repeats, n_episodes))
        
        replay_buffer = AllEpisodeData()
        residual_val = 0
        
        
        for rep in range(n_repeats):
            tf.global_variables_initializer().run()    

            for episode in range(n_episodes):
                num_training_samples = replay_buffer.num_samples()
                if episode % trajectory_len == 0 and \
                        batch_size < num_training_samples: #update function

                    for epoch in range(num_training_epochs):
                        
                        for i in range(int( min(max_training_samples,num_training_samples) / batch_size)):

                            s_ts, a_ts, r_t1s, s_t1s = replay_buffer.sample(batch_size=batch_size)
                            Qfunc_s_t = sess.run(Qfunc, feed_dict={x: s_ts})
                            Qfunc_s_t1 = sess.run(Qfunc, feed_dict={x: s_t1s})
                            max_Q = np.amax(Qfunc_s_t1, axis=1)
                            
                            target_vals = Qfunc_s_t
                                
                            for target_val, action, r, mQ in zip(target_vals, a_ts, r_t1s, max_Q):
                                if r < 0:
                                    target_val[action] = r
                                else:
                                    target_val[action] = GAMMA * mQ
                
                            Qfunc_val, loss_val, residual_val, _ = sess.run([Qfunc, loss, residual, train_step], 
                                                    feed_dict={x: s_ts, target: target_vals})
                    print('{} len {} Q {:.3f} loss {:.3f} residual {:.3f}'.
                          format(episode, np.asscalar(np.mean(episode_length[rep, episode-5:episode])),np.asscalar(np.mean(Qfunc_val)), 
                                 np.asscalar(np.mean(loss_val)), np.asscalar(np.mean(residual_val))))


                s_t = env.reset()
                discount_factor = GAMMA
                
                for t in range(MAX_EPISODE_LENGTH):
                    a_t, _ = get_epsilon_greedy_action(sess, Qfunc, x, s_t, EPSILON)
                    s_t1, _, done, _ = env.step(a_t)
                    
                    if done:
                        r_t1 = -1
                    else:
                        r_t1 = 0

                    replay_buffer.add_data(s_t, a_t, r_t1, s_t1)
                    
#                     residual_val,_ = sess.run([residual], feed_dict={x: s_t})
        
                    if done:
                        break

                    discount_factor *= GAMMA
                    s_t = s_t1
    
#                 if episode % 20 ==0:
#                     print('{} Q {:.3f} loss {:.3f} residual {:.3f}'.
#                           format(episode, np.asscalar(np.mean(Qfunc_val)), loss_val, residual_val))
                    
                episode_length[rep,episode] = t+1
                if r_t1 == 0:
                    rewards[rep,episode] = 0
                else: #r_t1 == -1:
                    rewards[rep,episode] -= 1.0 * discount_factor
                residuals[rep,episode] = abs(np.asscalar(np.sum(residual_val)))
                    
                    

            print('run {} mean episode length {} discounted reward {}'.
                  format(rep, np.asscalar(np.mean(episode_length[rep])), 
                     np.asscalar(np.mean(rewards[rep]))))

            if FLAGS.eval:
                return

            pi.dump( (episode_length, residuals, rewards), open( 'qA6_data', "wb" ) )
        #save trained model
        model_file_name = '_'.join(path_arr)
        save_model(sess, model_file_name, root_dir)


#         (episode_length, losses, rewards) = pi.load( open( 'qA4_data', "rb" ) )
    
        x_s = np.arange(0, n_episodes, 1)
        
        plt.figure(1, figsize=(12, 8))
        plt.subplot(311)
        plt.plot(x_s, np.mean(residuals, axis=0), 'r-')
        plt.title("trajectory length: "+str(trajectory_len) + 
                  ' mean episode length: ' + str(np.asscalar(np.mean(episode_length))))
        plt.ylabel('Absolute Residual')
        
        plt.subplot(312)
        plt.plot(x_s, np.mean(episode_length, axis=0), 'b-')
        plt.ylabel('Episode length')
        
        plt.subplot(313)
        plt.plot(x_s, np.mean(rewards, axis=0), 'g-')
        plt.xlabel('episode #')
        plt.ylabel('Return')
        plt.tight_layout()
        plt.show() 
        
                    

def get_epsilon_greedy_action(sess, Qfunc, x, s, epsilon):

    s = np.expand_dims( np.transpose(s), axis=0)
#     s = np.expand_dims( np.transpose(s.copy()), axis=0)
                    
    Qfunc_s_t = sess.run(Qfunc, feed_dict={x: s})
    if random.random() < epsilon: #explore
        return random.randint(0,1), Qfunc_s_t
    else: #exploit
        max_a = np.argmax(Qfunc_s_t, axis=1)
        return max_a[0], Qfunc_s_t
    
# run_net(1e-3)

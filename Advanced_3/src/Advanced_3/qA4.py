import gym
import numpy as np
import random
import tensorflow as tf
import pickle as pi
import matplotlib.pyplot as plt

root_dir = 'C:/Users/Dave/Documents/GI13-Advanced/Assignment3';
summaries_dir = root_dir + '/Summaries';
save_dir = root_dir;

env = gym.make('CartPole-v0')

MAX_EPISODES = 2000
MAX_EPISODE_LENGTH = 300
GAMMA = 0.99
EPSILON = 0.05

        
def weight_variable(shape):
#     initial = tf.constant(0.0, shape=shape)
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, 'W')

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
#     initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, 'b')
 
def build_net(n_inputs, n_hidden, n_outputs, LAMBDA):
    x = tf.placeholder(tf.float32, [1,n_inputs], 'x')
    target = tf.placeholder(tf.float32, [1, n_outputs], 'target')
    
    W_1 = weight_variable([n_inputs, n_hidden])
    b_1 = bias_variable([n_hidden])   
    h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

    W_2 = weight_variable([n_hidden, n_outputs])
    b_2 = bias_variable([n_outputs])
    y = tf.matmul(h_1, W_2) + b_2
    
    loss = 0.5 * tf.reduce_mean(tf.square(tf.sub(target, y)))
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    total_loss = loss + LAMBDA * l2_loss

    return x, y, target, total_loss


def run_net(learning_rate):
    n_hidden = 100
    n_state_dims = 4
    LAMBDA = 0.00001
    n_inputs = n_state_dims
    n_outputs = 2

    x, Qfunc, target, loss = build_net(n_inputs, n_hidden, n_outputs, LAMBDA) 
    network_params = tf.trainable_variables()

    x_tar, Qfunc_tar, _, _ = build_net(n_inputs, n_hidden, n_outputs, LAMBDA) 
    target_network_params = tf.trainable_variables()[len(network_params):]
    
    update_target_op = [target_network_params[i].assign(network_params[i]) 
                            for i in range(len(target_network_params))]
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:    
#         merged = tf.summary.merge_all()
#         dir_name = summaries_dir + '/A3/' + str(random.randint(0,99)) + '/lRate_' + str(learning_rate) ;
#         train_writer = tf.summary.FileWriter(dir_name + '/train', sess.graph)
        
        n_episodes = 2000
        n_repeats = 1
        target_update_interval = 1
        step_update_interval = 0
        episode_length = np.zeros((n_repeats, n_episodes))
        rewards = np.zeros((n_repeats, n_episodes))
        losses = np.zeros((n_repeats, n_episodes))
        residuals = []
        
        for rep in range(n_repeats):
            tf.global_variables_initializer().run()    
            n_steps = 0

            for episode in range(n_episodes):
                s_t = env.reset()
                s_t = np.expand_dims( np.transpose(s_t), axis=0)
                discount_factor = GAMMA
                
                for t in range(MAX_EPISODE_LENGTH):

                    a_t, Qfunc_s_t_val = get_epsilon_greedy_action(sess, Qfunc, x, s_t, EPSILON)
                    s_t1, _, done, _ = env.step(a_t)
                    
                    target_val = Qfunc_s_t_val.copy()
                    if done:
                        r_t1 = -1
                        target_val[0,a_t] = r_t1
                    else:
                        s_t1 = np.transpose(np.expand_dims(s_t1, axis=1))
                        if target_update_interval > 0:
                            Qfunc_s_t1_val = sess.run(Qfunc_tar, feed_dict={x_tar: s_t1})
                        else:
                            Qfunc_s_t1_val = sess.run(Qfunc, feed_dict={x: s_t1})
                        max_Q_s_t1 = np.amax(Qfunc_s_t1_val)
                        target_val[0,a_t] = GAMMA * max_Q_s_t1
                        
                    
                    loss_val,_ = sess.run([loss,train_step], feed_dict={x: s_t, target: target_val})
        
                    if done:
                        break

                    if step_update_interval > 0 and t % step_update_interval == 0: 
                        sess.run(update_target_op)
                
                    discount_factor *= GAMMA
                    s_t = s_t1
                    n_steps += 1
                    
#                     if n_steps % 20 == 0:
#                         residuals.append(abs(target_val[0,a_t] - Qfunc_s_t_val[0,a_t]))
                    
                episode_length[rep,episode] = t+1
                if r_t1 == 0:
                    rewards[rep,episode] =0
                else: #r_t1 == -1:
                    rewards[rep,episode] -= 1.0 * discount_factor
                losses[rep,episode] = loss_val
                    
                if target_update_interval > 0 and episode % target_update_interval == 0: 
                    sess.run(update_target_op)
                    
            print('run {} len {}'.format(rep, np.asscalar(np.mean(episode_length[rep]))))
            pi.dump( (episode_length, losses, rewards), open( 'qA4_data', "wb" ) )


#         (episode_length, losses, rewards) = pi.load( open( 'qA4_data', "rb" ) )
    
        x_s = np.arange(0, n_episodes, 1)
        
        plt.figure(1, figsize=(15, 5))
        plt.subplot(311)
        plt.plot(x_s, np.mean(losses, axis=0), 'r-')
#         plt.title("target_update_interval: "+str(target_update_interval) + 
#                   ' mean episode length: ' + str(np.asscalar(np.mean(episode_length))))
        plt.ylabel('mean loss')
        
        plt.subplot(312)
        plt.plot(x_s, np.mean(episode_length, axis=0), 'b-')
        plt.ylabel('mean episode length')
        
        plt.subplot(313)
        plt.plot(x_s, np.mean(rewards, axis=0), 'g-')
        plt.xlabel('episode #')
        plt.ylabel('mean return')
        plt.tight_layout()
        plt.show() 
        
                    

def get_epsilon_greedy_action(sess, Qfunc, x, s, epsilon):

    Qfunc_s_t = sess.run(Qfunc, feed_dict={x: s})
    if random.random() < epsilon: #explore
        return random.randint(0,1), Qfunc_s_t
    else: #exploit
        max_a = np.argmax(Qfunc_s_t, axis=1)
        return max_a[0], Qfunc_s_t
    
run_net(1e-3)

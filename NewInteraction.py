import tensorflow as tf
import numpy as np
import dataManagement as dm


pairedFrames = dm.get_data('Resources/LogsCut/2018-06-19_19-24_CMμs-vs-TIGERs_Mannheim.json', 30)
np.random.shuffle(pairedFrames)


def train_on_teammate(bot1, bot2, ball):
    t_input = tf.reshape(tf.concat([bot1, bot2, ball], 0), [1,16])
    t_hidden1 = tf.layers.dense(inputs=t_input, units=30, activation=tf.nn.tanh)
    t_hidden2 = tf.layers.dense(inputs=t_hidden1, units=30, activation=tf.nn.tanh)
    t_output = tf.layers.dense(inputs=t_hidden2, units=6)
    return tf.to_float(t_output)

def train_on_opponent(bot1, bot2, ball):
    o_input = tf.reshape(tf.concat([bot1, bot2, ball], 0), [1,16])
    o_hidden1 = tf.layers.dense(inputs=o_input, units=30, activation=tf.nn.tanh)
    o_hidden2 = tf.layers.dense(inputs=o_hidden1, units=30, activation=tf.nn.tanh)
    o_output = tf.layers.dense(inputs=o_hidden2, units=6)
    return tf.to_float(o_output)

def train_cycle(team1, team2, ball):
    t1len = tf.shape(team1)[0]
    t2len = tf.shape(team2)[0]
    state = tf.Variable(tf.zeros([0,6]), dtype = tf.float32, trainable=False)    # Create dataholder for estimates of bot positions

    i = tf.constant(0, dtype=tf.int32)

    while_condition_outer =     lambda i, state: tf.less(i, t1len)                         # Create while conditions for loops
    while_condition_team =      lambda j, state: tf.less(j, t1len)
    while_condition_opponent =  lambda k, state: tf.less(k, t2len)

    def teamloop(j, state):
        state = tf.reduce_sum([state, train_on_teammate(tf.gather_nd(team1, [i]), tf.gather_nd(team1, [j]), ball)], 0)
        return tf.add(j,1), state

    def opponentloop(k, state):
        state = tf.reduce_sum([state, train_on_teammate(tf.gather_nd(team1, [i]), tf.gather_nd(team2, [k]), ball)], 0)
        return tf.add(k,1), state

    def outerloop(i, state):
        j = tf.constant(0, dtype=tf.int32)
        k = tf.constant(0, dtype=tf.int32)
        bot = tf.zeros([1,6], tf.float32)
        j, bot = tf.while_loop(while_condition_team, teamloop, [j, bot])
        k, bot = tf.while_loop(while_condition_opponent, opponentloop, [k, bot])
        state = tf.concat([state, bot], 0)
        return tf.add(i,1), state

    i, state = tf.while_loop(while_condition_outer, outerloop, [i, state], shape_invariants=[tf.TensorShape(None), tf.TensorShape([None, 6])])

    # state[0].assign(train_on_opponent(team1[0], team2[0], ball))
    # sliced = tf.slice(state, [0,0], [t1len, 6])
    division = tf.divide(state, tf.to_float(t1len+t2len))
    return division


with tf.Graph().as_default():
    tf_t1 = tf.placeholder(tf.float32, shape=(None, 6))     # Create inputs for Team 1, Team 2 and the Ball
    tf_t2 = tf.placeholder(tf.float32, shape=(None, 6))
    tf_ball = tf.placeholder(tf.float32, shape=4)

    tf_l = tf.placeholder(tf.float32, shape=(None, 6))      # Create placeholder for the Labels

    tf_y = train_cycle(tf_t1, tf_t2, tf_ball)               # Estimate by network

    loss = tf.losses.mean_squared_error(tf_l, tf_y)
    train_op = tf.train.AdamOptimizer(0.000001).minimize(loss)

    init = train_cycle(pairedFrames[0, 0, 0],pairedFrames[0, 0, 1], pairedFrames[0, 0, 2])
    with tf.Session() as sess:

        print("hoi")
        tf.global_variables_initializer().run()
        for i in range(1,5000):
            for frame in pairedFrames:
                calcLoss = sess.run([tf_y], feed_dict={tf_t1: frame[0, 0], tf_t2: frame[0, 1], tf_ball: frame[0, 2], tf_l: frame[1, 0]})
            print(i, " Loss: ", calcLoss)

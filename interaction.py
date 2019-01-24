import json
import tensorflow as tf
import numpy as np
import json
import jsonGameProcessor
import gamedrawer
import time
import random

# Takes the file from string 'resource' and transforms the data into a 4 dimensional array: dataOrdered[set][frame][team1, team2, ball][robotID][x,y,x_vel,etc]
def file_to_tensor(resource):
    read_file = open(resource, "r")
    data = json.load(read_file)
    dataOrdered = []
    for set in data:
        dataOrdered.append([])
        for frame in set:
            dataOrdered[-1].append([])
            dataOrdered[-1][-1].append([])
            for j in frame["robots_yellow"]:
                dataOrdered[-1][-1][-1].append([])
                dataOrdered[-1][-1][-1][-1].append(j["x"])
                dataOrdered[-1][-1][-1][-1].append(j["y"])
                dataOrdered[-1][-1][-1][-1].append(j["x_vel"]*100)
                dataOrdered[-1][-1][-1][-1].append(j["y_vel"]*100)
                dataOrdered[-1][-1][-1][-1].append(j["x_orien"])
                dataOrdered[-1][-1][-1][-1].append(j["y_orien"])
            dataOrdered[-1][-1].append([])
            for j in frame["robots_blue"]:
                dataOrdered[-1][-1][-1].append([])
                dataOrdered[-1][-1][-1][-1].append(j["x"])
                dataOrdered[-1][-1][-1][-1].append(j["y"])
                dataOrdered[-1][-1][-1][-1].append(j["x_vel"]*100)
                dataOrdered[-1][-1][-1][-1].append(j["y_vel"]*100)
                dataOrdered[-1][-1][-1][-1].append(j["x_orien"])
                dataOrdered[-1][-1][-1][-1].append(j["y_orien"])
            dataOrdered[-1][-1].append([])
            for j in frame["balls"]:
                dataOrdered[-1][-1][-1].append([])
                dataOrdered[-1][-1][-1][-1].append(j["x"])
                dataOrdered[-1][-1][-1][-1].append(j["y"])
                dataOrdered[-1][-1][-1][-1].append(j["x_vel"])
                dataOrdered[-1][-1][-1][-1].append(j["y_vel"])
    len1 = len(dataOrdered)
    len2 = len(dataOrdered[0])
    len3 = len(dataOrdered[0][0])
    len4 = len(dataOrdered[0][0][0])
    len5 = len(dataOrdered[0][0][0][0])
    npData = np.zeros((len1,len2,len3,len4, len5))
    for i in range(len1):
        for j in range(len2):
            for k in range(len3):
                for l in range(len4):
                    for m in range(len5):
                        try:
                            npData[i, j, k, l, m] = dataOrdered[i][j][k][l][m]
                        except:
                            pass
    return npData

def train_bot(training_robot, other_bot, weights1, weights2, weights3, weights4):
    input = tf.reshape(tf.concat([training_robot, other_bot], 0), [12,1])
    h1 = ( tf.nn.sigmoid(tf.matmul(tf.transpose(input), weights1)) -0.5 ) * 2
    o1 = tf.transpose(tf.matmul(h1, weights2))
    h2 = ( tf.nn.sigmoid(tf.matmul(tf.transpose(o1), weights3)) -0.5 ) * 2
    return tf.transpose(tf.matmul(h2, weights4))


def train_ball(training_robot, ball, weights1, weights2, weights3, weights4):
    input = tf.reshape(tf.concat([training_robot, ball], 0), [10,1])
    h1 = ( tf.nn.sigmoid(tf.matmul(tf.transpose(input), weights1)) -0.5 ) * 2
    o1 = tf.transpose(tf.matmul(h1, weights2))
    h2 = ( tf.nn.sigmoid(tf.matmul(tf.transpose(o1), weights3)) -0.5 ) * 2
    return tf.transpose(tf.matmul(h2, weights4))

def train_cycle(frame, w_h1_o, w_o1_o, w_h1_t, w_o1_t, w_h1_b, w_o1_b, w_h2_o, w_o2_o, w_h2_t, w_o2_t, w_h2_b, w_o2_b):
    state = tf.Variable(tf.zeros([8,6],tf.float32))
    for i in range(0,8):
        for j in range(0,8):    # Train on teammates
            if i != j:
                state[i].assign(tf.math.add(train_bot(frame[0,i], frame[0,j], w_h1_t, w_o1_t, w_h2_t, w_o2_t), state[i]))

        for j in range(0,8):    # Train on opponents
            state[i].assign(tf.math.add(train_bot(frame[0,i], frame[1,j], w_h1_o, w_o1_o, w_h2_o, w_o2_o), state[i]))

                                # Train on ball
        state[i].assign(tf.math.add(train_ball(frame[0,i], frame[2,0,0:4], w_h1_b, w_o1_b, w_h2_b, w_o2_b), state[i]))

        state[i].assign(tf.divide(state[i], 16))
    return state


data = file_to_tensor('Resources/LogsCut/2018-06-19_19-24_CMμs-vs-TIGERs_Mannheim.json')        # Load data from file
predict_frames = 30                                        # Amount of frames to predict into the future.
x = data[:,:-predict_frames]
y = data[:,predict_frames:]
pairedSet = np.stack((x, y))
pairedSet = pairedSet.reshape([pairedSet.shape[0],-1,pairedSet.shape[3], pairedSet.shape[4], pairedSet.shape[5]])
pairedSet = pairedSet.transpose(1,0,2,3,4)
# np.random.shuffle(pairedSet)

tf_x = tf.placeholder(tf.float32, shape=(3, 8, 6))  # Placeholder for input data of network
tf_y = tf.placeholder(tf.float32, shape=(8, 6))     # Placeholder for real data.

w_h1_o = tf.Variable(tf.random.normal(shape = (12, 12), mean = 0, stddev = 0.1))   # Weights opponent
w_o1_o = tf.Variable(tf.random.normal(shape = (12, 12), mean = 0, stddev = 0.1))
w_h2_o = tf.Variable(tf.random.normal(shape = (12, 12), mean = 0, stddev = 0.1))
w_o2_o = tf.Variable(tf.random.normal(shape = (12, 6), mean = 0, stddev = 0.1))
w_h1_t = tf.Variable(tf.random.normal(shape = (12, 12), mean = 0, stddev = 0.1))   # Weights teammates
w_o1_t = tf.Variable(tf.random.normal(shape = (12, 12), mean = 0, stddev = 0.1))
w_h2_t = tf.Variable(tf.random.normal(shape = (12, 12), mean = 0, stddev = 0.1))
w_o2_t = tf.Variable(tf.random.normal(shape = (12, 6), mean = 0, stddev = 0.1))
w_h1_b = tf.Variable(tf.random.normal(shape = (10, 10), mean = 0, stddev = 0.1))   # Weights ball
w_o1_b = tf.Variable(tf.random.normal(shape = (10, 10), mean = 0, stddev = 0.1))
w_h2_b = tf.Variable(tf.random.normal(shape = (10, 10), mean = 0, stddev = 0.1))
w_o2_b = tf.Variable(tf.random.normal(shape = (10, 6), mean = 0, stddev = 0.1))

py_x = train_cycle(tf_x, w_h1_o, w_o1_o, w_h1_t, w_o1_t, w_h1_b, w_o1_b, w_h2_o, w_o2_o, w_h2_t, w_o2_t, w_h2_b, w_o2_b)

loss = tf.losses.mean_squared_error(tf_y, py_x)                             # Loss is calculated taking the mean of the squared error of the output
train_op = tf.train.AdamOptimizer(0.000001).minimize(loss)                         # Trained with gradient descent
dg = gamedrawer.GameDrawer()
# initialize the jsonGameProcessor with the appropriate keys
# RD_RT = jsonGameProcessor.JsonToArray('Resources/Logs/RD_RT.json')
counter = 0
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(1,5000):
            for frame in range(len(pairedSet)-200):
                _, calcLoss, py, ty = sess.run([train_op, loss, py_x, tf_y], feed_dict={tf_x: pairedSet[frame,0], tf_y: pairedSet[frame,1,0]})
                time.sleep(0.02)
                # py = py.flatten()
                ty = pairedSet[frame,1,0].flatten()
                # py = np.append(ty,py)
                dg.clear_canvas()
                dg.draw_game_from_nparray(ty)
                # counter += 1
            print(i, " Loss: ", calcLoss)
    counter = 0
    # for set in range(1,5):
    #     for frame in range(len(x[set]-5)):
    #         # ty = y[set, frame][0].flatten()
    #         loss, py, ty = sess.run([loss, py_x, tf_y], feed_dict={tf_x: x[set,frame], tf_y: y[set,frame][0]})
    #         print("Loss:", loss)
    #         py = np.append(ty,py)
    #         # dg.clear_canvas()
    #         # dg.draw_game_from_nparray(py)

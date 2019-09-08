import tensorflow as tf
import numpy as np

import process_data

num2alpha = lambda c: chr(c + 64)

def ready(restore = False):
    global X, Y
    X = tf.compat.v1.placeholder(tf.float32, [None, 5])
    Y = tf.compat.v1.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.ones([5, 1]))
    b = tf.Variable(0.0)

    global y, cost, step
    y = tf.add(b, tf.matmul(X, W))
    cost = tf.reduce_mean(tf.square(y - Y))
    step = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(cost)

    global sess
    sess = tf.compat.v1.Session()
    if restore:
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, "./models/trained.model")
    else:
        sess.run(tf.compat.v1.global_variables_initializer())

    global learn_x, learn_y, test_x, test_y
    learn_x, learn_y, test_x, test_y = process_data.do()

def do(count = 5000, stop = True):
    cost_hist = np.array([])
    for i in range(count):
        sess.run(step, feed_dict = { X: learn_x, Y: learn_y})
        cost_cur = sess.run(cost, feed_dict = { X: learn_x, Y: learn_y})
        print(str(i) + ":" + str(cost_cur))
        cost_hist = np.append(cost_hist, cost_cur)
        if stop and i > 100 and cost_cur == cost_hist[i - 100]:
            print("Completed: " + str(i + 1) + " times learned.")
            return
    print("End: " + str(count) + " times learned.")

def check():
    result = sess.run(y, feed_dict = { X: test_x })
    for i in range(result.shape[0]):
        print("actual: " + num2alpha(int(test_y[i, 0])) + " predict: " + num2alpha(int(round(result[i, 0]))) + "(" + str(result[i, 0]) + ")")

def save():
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, "./models/trained.model")
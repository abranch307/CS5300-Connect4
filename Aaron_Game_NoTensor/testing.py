#This file is used for testing/learning Tensorflow functions

import tensorflow as tf

x = tf.constant([[35, 40, 45]], name='x')
y = tf.Variable(x + 5, name='y')

with tf.Session() as session:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/basic", session.graph)
    model = tf.initialize_all_variables()
    session.run(model)
    print(session.run(y))

x = tf.Variable(0, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    for i in range(5):
        session.run(model)
        x = x + 1
        print(session.run(x))


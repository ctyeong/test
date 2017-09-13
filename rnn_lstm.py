# https://pythonprogramming.net/rnn-tensorflow-python-machine-learning-tutorial/

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import cv2


filenames = ['cur_frame200.jpg', 'cur_frame201.jpg', 'cur_frame202.jpg', 'cur_frame203.jpg', 'cur_frame204.jpg', \
            'cur_frame205.jpg', 'cur_frame206.jpg', 'cur_frame207.jpg', 'cur_frame208.jpg', 'cur_frame209.jpg']
labels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]            

images = []
labels = []
for f, l in zip(filenames, labels):
    image = cv2.imread('images/' + f,cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    images.append(np.array(image))
    labels.append(np.array(l))
    # training_data.append([np.array(image),np.array(l)])
# shuffle(training_data)

# training_data = np.array(training_data)

# filename_queue = tf.train.string_input_producer(filenames)
# reader = tf.WholeFileReader()
# filename, content = reader.read(filename_queue)
# image = tf.image.decode_jpeg(content, channels=1) #3
# image = tf.cast(image, tf.float32)
# resized_image = tf.image.resize_images(image, [28,28])
# # resized_image = tf.cast(resized_image, tf.float32)
# labels = np.array(labels, dtype=np.float32)


# mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

hm_epochs = 3
n_classes = 10
batch_size = 2 #128
chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder(tf.float32, shape=[None, n_chunks,chunk_size])
y = tf.placeholder(tf.float32, shape=[None])

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])    
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)    
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # sess.run(tf.initialize_all_variables())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for k in range(int(len(labels)/batch_size)):
            # for _ in range(int(mnist.train.num_examples/batch_size)):
                
                # epoch_x, epoch_y = tf.train.batch([training_data, labels], batch_size=batch_size)
                # print(run([epoch_x, epoch_y]))
                # epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # epoch_x = tf.reshape(epoch_x, [batch_size,n_chunks,chunk_size])
                # print(imgs.shape)
                epoch_x = labels[k*batch_size:(k+1)*batch_size-1]
                epoch_x = np.reshape(epoch_x, [batch_size,n_chunks,chunk_size])

                epoch_y = labels[k*batch_size:(k+1)*batch_size-1]
                
                # print(epoch_x)
                
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)
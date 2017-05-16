import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)
house_price = house_size * 100 + np.random.randint(low=20000, high=70000, size=num_house)

#print(house_size)
#print(house_price)
#touples = zip(house_size, house_price)
#for size, price in touples:
#    print(size, price , price / size)
plt.plot(house_size, house_price, "bx")
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()

def normalize(array):
    return (array - array.mean()) / array.std()

num_train_examples = math.floor(num_house * 0.7)

#training data
train_house_size = np.asarray(house_size[:num_train_examples])
train_price = np.asarray(house_price[:num_train_examples])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

#test data
test_house_size = np.asarray(house_size[num_train_examples:])
test_price = np.asarray(house_size[:num_train_examples])

test_house_size_norm = normalize(test_house_size)
test_price_norm = normalize(test_price)

tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

#loss function
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2) / (2 * num_train_examples))

learning_rate = 0.1

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    display_every = 2
    num_training_iter = 50

    for iteration in range(num_training_iter):
        for(x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size:x, tf_price:y })

        c = sess.run(tf_cost, feed_dict={tf_house_size:train_house_size_norm, tf_price:train_price_norm })
        print("iteration #: ", '%04d' % (iteration + 1),    "cost=", "{:.9f}".format(c),
            "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

    print("Optimization finished!")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size:train_house_size_norm, tf_price: train_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n' )

    
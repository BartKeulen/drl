import tensorflow as tf

size = 1000
batch_size = 10

queue = tf.FIFOQueue(size, shapes=[1], dtypes=[tf.float32])
enqueue_op = queue.enqueue(tf.random_uniform(shape=[1], minval=-10., maxval=10., dtype=tf.float32))
inputs = queue.dequeue_many(batch_size)

train_op = tf.reduce_sum(inputs)

qr = tf.train.QueueRunner(queue, [enqueue_op] * 1)

sess = tf.Session()

coord = tf.train.Coordinator()
enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

for step in range(1000):
    print(sess.run(queue.size()))

coord.request_stop()

coord.join(enqueue_threads)

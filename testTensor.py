import tensorflow as tf
def create_initializer(initialize_range=0.2):
    return tf.truncated_normal_initializer(stddev=initialize_range)


indices=[0,1,2]
depth=8
onehot=tf.one_hot(indices,depth)

vocab_size=10
embedding_size=5
stddev=0.1
#v=tf.get_variable(name="word_embedding",shape=[vocab_size,embedding_size],initializer=create_initializer(stddev))
v=tf.get_variable(name="word_embedding",initializer=onehot)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    val=sess.run(v)
    print(val.shape)
    print(val)
    val2=sess.run(onehot)
    print(val2.shape)
    print(val2)
    print(val==val2)



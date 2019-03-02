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
batch_size=2
from_seq_size=3
to_seq_size=2
a=tf.ones(shape=[batch_size,from_seq_size,1],dtype=tf.float32)
b=tf.ones(shape=[batch_size,1,to_seq_size],dtype=tf.float32)
c=a*b
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    c_val=sess.run(c)
    a_val=sess.run(a)
    b_val=sess.run(b)
    print(a_val)
    print('**************')
    print(b_val)
    print('**************')
    print(c_val)
    print('**************')
    print(a_val.shape,b_val.shape,c_val.shape)

#    val=sess.run(v)
#    print(val.shape)
#    print(val)
#    val2=sess.run(onehot)
#    print(val2.shape)
#   print(val2)
#    print(val==val2)



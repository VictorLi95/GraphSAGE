import tensorflow as tf
import numpy as np
from k_sage import *
from layers import *
from sklearn import metrics

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('graph_name','cora','Name of the input graph.')
flags.DEFINE_integer('k',3,'The value of k in ksage.')
flags.DEFINE_integer('identity_dim',0,'Dimension of the id-based embeddings of nodes.')
flags.DEFINE_boolean('use_features',False,'Whether to use features-based embeddings or not.')
flags.DEFINE_integer('output_dim',50,'Dimension of the ksage model output embeddings.')
flags.DEFINE_list('fanouts',[5,5],'Shape of the fanouts of the GraphSAGE sampler.')
flags.DEFINE_string('aggregator_type','mean','Type of the aggregator in GraphSAGE, mean, seq, maxpool, meanpool or gcn.')
flags.DEFINE_string('pooling_type','mean','Type of the pooling operator from low-level motif embeddings to high-level motif embeddings.')
flags.DEFINE_list('intra_k_pooling_num',[1,5,5],'Number of embeddings each motif layer contributes to the final node embedding.')
flags.DEFINE_string('intra_k_pooling_type','mean','How to combine the embeddings in each motif layer, mean or max.')
flags.DEFINE_string('inter_k_pooling_type','concat_and_dense','How to combine the embeddings from different motif layers, concat_and_dense, mean or max.')

flags.DEFINE_boolean('supervised',True,'Use supervised training setting or unsupervised.')
flags.DEFINE_integer('num_epochs',100,'Number of epochs in training process.')
flags.DEFINE_integer('evalate_per_epochs',20,'Evaluate every the specified number of epochs.')
flags.DEFINE_float('train_ratio',0.1,'Ratio of training nodes/all nodes or training edges/all edges.')
flags.DEFINE_integer('batch_size',200,'Size of the training batch.')
flags.DEFINE_float('learning_rate',0.001,'Learning rate of the training process.')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.0, 'Value of dropout.')
flags.DEFINE_boolean('sigmoid_loss',False,'Use sigmoid or softmax for the prediction layer activation.')

def make_sup_batch(train_nodes, node_labels, batch_size):
    train_nodes_ = np.random.choice(train_nodes, batch_size, replace=False)
    node_labels_ = [node_labels[i] for i in train_nodes_]
    return train_nodes_, node_labels_

if __name__ == '__main__':
    FLAGS.fanouts = map(int, FLAGS.fanouts)
    FLAGS.intra_k_pooling_num = map(int, FLAGS.intra_k_pooling_num)

    G = build_graph(FLAGS.graph_name)
    model = KSage(G=G,k=FLAGS.k,identity_dim=FLAGS.identity_dim,
                  use_features=FLAGS.use_features,output_dim=FLAGS.output_dim,fanouts=FLAGS.fanouts,
                  ksage_aggregator_type=FLAGS.aggregator_type,ksage_pooling_type=FLAGS.pooling_type,
                  intra_k_pooling_num=FLAGS.intra_k_pooling_num,intra_k_pooling_type=FLAGS.intra_k_pooling_type,
                  inter_k_pooling_type=FLAGS.inter_k_pooling_type)
    
    if FLAGS.supervised:
        _, id_map, _, _ = build_k_graph(G, 1)
        train_nodes = []
        test_nodes = []
        node_labels = [0]
        for n in G.nodes():
            if np.random.random() < FLAGS.train_ratio:
                train_nodes.append(id_map[n])
            else:
                test_nodes.append(id_map[n])
            node_labels.append(G.node[n]['label'])
        num_classes = len(G.graph['label_set'])

        batch = tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name='batch1')
        labels = tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name='labels')
        prediction_layer = Dense(FLAGS.output_dim, num_classes, dropout=FLAGS.dropout, act=lambda x:x)
        batch_embeddings = model(batch)
        batch_predictions = prediction_layer(batch_embeddings)
        labels_ = tf.one_hot(labels, num_classes)
        if FLAGS.sigmoid_loss:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=batch_predictions,labels=labels_)) 
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=batch_predictions,labels=labels_))
        
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(loss)
    #else:
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epochs in range(FLAGS.num_epochs):
            sup_batch = make_sup_batch(train_nodes, node_labels, FLAGS.batch_size)
            outs = sess.run([loss,train_op],feed_dict={batch:sup_batch[0],labels:sup_batch[1]})
            print('epochs %d, loss=%f'%(epochs,outs[0]))
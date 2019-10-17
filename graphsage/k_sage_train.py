import tensorflow as tf
import numpy as np
import time
from k_sage import *
from layers import *
from minibatch import *
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
flags.DEFINE_integer('num_epochs',50,'Number of epochs in training process.')
flags.DEFINE_integer('evaluate_per_epochs',5,'Evaluate every the specified number of training steps.')
flags.DEFINE_integer('train_nodes_cnt',140,'Number of nodes for training.')
flags.DEFINE_integer('val_nodes_cnt',500,'Number of nodes for validation.')
flags.DEFINE_integer('test_nodes_cnt',1000,'Number of nodes for test.')
flags.DEFINE_integer('batch_size',20,'Size of the training batch.')
flags.DEFINE_float('learning_rate',0.01,'Learning rate of the training process.')
flags.DEFINE_float('weight_decay', 0.001, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.0, 'Value of dropout.')
flags.DEFINE_boolean('sigmoid_loss',False,'Use sigmoid or softmax for the prediction layer activation.')

def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid_loss:
        y_true = np.argmax(y_true, axis=1) # np.argmax: return the index of the max element, along the axis
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1 # tricky expression
        y_pred[y_pred <= 0.5] = 0 
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")  #micro f1 score and macro f1 score

def incremental_evaluate(sess, preds, loss, minibatch_iter, size, data='val'):
    assert data=='train' or data=='val' or data=='test', 'data should be one of [train, val, test]'
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, data)
        node_outs_val = sess.run([preds, loss], 
                         feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)

if __name__ == '__main__':
    '''
    G = build_graph('cora')
    model = KSage(G=G,k=1,identity_dim=10,
                  use_features=False,output_dim=10,fanouts=[2,2],
                  ksage_aggregator_type='mean',ksage_pooling_type='mean',
                  intra_k_pooling_num=[1],intra_k_pooling_type='mean',
                  inter_k_pooling_type='concat_and_dense')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tmp = tf.placeholder(tf.int32, shape=(5,))
    #print sess.run(model.encoders[1].sample(tmp)[0],feed_dict={tmp:[1,2,3,4,5]})
    print sess.run(model.initial_encoder(tmp),feed_dict={tmp:[1,2,3,4,5]})
    print sess.run(model.encoders[1](tmp),feed_dict={tmp:[1,2,3,4,5]})
    '''
    assert FLAGS.train_nodes_cnt % FLAGS.batch_size == 0 and FLAGS.val_nodes_cnt % FLAGS.batch_size == 0 and FLAGS.test_nodes_cnt % FLAGS.batch_size == 0, 'train/val/test nodes cnt must be integral mutiples of batch size. '
    FLAGS.fanouts = map(int, FLAGS.fanouts)
    FLAGS.intra_k_pooling_num = map(int, FLAGS.intra_k_pooling_num)
    G = build_graph(FLAGS.graph_name)
    model = KSage(G=G,k=FLAGS.k,identity_dim=FLAGS.identity_dim,
                  use_features=FLAGS.use_features,output_dim=FLAGS.output_dim,fanouts=FLAGS.fanouts,
                  ksage_aggregator_type=FLAGS.aggregator_type,ksage_pooling_type=FLAGS.pooling_type,
                  intra_k_pooling_num=FLAGS.intra_k_pooling_num,intra_k_pooling_type=FLAGS.intra_k_pooling_type,
                  inter_k_pooling_type=FLAGS.inter_k_pooling_type)
    
    if FLAGS.supervised:
        assert FLAGS.train_nodes_cnt % len(G.graph['label_set']) == 0, 'In supervised settings, we need same number of training nodes for each class'
        _, id_map, _, _ = build_k_graph(G, 1)
        
        for n in G.nodes():
            G.node[n]['train'] = False
            G.node[n]['val'] = False
            G.node[n]['test'] = False
        nodes_by_labels = {l:[] for l in G.graph['label_set']}
        for n in G.nodes():
            nodes_by_labels[G.node[n]['label']].append(n)
        train_nodes_cnt_per_label = FLAGS.train_nodes_cnt / len(G.graph['label_set'])
        for l in G.graph['label_set']:
            train_nodes_ = np.random.choice(nodes_by_labels[l], train_nodes_cnt_per_label, replace=False)
            for n in train_nodes_:
                G.node[n]['train'] = True
        nodes_rand_permutation = np.random.permutation(G.nodes())
        j = 0
        for n in nodes_rand_permutation:
            if j>= FLAGS.val_nodes_cnt + FLAGS.test_nodes_cnt:
                break
            if G.node[n]['train']:
                continue
            if j>=FLAGS.val_nodes_cnt:
                G.node[n]['test']=True
            else:
                G.node[n]['val']=True
            j+=1     
        
        
        #for edge in G.edges():
        #   if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
        #       G[edge[0]][edge[1]]['train_removed'] = True
        #   else:
        #       G[edge[0]][edge[1]]['train_removed'] = False
        
        num_classes = len(G.graph['label_set'])
        class_map = {n:G.node[n]['label'] for n in G.nodes()}
        placeholders = {    'labels' : tf.placeholder(tf.float32, shape=(FLAGS.batch_size, num_classes), name='labels'),
                            'batch' : tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name='batch1'),
                            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
                            'batch_size' : tf.placeholder(tf.int32, name='batch_size') }
        
        minibatch = NodeMinibatchIterator(G, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=G.graph['max_degree'])
        
        prediction_layer = Dense(FLAGS.output_dim, num_classes, dropout=FLAGS.dropout, act=lambda x:x)
        batch_embeddings = model(placeholders['batch'])
        batch_embeddings_after_norm = tf.nn.l2_normalize(batch_embeddings, 1)
        batch_outputs = prediction_layer(batch_embeddings_after_norm)
        if FLAGS.sigmoid_loss:
            batch_preds = tf.nn.sigmoid(batch_outputs)
        else:
            batch_preds = tf.nn.softmax(batch_outputs)
    
        if FLAGS.sigmoid_loss:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=batch_outputs,labels=placeholders['labels'])) 
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=batch_outputs,labels=placeholders['labels']))
        for var in prediction_layer.vars.values():
            loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for sage_encoder_ in model.encoders:
            if not sage_encoder_ is None:
                for aggregator_ in sage_encoder_.aggregators:
                    for var in aggregator_.vars.values():
                        loss += FLAGS.weight_decay *tf.nn.l2_loss(var)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
       
        grads_and_vars = optimizer.compute_gradients(loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        grad, _ = clipped_grads_and_vars[0]
        
        train_op = optimizer.apply_gradients(clipped_grads_and_vars)
       #train_op = optimizer.minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
           
        for epoch in range(1,FLAGS.num_epochs+1):

            minibatch.shuffle()
            
            #print('Epoch %04d' % (epoch+1))
            loss_in_one_epoch = 0.0
            batch_cnt = 0
            while not minibatch.end():

                feed_dict, labels = minibatch.next_minibatch_feed_dict()
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                t = time.time()
                outs = sess.run([train_op, loss, batch_preds], feed_dict=feed_dict)
                loss_in_one_epoch += outs[1]
                batch_cnt += 1
            print('Epoch '+str(epoch)+': loss = '+str(loss_in_one_epoch/batch_cnt))

            if epoch % FLAGS.evaluate_per_epochs == 0:
                val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, batch_preds, loss, minibatch, FLAGS.batch_size, data='val')
                train_cost, train_f1_mic, train_f1_mac, duration = incremental_evaluate(sess, batch_preds, loss, minibatch, FLAGS.batch_size, data='train')
                #train_cost = outs[1]
                #train_f1_mic , train_f1_mac = calc_f1(labels, outs[2])
                print("After Epoch"+str(epoch),
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                      "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                      "val_f1_mac=", "{:.5f}".format(val_f1_mac))

        print("Optimization Finished!")
        test_cost, test_f1_mic, test_f1_mac, duration = incremental_evaluate(sess, batch_preds, loss, minibatch, FLAGS.batch_size, data='test')
        print('On test data:')
        print("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f}".format(val_cost, val_f1_mic, val_f1_mac))
        '''
        minibatch.shuffle()
        while not minibatch.end():
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            print(placeholders['batch'],sess.run(batch_embeddings, feed_dict=feed_dict))
            break
        '''
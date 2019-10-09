import tensorflow as tf
import numpy as np
from graph_builder import *
from aggregators import *
from neigh_samplers import *
from models import *
from layers import *

class ShallowEncoder(Layer):
    def __init__(self, node_nums, identity_dim, features, output_dim, **kwargs):
        assert identity_dim > 0 or not features is None, 'Must use id-based embeddings or feature-based embeddings or both.'
        super(ShallowEncoder, self).__init__(**kwargs)
        self.output_dim = output_dim
        if identity_dim > 0:
            with tf.variable_scope(self.name):
                self.id_embeddings = tf.get_variable(name='id_embeddings',shape=[node_nums, identity_dim])
        else:
            self.id_embeddings = None
        if features is None:
            self.outputs_ = self.id_embeddings
        else:
            self.features = tf.get_variable(name='features',initializer=tf.constant(features,dtype=tf.float32),trainable=False)
            if not self.id_embeddings is None:
                self.outputs_ = tf.concat([self.id_embeddings, self.features], axis=1)
            else:
                self.outputs_ = self.features
            
        self.dense = Dense(input_dim = identity_dim + (features.shape[1] if not features is None else 0), output_dim=output_dim, act=lambda x:x)
        self.outputs = self.dense(self.outputs_)

    def __call__(self, inputs):
        return tf.nn.embedding_lookup(self.outputs, inputs)

class SageEncoder(Layer):

    def __init__(self, input_encoder, fanouts, adj, aggregator_type='mean', concat=False, **kwargs):
        assert aggregator_type in ['mean','seq','maxpool','meanpool','gcn'], 'Unknown aggregator_type: '+str(aggregator_type)+'.'
        super(SageEncoder, self).__init__(**kwargs)
        self.input_encoder = input_encoder
        self.fanouts = fanouts
        self.output_dim = input_encoder.output_dim
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        self.aggregators = [self.aggregator_cls(input_dim=self.output_dim, output_dim=self.output_dim,concat=concat) for i in range(len(fanouts)-1),self.aggregator_cls(input_dim=self.output_dim, output_dim=self.output_dim,concat=concat,act=lambda x : x)]
        with tf.variable_scope(self.name):
            self.adj = tf.get_variable(name='adj',initializer=tf.constant(adj,dtype=tf.int32),trainable=False)
        self.neigh_sampler = UniformNeighborSampler(self.adj)

    def sample(self, inputs):
        samples = [inputs]
        support_size = 1
        support_sizes = [support_size] # [ 1 ]
        for k in range(len(self.fanouts)):
            support_size *= self.fanouts[k] # 1->5->25->125
            node = self.neigh_sampler((samples[k], self.fanouts[k]))
            samples.append(tf.reshape(node, [-1])) # example: samples [shape (batch_size) tensor, shape (batch_size*5) tensor, shape (batch_size*5*5) tensor, ... ]
            support_sizes.append(support_size) # example: support_sizes = [1,5,25,125]
        return samples, support_sizes

    def __call__(self, inputs):
        inputs_shape=inputs.shape
        inputs = tf.reshape(inputs,[-1])
        samples, support_sizes = self.sample(inputs)
        hidden = [self.input_encoder(node_samples) for node_samples in samples]
        for layer in range(len(self.fanouts)):
            aggregator = self.aggregators[layer]
            next_hidden = []
            for hop in range(len(self.fanouts) - layer):
                neigh_shape = [-1, self.fanouts[hop], self.output_dim]
                h = aggregator((hidden[hop], tf.reshape(hidden[hop + 1], neigh_shape)))
                next_hidden.append(h)
            hidden = next_hidden
        output_shape = inputs_shape.concatenate(self.output_dim)
        output_shape = [d if d is not None else -1 for d in output_shape.as_list()]
        return tf.reshape(hidden[0],output_shape)

class PoolingLayer(Layer):
    def __init__(self, input_encoder, pooling_map, pooling_type='mean', sample_num=-1, **kwargs):
        assert pooling_type in ['mean','max']
        super(PoolingLayer,self).__init__(**kwargs)
        self.input_encoder = input_encoder
        self.pooling_type = pooling_type
        self.pooling_map = pooling_map
        self.output_dim = input_encoder.output_dim
        self.sample_num = sample_num

    def __call__(self, inputs):
        if self.sample_num == -1:
            pooling_map_ = self.pooling_map
        else:
            pooling_map_ = tf.transpose(tf.random_shuffle(tf.transpose(self.pooling_map)))
            pooling_map_ = tf.slice(pooling_map_, [0,0], [-1,self.sample_num])
        inputs_ = self.input_encoder(tf.nn.embedding_lookup(pooling_map_, inputs))
        if self.pooling_type == 'mean':
            outputs = tf.reduce_mean(inputs_, axis=1)
        elif self.pooling_type == 'max':
            outputs = tf.reduce_max(inputs_, axis=1)
        return outputs

class KSage(GeneralizedModel):
    def __init__(self, G, k=2, identity_dim=0, use_features=False, output_dim=20, 
                 fanouts=[5], ksage_aggregator_type='mean', ksage_pooling_type='max', 
                 inter_k_pooling_type='concat_and_dense', intra_k_pooling_num=[1], intra_k_pooling_type='mean',
                 concat=False,
                 **kwargs):
            
        assert isinstance(k,int) and k>=1 and k<=3, 'Invalid value of k: '+str(k)+'.'
        assert ksage_aggregator_type in ['mean','seq','maxpool','meanpool','gcn'], 'Unknown aggregator_type: '+str(aggregator_type)+'.'
        assert ksage_pooling_type in ['mean','max'], 'Unknown ksage_pooling_type: '+str(ksage_pooling_type)+'.'
        assert inter_k_pooling_type in ['concat_and_dense','mean','max'], 'Unknown inter_k_pooling_type: '+str(inter_k_pooling_type)+'.'
        assert len(intra_k_pooling_num)==k and intra_k_pooling_num[0] == 1, 'Invalid value of intra_k_pooling_num: '+str(intra_k_pooling_num)+'.'
        assert intra_k_pooling_type in ['mean','max'], 'Unknown intra_k_pooling_type: '+str(intra_k_pooling_type)+'.'
        assert identity_dim > 0 or use_features, 'Must use id-based embeddings or feature-based embeddings or both.'

        super(KSage, self).__init__(**kwargs)
        self.k = k
        self.identity_dim = identity_dim
        self.use_features = use_features
        self.output_dim = output_dim
        self.fanouts = fanouts
        self.ksage_aggregator_type = ksage_aggregator_type
        self.ksage_pooling_type = ksage_pooling_type
        self.concat = concat

        self.G = [None]
        self.id_map = [None]
        self.pooling_map = [None]
        self.reverse_pooling_map = [None]
        self.adj = [None]
        self.deg = [None]
        
        for k_ in range(1,self.k+1):
            G_, id_map_, pooling_map_, reverse_pooling_map_ = build_k_graph(G, k_)
            adj_, deg_ = construct_adj(G_, id_map_)
            self.G.append(G_.copy())
            self.id_map.append(id_map_.copy())
            if pooling_map_ is None:
                self.pooling_map.append(None)
            else:
                self.pooling_map.append(pooling_map_.copy())
            if reverse_pooling_map_ is None:
                self.reverse_pooling_map.append(None)
            else:
                self.reverse_pooling_map.append(reverse_pooling_map_.copy())
            self.adj.append(adj_.copy())
            self.deg.append(deg_.copy())
            
        if self.use_features:
            id_map_ = self.id_map[1]
            features = np.zeros((G.number_of_nodes()+1,G.graph['features_dim']))
            for n in G.nodes():
                features[id_map_[n]] = G.node[n]['features']

        self.initial_encoder = ShallowEncoder(self.adj[1].shape[0], identity_dim, features, output_dim=self.output_dim)
        self.encoders = [None, SageEncoder(input_encoder=self.initial_encoder,fanouts=self.fanouts,adj=self.adj[1],aggregator_type=self.ksage_aggregator_type,concat=self.concat)]
        self.pooling_layers = [None, None]
        self.intra_k_pooling_layers = [None, None]
        for k_ in range(2,self.k+1):
            self.pooling_layers.append(PoolingLayer(input_encoder=self.encoders[k_-1],pooling_map=self.pooling_map[k_],pooling_type=self.ksage_pooling_type))
            self.encoders.append(SageEncoder(input_encoder=self.pooling_layers[k_],fanouts=self.fanouts,adj=self.adj[k_],aggregator_type=self.ksage_aggregator_type,concat=self.concat))
            self.intra_k_pooling_layers.append(PoolingLayer(input_encoder=self.encoders[k_],pooling_map=self.reverse_pooling_map[k_],pooling_type=intra_k_pooling_type,sample_num=intra_k_pooling_num[k_-1]))
        
        self.inter_k_pooling_type = inter_k_pooling_type
        if self.inter_k_pooling_type == 'concat_and_dense':
            self.inter_dense = Dense(input_dim = self.output_dim * self.k, output_dim=self.output_dim,act=lambda x:x)
        
    def __call__(self, inputs):
        if self.inter_k_pooling_type == 'concat_and_dense':
            inputs_ = [self.encoders[1](inputs)]
            if self.k == 1:
	            return inputs_[0]
            else:
                for k_ in range(2,self.k+1):
                    inputs_.append(self.intra_k_pooling_layers[k_](inputs))
                inputs__ = tf.concat(inputs_,axis=-1)
                return self.inter_dense(inputs__)
        else:
            inputs_ = [tf.expand_dims(self.encoders[1](inputs),axis=1)]
            for k_ in range(2,self.k+1):
                inputs_.append(tf.expand_dims(self.intra_k_pooling_layers[k_](inputs),axis=1))
            inputs__ = tf.concat(inputs_,axis=1)
            if self.inter_k_pooling_type == 'mean':
                return tf.reduce_mean(inputs__,axis=1)
            else:
                return tf.reduce_max(inputs__,axis=1)
    '''
if __name__ == "__main__":
    a = ShallowEncoder(node_nums=5, identity_dim=16, features=None, output_dim=5)
    adj_ = np.array([[0,1,2,3,4] for i in range(5)])
    b = SageEncoder(input_encoder=a, fanouts=[5,5], adj=adj_)
    c = PoolingLayer(input_encoder=b, pooling_type='max',pooling_map=np.array([[1,2],[1,2],[3,4],[0,4]]))
    inputs_ = tf.placeholder(tf.int32,[None])
    d = KSage(graph_name='cora',k=3,identity_dim=5,features=None,output_dim=5,fanouts=[5,5],intra_k_pooling_num=[1,5,5])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(d(inputs_),feed_dict={inputs_:[76,250,294,83]})
    '''

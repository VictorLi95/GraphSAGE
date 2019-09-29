import tensorflow as tf
import numpy as np
from graph_builder import *
#from .aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator
from aggregators import *
from neigh_samplers import *
from models import *
from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')

class ShallowEncoder(Layer):
    def __init__(self, node_nums, identity_dim, features, output_dim, **kwargs):
        assert identity_dim > 0 or not features is None, 'Must use id-based embeddings or feature-based embeddings or both.'
        super(ShallowEncoder, self).__init__(**kwargs)
        if identity_dim > 0:
            with tf.variable_scope(self.name):
                self.id_embeddings = tf.get_variable(name='id_embeddings',shape=[node_nums, identity_dim])
        else:
            self.id_embeddings = None
        if features is None:
            self.outputs_ = self.id_embeddings
        else:
            self.features = tf.get_variable(name='features',initializer=tf.constant(features))
            if not self.embeddings is None:
                self.outputs_ = tf.concat([self.id_embeddings, self.features], axis=1)
            else:
                self.outputs_ = self.features
            
        self.dense = Dense(input_dim = identity_dim + (features.shape[1] if not features is None else 0), output_dim=output_dim)
        self.outputs = self.dense(self.outputs_)

    def __call__(self, inputs):
        return tf.nn.embedding_lookup(self.outputs, inputs)

class SageEncoder(Layer):

    def __init__(self, input_encoder, fanouts, adj, dim=10, aggregator_type='mean', concat=False, **kwargs):
        assert aggregator_type in ['mean','seq','maxpool','meanpool','gcn'], 'Unknown aggregator_type: '+str(aggregator_type)+'.'
        super(SageEncoder, self).__init__(**kwargs)
        self.input_encoder = input_encoder
        self.fanouts = fanouts
        self.dim = dim
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
        self.aggregators = [self.aggregator_cls(input_dim=self.dim, output_dim=self.dim,concat=concat) for i in range(len(fanouts)-1), 
                            self.aggregator_cls(input_dim=self.dim, output_dim=self.dim,concat=concat,act=lambda x : x)]
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
        inputs_shape = inputs.shape
        inputs = tf.reshape(inputs,[-1])
        samples, support_sizes = self.sample(inputs)
        hidden = [self.input_encoder(node_samples) for node_samples in samples]
        for layer in range(len(self.fanouts)):
            aggregator = self.aggregators[layer]
            next_hidden = []
            for hop in range(len(self.fanouts) - layer):
                neigh_shape = [-1, self.fanouts[hop], self.dim]
                h = aggregator((hidden[hop], tf.reshape(hidden[hop + 1], neigh_shape)))
                next_hidden.append(h)
            hidden = next_hidden
        output_shape = inputs_shape.concatenate(self.dim)
        output_shape = [d if d is not None else -1 for d in output_shape.as_list()]
        return tf.reshape(hidden[0],output_shape)

class PoolingLayer(Layer):
    def __init__(self, input_encoder, pooling_map, pooling_type='mean', **kwargs):
        assert pooling_type in ['mean','max']
        super(PoolingLayer,self).__init__(**kwargs)
        self.input_encoder = input_encoder
        self.pooling_type = pooling_type
        self.pooling_map = pooling_map

    def __call__(self, inputs):
        inputs_ = self.input_encoder(tf.nn.embedding_lookup(self.pooling_map, inputs))
        if self.pooling_type == 'mean':
            outputs = tf.reduce_mean(inputs_, axis=1)
        elif self.pooling_type == 'max':
            outputs = tf.reduce_max(inputs_, axis=1) 
        return outputs

class KSage(GeneralizedModel):
    def __init__(self, graph_name='cora', k=2, identity_dim=0, features=None, fanouts=[5], aggregator_type='mean', pooling_type='max', concat=True, **kwargs):
            
        assert isinstance(k,int) and k>=1 and k<=3, 'Invalid value of k: '+str(k)+'.'
        assert aggregator_type in ['mean','seq','maxpool','meanpool','gcn'], 'Unknown aggregator_type: '+str(aggregator_type)+'.'
        assert identity_dim > 0 or not features is None, 'Must use id-based embeddings or feature-based embeddings or both.'
        super(KSage, self).__init__(**kwargs)
        self.graph_name = graph_name
        self.k = k
        self.identity_dim = identity_dim
        self.features = features
        self.fanouts = fanouts
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
        self.pooling_type = pooling_type
        self.concat = concat

        _G = build_graph(graph_name)
        self.G = [None]
        self.id_map = [None]
        self.pooling_map = [None]
        self.adj = [None]
        self.deg = [None]
        for k_ in range(self.k):
            G_, id_map_, pooling_map_ = build_k_graph(_G, k_ + 1)
            adj_, deg_ = construct_adj(G_, id_map_)
            self.G.append(G_)
            self.id_map.append(id_map_)
            self.pooling_map.append(pooling_map_)
            self.adj.append(adj_)
            self.deg.append(deg_)

        self.encoders = [None, ShallowEncoder(adj[1].shape[0], identity_dim, features)]
        self.pooling_layers = [None, None]
        for k_ in range(2,self.k+1):
            self.pooling_layers.append(Pooling_Layer(input_encoder=self.encoders[-1],pooling_map=self.pooling_map[k_],pooling_type=self.pooling_type))
            self.encoders.append(SageEncoder(input_encoder=self.pooling_layers[-1],fanouts=self.fanouts,dim=))
        #self. = [None, ShallowEncoder(adj[1].shape[0], identity_dim, features)]
        #self.inputs.extend([SageEncoder(k=i) for i in range(2,self.k+1)])
        #self.outputs = {}
        
        


if __name__ == "__main__":
    a = ShallowEncoder(node_nums=5, identity_dim=16, features=None, output_dim=5)
    print a.name
    adj_ = np.array([[0,1,2,3,4] for i in range(5)])
    b = SageEncoder(input_encoder=a, fanouts=[5,5], dim=5, adj=adj_)
    c = PoolingLayer(input_encoder=b, pooling_type='max',pooling_map=np.array([[1,2],[1,2],[3,4],[0,4]]))
    inputs_ = tf.placeholder(tf.int32,[None])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.initialize_all_variables())
        print sess.run(a(inputs_),feed_dict={inputs_:[1,2]})
        #print sess.run(b.sample([1,2],[5,5]))
        print sess.run(b(inputs_),feed_dict={inputs_:[2,3]})
        print sess.run(c(inputs_),feed_dict={inputs_:[0,1]})
    

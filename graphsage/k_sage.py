import tensorflow as tf
import numpy as np
from graph_builder import *

_LAYER_UIDS = {}
def get_layer_uid(layer_name='')
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    def __init__(self, **kwargs):
        class_name = self.__class__.__name__.lower()
        self.name = class_name + '_' + str(get_layer_uid(class_name))
        
class ShallowEncoder(Layer):
    def __init__(self, identity_dim, features):
        if identity_dim > 0:
            self.id_embeddings = tf.get_variable('node_id_embeddings',[adj.shape[0], identity_dim])
        else:
            self.id_embeddings = None
        if features is None:
            if identity_dim == 0:
                raise Exception('Must have a positive value for identity feature dimension if no input features given.')                                          self.raw_input = self.id_embeddings
        else:
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
                                                                                                                                        if not self.embeddings is None:
                                                                                                                                                            self.raw_input = tf.concat([self.id_embeddings, self.features], axis=1)
                                                                                                                                                                        else:
                                                                                                                                                                                            self.raw_input = self.features
                                                                                                                                                                                                    self.concat = concat
    def __call__(self, inputs):
        

class SageEncoder(Layer):
    def __init__(self, input_encoder, ):

    def __call__():

class PoolingLayer(Layer):
    def __init__():

    def __call__():

class KSage(object):

    def __init__(self, G, k=2, identity_dim=0, features=None, concat=True, aggregator_type, layer_infos, **kwargs):
        '''
        G, the input networkx.Graph()
        k, 2 or 3
        
        '''
        self.k = k
        self.G = {}
        self.id_map = {}
        self.adj = {}
        self.deg = {}
        for i in range(1,self.k+1):
            G_, id_map, pooling_map = build_k_graph(G, i)
            self.G[i] = G_
            self.id_map[i] = id_map
            adj, deg = construct_adj(G_, id_map) 
            self.adj[i] = adj
            self.deg[i] = deg
        self.inputs = {1:ShallowEncoder(identity_dim, features), i:SageEncoder(k=i) for i in range(2,k+1)}
        self.outputs = {}

        if identity_dim > 0:
	    self.id_embeddings = tf.get_variable('node_id_embeddings',[adj.shape[0], identity_dim])
        else:
            self.id_embeddings = None
        if features is None:
            if identity_dim == 0:
                raise Exception('Must have a positive value for identity feature dimension if no input features given.')
            self.raw_input = self.id_embeddings
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeddings is None:
                self.raw_input = tf.concat([self.id_embeddings, self.features], axis=1)
            else:
 		self.raw_input = self.features
 	self.concat = concat

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
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)



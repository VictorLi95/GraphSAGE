import tensorflow as tf
from graph_builder import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class KSage(Model):

    def __init__(self, G, k=2, adj, identity_dim=0, features=None, concat=True, aggregator_type, layer_infos, **kwargs):
        '''
        G, the input networkx.Graph()
        k, 2 or 3
        
        '''
        self.G = {i:build_k_graph(G,i)[0] for i in range(1,k+1)}
        self.k = k
        self.adj = adj
        if identity_dim > 0:
	    self.embeds = tf.get_variable('node_embeddings',[adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception('Must have a positive value for identity feature dimension if no input features given.')
            self.raw_input = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.raw_input = tf.concat([self.embeds, self.features], axis=1)
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
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

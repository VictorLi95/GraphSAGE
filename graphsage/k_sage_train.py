import tensorflow as tf
import numpy as np
from k_sage import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('graph_name','cora','Name of the input graph.')
flags.DEFINE_integer('k',2,'The value of k in ksage.')
flags.DEFINE_boolean('supervised',True,'Use supervised training setting or unsupervised.')
flags.DEFINE_integer('identity_dim',0,'Dimension of the id-based embeddings of nodes.')

if __name__ == '__main__':
    model = KSage(graph_name=FLAGS.graph_name,k=FLAGS.k,identity_dim=5,features=None,output_dim=5,fanouts=[5,5],intra_k_pooling_num=[1,5,5])

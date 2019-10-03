import os
import numpy as np 
import networkx as nx 

def build_graph(graph_name,directed=False):
  #build graph from file 'graph_data/xxx_edge_list.txt' & 'graph_data/xxx_label.txt'
  #return G (default = None)
  
  if not os.path.exists('graphsage/graph_data/'+graph_name + '/edge_list.txt'):
    print('File graph_data/'+graph_name+'/edge_list.txt does not exist. Graph building failed.')
    return None
  
  if directed:
    G = nx.DiGraph()
  else:
    G = nx.Graph()
  
  label_set = set()
  if os.path.exists('graphsage/graph_data/'+graph_name+'/label.txt'):
    fr = open('graphsage/graph_data/'+graph_name + '/label.txt')
    while True:
      s = fr.readline().strip()
      if not s:
        break
      s = s.split('\t')
      nodeid = int(s[0])
      label = int(s[1])
      label_set.add(label)
      G.add_node(nodeid, label=label) 
    fr.close()
  G.graph['label_set'] = label_set
  
  features_dim = 0
  if os.path.exists('graphsage/graph_data/'+graph_name+'/features.txt'):
    fr = open('graphsage/graph_data/'+graph_name+'/features.txt')
    while True:
      s = fr.readline().strip()
      if not s:
        break
      s = s.split(' ')
      if not features_dim:
        features_dim = len(s) - 1
      nodeid = int(s[0])
      features = []
      for j in range(features_dim):
        features.append(float(s[1+j]))
      G.add_node(nodeid, features=features[:])
    fr.close()
  G.graph['features_dim'] = features_dim

  fr = open('graphsage/graph_data/'+graph_name+'/edge_list.txt')
  while True:
    s = fr.readline().strip()
    if not s:
      break
    s = s.split('\t')
    node1id = int(s[0])
    node2id = int(s[1])
    G.add_edge(node1id,node2id)
  fr.close()
   
  max_degree = 0
  for n in G.nodes():
    max_degree = max(max_degree, len(list(G.neighbors(n))))
  G.graph['max_degree'] = max_degree

  print('Graph building finished.')
  print('Graph name = '+graph_name+'.')
  print('Number of nodes = '+str(G.number_of_nodes())+'.')
  print('Number of edges = '+str(G.number_of_edges())+'.')
  print('Max degree = '+str(max_degree)+'.')
  if directed:
    print('The graph is a directed graph.')
  else:
    print('The graph is an undirected graph.') 
  if len(label_set)==0:
    print('The graph is an unlabeled graph.')
  else:
    print('The graph is a labeled graph. Set of labels = '+str(label_set))
  if features_dim == 0:
    print('Nodes of the graph have no input features.')
  else:
    print('Nodes of the graph have input features of '+str(features_dim)+' dimensions.')
  return G

def make_ordered_tuple_of_2(x1,x2):
    if x1<x2:
        return (x1,x2)
    else:
        return (x2,x1)

def make_ordered_tuple_of_3(x1,x2,x3):
    if x1<x2:
        if x2<x3:
            return (x1, x2, x3)
        elif x1<x3:
            return (x1, x3, x2)
        else:
            return (x3, x1, x2)
    else:
        if x1<x3:
            return (x2, x1, x3)
        elif x2<x3:
            return (x2, x3, x1)
        else:
            return (x3, x2, x1)

def construct_adj(G, id_map):
    max_degree = G.graph['max_degree']
    num_of_nodes = G.number_of_nodes()
    adj = len(G.nodes()) * np.ones((len(G.nodes())+1, max_degree),dtype=np.int32) 
    deg = np.zeros((len(G.nodes())+1,),dtype=np.int32)

    for n in G.nodes():
        neighbors = np.array([id_map[neighbor] for neighbor in G.neighbors(n)],dtype=np.int32)       
        deg[id_map[n]] = len(neighbors)
        if len(neighbors) == 0:
            neighbors = np.array([num_of_nodes]*max_degree,dtype=np.int32)
        elif len(neighbors) > max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)
        elif len(neighbors) < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)
        adj[id_map[n], :] = neighbors
    return adj, deg

def build_k_graph(G, k):
    if k == 1:
        id_map = dict()
        cnt = 0
        for n in G.nodes():
            cnt += 1
            id_map[n] = cnt    
        return G, id_map, None, None
    elif k == 2:
        print('start building G_2...')
        G_1, id_map_1, _, _ = build_k_graph(G, 1)
        G_2 = nx.Graph()
        id_map_2 = dict()
        cnt = 0
        # add nodes of G_2
        print('start adding nodes of G_2...')
        for e in G_1.edges():
            G_2.add_node(e)
            cnt += 1
            id_map_2[e] = cnt
        # add edges of G_2
        print('start adding edges of G_2...')
        for n in G_2.nodes():
            n1, n2 = n
            for i in G.neighbors(n1):
                if not i == n2:
                    G_2.add_edge( n, make_ordered_tuple_of_2(i,n1) )
            for i in G.neighbors(n2):
                if not i == n1:
                    G_2.add_edge( n, make_ordered_tuple_of_2(i,n2) )
        # get pooling map from nodes of G_2 to nodes of G_1
        print('start calculating pooling map from nodes of G_2 to nodes of G...')
        pooling_map_2 = np.zeros((G_2.number_of_nodes()+1, 2),dtype=np.int32)
        for n in G_2.nodes():
            n1, n2 = n
            pooling_map_2[id_map_2[n]][0] = id_map_1[n1]
            pooling_map_2[id_map_2[n]][1] = id_map_1[n2]
        #get reverse pooling map from nodes of G_1 to nodes of G_2
        print('start calculating reverse pooling map from nodes of G_1 to nodes of G_2')
        reverse_pooling_map_2 = np.zeros((G.number_of_nodes()+1,G_1.graph['max_degree']),dtype=np.int32)
        a = np.zeros(G_1.number_of_nodes()+1,dtype=np.int32)
        for n in G_2.nodes():
            n1, n2 = n
            n1_ = id_map_1[n1]
            n2_ = id_map_1[n2]
            n_ = id_map_2[n]
            reverse_pooling_map_2[n1_][a[n1_]] = n_
            a[n1_] += 1
            reverse_pooling_map_2[n2_][a[n2_]] = n_
            a[n2_] += 1
        max_degree = 0
        for n in G.nodes():
            max_degree = max(max_degree, len(list(G.neighbors(n))))
        G_2.graph['max_degree'] = max_degree
        G_2.graph['max_rev_degree'] = G_1.graph['max_degree']
        print('G_2 building finished.')
        return G_2, id_map_2, pooling_map_2, reverse_pooling_map_2
    elif k == 3:
        print('start building G_3...')
        print('first build G_1 and G_2...')
        G_1, id_map_1, _, _ = build_k_graph(G, 1)
        G_2, id_map_2, pooling_map_2, reverse_pooling_map_2 = build_k_graph(G , 2)
        G_3 = nx.Graph()
        id_map_3 = dict()
        cnt = 0
        # add nodes of G_3
        print('start adding nodes of G_3...')
        for n in G_2.nodes():
            n1, n2 = n
            for i in G.neighbors(n1):
                if not i == n2:
                    t = make_ordered_tuple_of_3(n1, n2, i)
                    G_3.add_node(t)
                    if id_map_3.get(t) == None:
                        cnt += 1
                        id_map_3[t] = cnt
            for i in G.neighbors(n2):
                if not i == n1:
                    t = make_ordered_tuple_of_3(n1, n2, i)
                    G_3.add_node(t)
                    if id_map_3.get(t) == None:
                        cnt += 1
                        id_map_3[t] = cnt
        # add edges of G_3
        print('start adding edges of G_3...')
        for n in G_3.nodes():
            n1, n2, n3 = n
            if G_2.has_node((n1,n2)):
                for i in G_2.neighbors((n1,n2)):
                    i1, i2 = i
                    if i1 == n1 or i1 == n2:
                        i_ = i2
                    else:
                        i_ = i1
                    if not i_ == n3:
                        G_3.add_edge(n, make_ordered_tuple_of_3(n1, n2, i_))
            if G_2.has_node((n2,n3)):
                for i in G_2.neighbors((n2,n3)):
                    i1, i2 = i
                    if i1 == n2 or i1 == n3:
                        i_ = i2
                    else:
                        i_ = i1
                    if not i_ == n1:
                        G_3.add_edge(n, make_ordered_tuple_of_3(n2, n3, i_))
            if G_2.has_node((n1,n3)):
                for i in G_2.neighbors((n1,n3)):
                    i1, i2 = i
                    if i1 == n1 or i1 == n3:
                        i_ = i2
                    else:
                        i_ = i1
                    if not i_ == n2:
                        G_3.add_edge(n, make_ordered_tuple_of_3(n1, n3, i_))
        # get pooling map from nodes of G_3 to nodes of G_2
        print('start calculating pooling map from nodes of G_3 to nodes of G_2...')
        pooling_map_3 = np.zeros((G_3.number_of_nodes()+1, 3),dtype=np.int32)
        for n in G_3.nodes():
            n1, n2, n3 = n
            t = 0
            if G_2.has_node((n1, n2)):
                pooling_map_3[id_map_3[n]][t] = id_map_2[(n1,n2)]
                t += 1
            if G_2.has_node((n2, n3)):
                pooling_map_3[id_map_3[n]][t] = id_map_2[(n2,n3)]
                t += 1
            if G_2.has_node((n1, n3)):
                pooling_map_3[id_map_3[n]][t] = id_map_2[(n1,n3)]
                t += 1
        # get reverse pooling map from nodes of G_1 to nodes of G_2
        print('start calculating reverse pooling map from nodes of G_1 to nodes of G_2')
        a = np.zeros(G_1.number_of_nodes()+1,dtype=np.int32)
        for n in G_3.nodes():
            n1, n2, n3 = n
            a[id_map_1[n1]] += 1
            a[id_map_1[n2]] += 1
            a[id_map_1[n3]] += 1
        max_rev_degree = max(a)
        a = np.zeros(G_1.number_of_nodes()+1,dtype=np.int32)
        reverse_pooling_map_3 = np.zeros((G_1.number_of_nodes()+1,max_rev_degree),dtype=np.int32)
        for n in G_3.nodes():
            n1, n2, n3 = n
            n1_ = id_map_1[n1]
            n2_ = id_map_1[n2]
            n3_ = id_map_1[n3]
            n_ = id_map_3[n]
            reverse_pooling_map_3[n1_][a[n1_]] = n_
            a[n1_] += 1
            reverse_pooling_map_3[n2_][a[n2_]] = n_
            a[n2_] += 1
            reverse_pooling_map_3[n3_][a[n3_]] = n_
            a[n3_] += 1

        max_degree = 0
        for n in G_3.nodes():
            max_degree = max(max_degree, len(list(G_3.neighbors(n))))
        G_3.graph['max_degree'] = max_degree
        G_3.graph['max_rev_degree'] = max_rev_degree
        print('G_3 building finished.')
        return G_3, id_map_3, pooling_map_3, reverse_pooling_map_3

if  __name__ == "__main__":
    #G__ = nx.karate_club_graph()
    G = build_graph('cora')
    print(G.graph['max_degree'])
    G_2, id_map_2, pooling_map_2, reverse_pooling_map_2 = build_k_graph(G,2)
    print(G_2.number_of_nodes(),G_2.number_of_edges(),pooling_map_2.shape[1],reverse_pooling_map_2.shape[1])
    G_3, id_map_3, pooling_map_3, reverse_pooling_map_3 = build_k_graph(G,3)
    print(G_3.number_of_nodes(),G_3.number_of_edges(),pooling_map_3.shape[1],reverse_pooling_map_3.shape[1])

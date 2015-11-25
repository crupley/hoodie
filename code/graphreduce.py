import numpy as np
import networkx as nx
from collections import Counter

from graph_tool.all import *

def graph_reduce_gt(graph, filename):
    '''
    graph with dist and btw attributes
    filename to store the cutlist
    '''
    g = Graph(graph)
    cuts = []
    with open(filename, 'wb') as f:
        f.write('source,target,num_edges,timestamp\n')
    while g.num_edges() > 0:
        betweenness(g, eprop = g.ep.btw, weight = g.ep.dist)

        meidx = np.argmax(g.ep.btw.fa)
        maxedge = list(g.edges())[meidx]
        metup = eval(str(maxedge))
        cuts.append(metup)
        
        wstr = '%d,%d,%d,%f\n' % (metup[0], metup[1], g.num_edges(), time())
        with open(filename, 'a') as f:
            f.write(wstr)
        
        g.remove_edge(maxedge)
    return cuts


def build_graph(edges, distances, graph_name=None):
    '''
    edges: df with node1, node2, and dist
    '''

    g = Graph(directed=False)

    # create graph properties
    gp = g.new_graph_property('string')
    g.graph_properties['Name'] = gp
    g.graph_properties['Name'] = graph_name
    eprop = g.new_edge_property('float')
    g.edge_properties['dist'] = eprop #feature distance
    g.edge_properties['btw'] = eprop  #betweenness

    # create edges and edge weights
    g.add_edge_list(zip(edges.node1, edges.node2))
    for i, edge in enumerate(g.edges()):
        g.ep.dist[edge] = distances.iloc[i]

    remove_parallel_edges(g)
    return g

import numpy as np

from graph_tool.all import *

from code.featurize import find_closest

"""
Collection of functions for building and analyzing a graph model.
"""


def make_edges(latlondf, **kwargs):
    """
    Generate a list of edges between adjacent latitude/longitude points

    Args:
        latlondf : pandas dataframe of latitude/longitude coordinates
            with features 'lat' and 'lon', indexed by node number
        kwargs : arguments to be passed to featurize.find_closest
    Returns:
        edges pandas dataframe of node pairs where edges exist
    """
    def edgelambda(x):
        """featurize.find_closest for use in apply to dataframe"""
        return find_closest(x, latlondf)

    # return list of connected nodes for each node in latlondf
    n = latlondf.apply(edgelambda, axis=1)

    edges = pd.DataFrame(columns=['node1', 'node2'])

    # convert 'list of lists' to flatter dataframe
    for node1 in n.index:
        for node2 in n.ix[node1]:
            newrow = {'node1': node1, 'node2': node2}
            edges = edges.append(newrow, ignore_index=True)
    edges.index.name = 'edge'
    return edges.astype('int')


def build_graph(edges, distances, graph_name=None):
    """
    Creates a graph-tool graph object from lists of edges and
    feature distances

    Args:
        edges : pandas dataframe, list of edges by node pairs
            containing features 'node1' and 'node2'
        distances : pandas series/dataframe, length of edges
            containing the feature distance between adajacent nodes
    Returns:
        graph-tool graph object
    """
    g = Graph(directed=False)

    # create graph properties
    gp = g.new_graph_property('string')
    g.graph_properties['Name'] = gp
    g.graph_properties['Name'] = graph_name
    eprop = g.new_edge_property('float')
    g.edge_properties['dist'] = eprop  # feature distance
    g.edge_properties['btw'] = eprop  # betweenness

    # create edges and edge weights
    g.add_edge_list(zip(edges.node1, edges.node2))
    for i, edge in enumerate(g.edges()):
        g.ep.dist[edge] = distances.iloc[i]

    remove_parallel_edges(g)
    return g


def graph_reduce_gt(graph, filename=None):
    '''
    graph with dist and btw attributes
    filename to store the cutlist
    '''
    """
    Reduces a graph, edge-by-edge, in order of a combination of
    most-connected and highest feature distance edges using
    graph betweenness similarity.

    Args:
        graph : graph-tool graph object to be reduced
        filename : file location to save results in csv
    Returns:
        list of node pair tuples in order from first cut to last
    """
    # make a copy
    g = Graph(graph)

    cuts = []
    if filename is not None:
        with open(filename, 'wb') as f:
            f.write('source,target,num_edges,timestamp\n')

    # reduce graph
    while g.num_edges() > 0:
        betweenness(g, eprop=g.ep.btw, weight=g.ep.dist)

        meidx = np.argmax(g.ep.btw.fa)  # max edge index
        maxedge = list(g.edges())[meidx]  # max edge nodes
        metup = eval(str(maxedge))  # max edge tuple
        cuts.append(metup)

        if filename is not None:
            wvalues = (metup[0], metup[1], g.num_edges(), time())
            wstr = '%d,%d,%d,%f\n' % wvalues
            with open(filename, 'a') as f:
                f.write(wstr)

        g.remove_edge(maxedge)
    return cuts


import numpy as np
import pandas as pd
import networkx as nx
import os
import matplotlib
import cPickle as pickle
import math

from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial import Delaunay
from shapely.geometry import mapping
from pyshp import shapefile
import shapely.geometry as geometry
from shapely.geometry import Polygon, mapping
import shapely.ops
from shapely.ops import cascaded_union, polygonize

from code.featurize import fdist
from code.shapefiles import merge_shapefiles, make_shapefiles

"""
Final function set for creating neighborhood clusters
"""

# convert feature numbers to feature name
FDICT = {0: 'taxable_value',
         1: 'grocery',
         2: 'restaurant',
         3: 'retail',
         4: 'ncrimes',
         5: 'sgnf',
         6: 'avg_hh_size',
         7: 'population',
         8: 'walkscore'}

# convert feature name to proper feature name
FNAMES = {'taxable_value': 'Property Value',
          'grocery': 'Grocery',
          'restaurant': 'Restaurants',
          'retail': 'Retail',
          'ncrimes': 'Crime',
          'sgnf': 'Female:Male ratio',
          'avg_hh_size': 'Household Size',
          'population': 'Population',
          'walkscore': 'Walkscore'}


def mapno2list(s):
    """Convert from map number string of form 'f1f2f3' to list of int"""
    return [int(s[i] + s[i+1]) for i in range(len(s)) if i % 2 == 0]


def list2mapno(featurenumlist):
    """Convert from list of int to mapno string"""
    f = tuple(featurenumlist)
    return '%02d' * len(f) % f


def mapno2fname(s):
    """Convert from map number string of form 'f1f2f3' to feature name"""
    featurenumlist = mapno2list(s)
    return [FDICT[i] for i in featurenumlist]


def bigsize(row):
    """
    Remove edge in graph and calculate biggest cluster in graph after.
    To be used in .apply on pandas dataframe

    Args:
        row : row from a pandas DataFrame (pandas Series) with
            'source' and 'target' node features
    Returns:
        number of nodes in largest cluster of graph, int
    """
    g.remove_edge(row.source, row.target)
    return len(max(nx.connected_components(g), key=len))


def cutcon(row, graph):
    """
    Remove edge in graph and calculate biggest cluster in graph after.
    To be used in .apply on pandas dataframe

    Args:
        row : row from a pandas DataFrame (pandas Series)
    Returns:
        number of nodes in largest cluster of graph, int
    """
    graph.remove_edge(row.source, row.target)
    return nx.number_connected_components(graph)


def cutrow(row, graph):
    """
    Remove edges in graph.
    To be used in .apply on pandas dataframe of edges.

    Args:
        row : row from a pandas DataFrame (pandas Series)
            containing 'source' and 'target' features (edges)
    Returns:
        None
    """
    graph.remove_edge(row.source, row.target)
    return


def make_graph(cutdf):
    """
    Convert dataframe of a list of edges into a networkx graph object

    Args:
        cutdf : pandas dataframe containing list of edges with
            features 'source' and 'target' containing node numbers
    Returns:
        networkx graph object
    """
    g = nx.from_pandas_dataframe(cutdf, 'source', 'target')
    return g


def assign_clusters(nodelist, graph):
    """
    Assigns a cluster number to each node in a graph

    Args:
        nodelist : nodes in the graph to be assigned to a cluster,
            iterable of ints
        graph : networkx graph object
    Returns:
        Assigned cluster numbers, pandas Series indexed by node number, ints
    """
    cc = list(nx.connected_components(graph))

    cnum = pd.Series(-1, index=nodelist)
    for node in nodelist:
        for i, cluster in enumerate(cc):
            if node in cluster:
                cnum.ix[node] = i
    return cnum


def row_errorsq(row, cluster_means):
    """
    Calculate the feature distance between each node and its cluster mean,
    to be used in an .apply to pandas dataframe

    Args:
        row : row of a pandas dataframe containing columns of features and
            one of assigned cluster number designated 'cnum'
        cluster_means : pandas dataframe of the mean of each feature in 'row'
            indexed by cluster number.
    Returns:
        feature distance, float
    """
    rowf = row.drop(['cnum'])
    return (fdist(rowf, cluster_means.ix[int(row.cnum)]))


def wcss(featuredf, cnum):
    """
    Calculate the within cluster sum of squares error (wcss)

    Args:
        featuredf : pandas dataframe of features
        cnum : assigned cluster numbers, same length as featuredf,
            list-like of int
    Returns:
        wcss error, float
    """
    df = featuredf.copy()
    df['cnum'] = cnum
    cluster_means = df.groupby('cnum').mean()
    df['errors'] = df.apply(lambda x: row_errorsq(x, cluster_means),
                            axis=1)
    return df.groupby('cnum').sum()['errors']


def cut2cluster(featurelist, nclusters, allowed_nodes=None):
    """
    Using a list of cuts from a betweenness similarity graph reduction,
    returns the assigned cluster number for each node for the given
    number of clusters.

    Args:
        featurelist : combination of features to use, string of numbers
            designating features from FDICT in form 'f1f2f3', e.g. '010507'
        nclusters : number of clusters to split nodes into, int. Minimum
            is the starting number of clusters of the full graph. If nclusters
            is greater than number of nodes in graph, will assign each
            node to unique cluster
        allowed_nodes : node numbers of allowed nodes if only
        certain nodes are to be considered, list-like of ints
    Returns:
        Assigned cluster numbers, pandas Series indexed by node number, ints
    """
    # load from cut list file
    fn = 'results/CL' + featurelist + '.csv'
    edges = pd.read_csv(fn)
    edges = edges[['source', 'target']]
    graph = make_graph(edges)

    # reduce graph according to cut list until nclusters are achieved
    for i in edges.index:
        ncc = cutcon(edges.ix[i], graph)
        if ncc >= nclusters:
            break

    nodeset = set(edges.source).union(set(edges.target))

    if allowed_nodes is not None:
        # remove unallowed nodes from nodelist
        nodeset = nodeset.intersection(set(allowed_nodes))

        # remove unallowed nodes from graph
        unallowed_nodesg = set(graph.nodes()).difference(set(allowed_nodes))
        graph.remove_nodes_from(unallowed_nodesg)
    nodelist = list(nodeset)

    # assign cluster numbers
    return assign_clusters(nodelist, graph)


def feature_bars(featuredf, cnum, plot=False, **kwargs):
    """
    Normalizes features to zero mean and unit range for use in bar plots.
    Can optionally produce bar plot.

    Args:
        featuredf : features to be scaled indexed by cluster number,
            length: number of nodes,
            width: number of features,
            pandas dataframe
        cnum : assigned cluster number for each node
            length: number of nodes
            list-like of ints
        plot : boolean, create barplot of scaled features
        kwargs : additional arguments to be passed to
            pandas dataframe.plot(kind='bar')
    Returns:
        scaled feature value for each feature and cluster
            length : number of clusters
            width : number of features
            pandas dataframe
    """
    df = featuredf.copy()
    df['cnum'] = cnum

    # scale values 0-1 then subtract average
    df = df.groupby('cnum').mean()
    df = df.sub(df.min(axis=0))
    df = df.div(df.max(axis=0))
    df = df.sub(df.mean(axis=0))

    if plot:
        df.T.plot(kind='bar', subplots=True, sharey=True, **kwargs)

    return df


def most_similar(featuredf, cluster_labels):
    """
    Pairwise feature comparison between each cluster

    Args:
        featuredf : node features,
            length: number of nodes,
            width: number of features,
            pandas dataframe
        cluster_labels : assigned cluster number for each node
            length: number of nodes
            list-like of ints
    Returns:
        Feature distances, pandas dataframe, n clusters x n clusters
    """
    cluster_means = featuredf.groupby(cluster_labels).mean()
    df = pd.DataFrame(pairwise_distances(cluster_means,
                      metric='l2'),
                      index=cluster_means.index,
                      columns=cluster_means.index)
    return df


def gencolors(n, cmap='jet'):
    """Generates list of hex colors n long in given matplotlib colormap"""
    c = matplotlib.cm.get_cmap('Set1')
    cols = c(np.linspace(0, 1, n))
    clist = [matplotlib.colors.rgb2hex(rgb) for rgb in cols]
    return clist


def to_rghex(n):
    """Converts number, [0-1], into hex color on green-to-red gradient"""
    return matplotlib.colors.rgb2hex([n, 1-n, 0, 1])


def rg_colormatrix(sim):
    """
    Creates red-green matrix of colors from similarity matrix
    covering the range of distances in the similarity matrix

    Args:
        sim : similarity matrix as created by most_similar, pandas dataframe
    Returns:
        matrix of hex colors of same dimension as sim, pandas dataframe
    """
    # normalize similarity values 0-1
    normed = sim / sim.max().max()
    # convert each 0-1 value to hex color on green-to-red gradient
    return normed.applymap(to_rghex)


def list_(*args):
    """Flattens list of lists"""
    return list(args)


def merge_map_data(path, featuredf, store=False):
    """
    For each set of features, take results of graph reduction,
    assign cluster numbers, and compute all other data for insertion
    into final geojson file.

    Args:
        path : path to directory containing graph cut list files, string
        featuredf : scaled features, pandas dataframe
    Returns:
        Data for insertion into geojson, pandas dataframe
    """
    # get filenames
    files = os.listdir(path)
    files = [f[2:-4] for f in files if f[:2] == 'CL']

    # null map
    files.remove('xx')

    # only allow 3 or less features
    mapnos = [f for f in files if len(f) <= 6]

    fnums = [mapno2list(f) for f in mapnos]

    # column names
    fnames = map(lambda x: [FDICT[n] for n in x], fnums)

    # fixed number of clusters
    nclustersmax = 28

    # make null map
    cnum = cut2cluster('xx', nclustersmax, allowed_nodes=featuredf.index)

    # retain only mutual nodes
    nodelist = set(featuredf.index).intersection(set(cnum.index))
    featuredf = featuredf.ix[nodelist]
    cnum = cnum.ix[nodelist]
    nclusters = len(cnum.unique())

    # compute data

    # similarity colors
    rgmatrix = rg_colormatrix(most_similar(featuredf, cnum))
    # feature bar graph data
    fbars = feature_bars(featuredf[FDICT.values()], cnum)

    # shape file polygons
    fn = 'data/uscensus/tl_2010_06075_tabblock10/tl_2010_06075_tabblock10.dbf'
    mergedf = merge_shapefiles(featuredf[['lat', 'lon']], fn)
    polys = make_shapefiles(featuredf[['lat', 'lon']], mergedf.polys, cnum)

    # compile into single dataframe
    alldf = pd.DataFrame({'cnum': cnum.unique(),
                          'polygon': polys})
    alldf['rgmatrix'] = map(lambda x: list(rgmatrix.ix[x]), cnum.unique())
    alldf['mapno'] = ''
    alldf['fbars'] = map(list, fbars.round(2).values)

    # store results
    if store:
        alldf.to_csv('results/alldf.csv')

    # make all other maps
    for i, f in enumerate(mapnos):
        cnum = cut2cluster(f, nclustersmax, allowed_nodes=featuredf.index)
        rgmatrix = rg_colormatrix(most_similar(featuredf, cnum))

        fbars = feature_bars(featuredf[fnames[i]], cnum)
        polys = make_shapefiles(featuredf[['lat', 'lon']],
                                mergedf.polys, cnum)

        onedf = pd.DataFrame({'cnum': cnum.unique(),
                              'polygon': polys})
        onedf['rgmatrix'] = map(lambda x: list(rgmatrix.ix[x]), cnum.unique())
        onedf['mapno'] = f
        onedf['fbars'] = map(list, fbars.round(2).values)

        # append results after each map
        if store:
            with open('results/alldf.csv', 'a') as storefile:
                onedf.to_csv(storefile, header=False)

        alldf = pd.concat((alldf, onedf), axis=0, ignore_index=True)

    return alldf


def make_json(cnum, polys, rgmatrix, mapno, fbars):
    """
    Take data from merge_map_data and convert to geojson format

    Args:
        cnum : assigned cluster number, pandas Series/dataframe of ints
        polys : shapely polygon for each cluster,
            pandas Series/dataframe of shapely objects
        rgmatrix : hex colors for each cluster,
            pandas Series/dataframe of lists each length n clusters
        mapno : assigned map number of each cluster,
            pandas Series/dataframe of str
        fbars : data for feature bar graph, pandas Series/dataframe
            of list of float, length n features

        All args of length number of clusters
    Returns:
        geojson-formatted dict
    """

    # column names
    fnamesc = map(lambda x: [FDICT[n] for n in mapno2list(x)], mapno)

    # proper names
    fnames = map(lambda x: [FNAMES[n] for n in x], fnamesc)

    featurelist = []
    for i in xrange(len(cnum)):
        featurelist.append({"type": "Feature",
                            "properties": {
                                "rgmat": rgmatrix.iloc[i],
                                "mapno": mapno.iloc[i],
                                "neibno": cnum.iloc[i],
                                "bars": map(list_, fnames[i],
                                            fbars.iloc[i]),
                                "visible": False
                                },
                            "geometry": mapping(polys.iloc[i])
                            })

    geojson = {"type": "FeatureCollection",
               "features": featurelist}

    return geojson


def load_featuredf():
    """
    Loads feature dataframe and removes unwanted data

    Args:
        None
    Returns:
        feature pandas dataframe
    """
    with open('featuresdf.pkl', 'rb') as f:
        fdf = pickle.load(f)

    # exclude Treasure Island
    fdf = fdf[(fdf.lon < -122.375) | (fdf.lat < 37.805)]

    # exclude piers off Mission Bay
    fdf = fdf.drop([5662, 6742], axis=0)

    return fdf


if __name__ == '__main__':
    fdf = load_featuredf()

    # may take a while
    alldf = merge_map_data('results', fdf, store=True)

    gjson = make_json(alldf.cnum, alldf.polygon, alldf.color,
                      alldf.rgmatrix, alldf.mapno, alldf.fbars)
    with open('results/geo.json', 'wb') as f:
        f.write(json.dumps(gjson))

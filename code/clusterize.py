
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
Final function for creating neighborhood clusters
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
	return [int(s[i] + s[i+1]) for i in range(len(s)) if i%2 == 0]


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
            if node in cluster: cnum.ix[node] = i
    return cnum


def row_errorsq(row, cluster_means):
    rowf = row.drop(['cnum'])
    return (fdist(rowf, cluster_means.ix[int(row.cnum)]))


def wcss(featuredf, cnum):
    df = featuredf.copy()
    df['cnum'] = cnum    # features = df.drop(['lat', 'lon'], axis = 1)
    cluster_means = df.groupby('cnum').mean()
    df['errors'] = df.apply(lambda x: row_errorsq(x, cluster_means),
    							axis = 1)
    return df.groupby('cnum').sum()['errors']


def elbow_plot(cutdf, featuredf, maxk=30):
	'''
	calculate wcss vs k

	cutdf: list of cuts containing columns 'source', 'target', and 'nclusters'

	'''

	ks = []
	wcsses = []

	graph = make_graph(cutdf)

	nclusterslast = nx.number_connected_components(graph)
	# ks.append(nclusterslast)
	# wcsses

	for i in cutdf.index:
		nclusters = cutcon(cutdf.ix[i], graph)
		if nclusters > nclusterslast:
			# print i, nclusters
			nclusterslast = nclusters
			ks.append(nclusters)
			cnum = assign_clusters(featuredf.index, graph)
			wcsses.append(wcss(featuredf, cnum).sum())
			if nclusters >= maxk: return ks, wcsses


	# for k in ks:
	#     # rebuild full graph
	#     #g = nx.from_pandas_dataframe(edges, 'source', 'target')
	#     g = graph.copy()

	#     # reduces graph to i clusters
	#     cutdf[cutdf.nclusters <= k].apply(lambda x: cutrow(x, g), axis=1)

	#     # assign cluster numbers
	#     featuredf['cnum'] = assign_clusters(featuredf.index, g)

	#     wcss = wcss(featuredf)
	#     wcsses.append(wcss.ix[wcss.index >= 0].sum())
	#     print 'k:', k, 'WCSSE:', wcss.ix[wcss.index >= 0].sum(),
	#     print g.number_of_edges()

	return ks, wcsses


def cut2cluster(featurelist, nclusters, allowed_nodes=None):
	'''
	returns Series with cluster number indexed by node number

	featurelist: '010507'
	nclusters: int
	'''

	fn = 'results/CL' + featurelist + '.csv'
	edges = pd.read_csv(fn)
	edges = edges[['source', 'target']]
	graph = make_graph(edges)

	# reduce graph according to cut list until nclusters are achieved
	for i in edges.index:
		ncc = cutcon(edges.ix[i], graph)
		if ncc >= nclusters: break


	# # add count of number of clusters after each cut
	# g = graph.copy()
	# edges['nclusters'] = edges.apply(lambda x: cutcon(x, g), axis=1)

	# # reduces graph to i clusters
	# g = graph.copy()
	# edges[edges.nclusters <= nclusters].apply(lambda x: cutrow(x, g), axis=1)

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
	'''
	barplot of features grouped by cluster number
	'''
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
	'''
	returns feature distance from cluster_number to all others
	'''
	cluster_means = featuredf.groupby(cluster_labels).mean()
	df =  pd.DataFrame(pairwise_distances(cluster_means,
						metric='l2'),
					   index = cluster_means.index,
					   columns = cluster_means.index)
	return df


def gencolors(n, cmap='jet'):
	c = matplotlib.cm.get_cmap('Set1')
	clist = [matplotlib.colors.rgb2hex(rgb) for rgb in c(np.linspace(0,1,n))]
	return clist


def to_rghex(n):
    return matplotlib.colors.rgb2hex([n, 1-n, 0, 1])


def rg_colormatrix(sim):
    normed = sim / sim.max().max()
    return normed.applymap(to_rghex)


def list_(*args): return list(args)


def make_json(cnum, polys, rgmatrix, mapno, fbars):
	
	# column names
	fnamesc = map(lambda x: [FDICT[n] for n in mapno2list(x)], mapno)

	# proper names
	fnames = map(lambda x: [FNAMES[n] for n in x], fnamesc)


	featurelist = []
	for i in xrange(len(cnum)):
	    featurelist.append({"type": "Feature",
	                        "properties": {
	                        # "color": clist.iloc[i],
	                        "rgmat": rgmatrix.iloc[i],
	                        "mapno": mapno.iloc[i],
	                        "neibno": cnum.iloc[i],
	                        "bars" : map(list_, fnames[i],
	                        			 fbars.iloc[i]),
	                        "visible": False
	                        },
	                        "geometry": mapping(polys.iloc[i])
	                        })
	    
	geojson = {"type": "FeatureCollection",
	           "features": featurelist}

	return geojson


def merge_map_data(path, featuredf, store=False):
	files = os.listdir(path)
	files = [f[2:-4] for f in files if f[:2] == 'CL']
	files.remove('xx')

	# incomplete cut list
	files.remove('000104')

	mapnos = [f for f in files if len(f) <= 6]
	# mapnos = ['010405']

	fnums = [mapno2list(f) for f in mapnos]

	# column names
	fnames = map(lambda x: [FDICT[n] for n in x], fnums)

	nclustersmax = 28

	### make null map
	cnum = cut2cluster('xx', nclustersmax, allowed_nodes=featuredf.index)

	# retain only mutual nodes
	nodelist = set(featuredf.index).intersection(set(cnum.index))
	featuredf = featuredf.ix[nodelist]
	cnum = cnum.ix[nodelist]
	nclusters = len(cnum.unique())

	# clist = gencolors(nclusters)
	rgmatrix = rg_colormatrix(most_similar(featuredf, cnum))
	fbars = feature_bars(featuredf[FDICT.values()], cnum)

	fn = 'data/uscensus/tl_2010_06075_tabblock10/tl_2010_06075_tabblock10.dbf'
	mergedf = merge_shapefiles(featuredf[['lat', 'lon']], fn)
	polys = make_shapefiles(featuredf[['lat', 'lon']], mergedf.polys, cnum)

	# pdb.set_trace()

	alldf = pd.DataFrame({'cnum': cnum.unique(),
                      'polygon': polys})
	# alldf['color'] = clist
	alldf['rgmatrix'] = map(lambda x: list(rgmatrix.ix[x]), cnum.unique())
	alldf['mapno'] = ''
	alldf['fbars'] = map(list, fbars.round(2).values)

	# store results
	if store:
		alldf.to_csv('results/alldf.csv')

	# make all other maps
	for i, f in enumerate(mapnos):
		print f
		cnum = cut2cluster(f, nclustersmax, allowed_nodes=featuredf.index)
		rgmatrix = rg_colormatrix(most_similar(featuredf, cnum))

		# cnum = cnum.ix[nodelist]

		fbars = feature_bars(featuredf[fnames[i]], cnum)
		polys = make_shapefiles(featuredf[['lat', 'lon']],
								mergedf.polys, cnum)

		onedf = pd.DataFrame({'cnum': cnum.unique(),
                      'polygon': polys})

		# onedf['color'] = clist
		onedf['rgmatrix'] = map(lambda x: list(rgmatrix.ix[x]), cnum.unique())
		onedf['mapno'] = f
		onedf['fbars'] = map(list, fbars.round(2).values)

		if store:
			with open('results/alldf.csv', 'a') as storefile:
			    onedf.to_csv(storefile, header=False)

		alldf = pd.concat((alldf, onedf), axis=0, ignore_index=True)

	return alldf


def load_featuredf():
	with open('featuresdf.pkl', 'rb') as f:
		fdf = pickle.load(f)

	# exclude Treasure Island
	fdf = fdf[(fdf.lon < -122.375) | (fdf.lat < 37.805)]

	# exclude piers off Mission Bay
	fdf = fdf.drop([5662, 6742], axis=0)

	return fdf


if __name__ == '__main__':
	fdf = load_featuredf()

	# may take some time
	alldf = merge_map_data('results', fdf, store=True)

	gjson = make_json(alldf.cnum, alldf.polygon, alldf.color,
					  alldf.rgmatrix, alldf.mapno, alldf.fbars)
	with open('results/geo.json', 'wb') as f:
		f.write(json.dumps(gjson))

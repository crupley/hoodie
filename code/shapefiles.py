# dealing with shapefiles

import numpy as np
import pandas as pd
import math

from scipy.spatial import Delaunay
from pyshp import shapefile
import shapely.geometry as geometry
from shapely.geometry import Polygon, mapping
import shapely.ops
from shapely.ops import cascaded_union, polygonize

"""
Collection of functions for loading, manipulating, and merging shapefiles.
"""


def sf_to_df(filename):
    """
    Converts shapefile records from US Census 2010 to dataframe.

    Args:
        filename: string
    Returns:
        shapefile records; pandas DataFrame
    """
	sf = shapefile.Reader(filename)
	records = np.array(sf.records())
	rdf = pd.DataFrame(records)

	colnames = ['state',
				'county',
				'tract',
				'block',
				'geoid',
				'name',
				'mtfcc',
				'ur',
				'uace',
				'funcstat',
				'land_area',
				'water_area',
				'lat',
				'lon']
	rdf.columns = colnames

	rdf.geoid = rdf.geoid.astype('int')
	rdf.land_area = rdf.land_area.astype('int')
	rdf.water_area = rdf.water_area.astype('int')
	rdf.lat = rdf.lat.astype('float')
	rdf.lon = rdf.lon.astype('float')

	# drop empty columns
	rdf.drop(['ur', 'uace', 'funcstat'], axis=1, inplace=True)
	return rdf


def get_shapefiles(filename):
    """
    Converts .dbf shapefile information to list of shapely polygons

    Args:
        filename: string
    Returns:
        list of shapely objects
    """
	sf = shapefile.Reader(filename)
	polys = map(lambda x: Polygon(x.points), sf.shapes())
	return polys


def merge_shapefiles(latlon, filename):
	"""
    Merge polygon list with associated lat/lon coordinates in dataframe

    Args:
        latlon : dataframe with 'lat' and 'lon' features
        filename : path to .dbf file containing shapefiles
    Returns:
        merged pandas dataframe with lat, lon and shapely objects
    """
	shapedf = sf_to_df(filename)
	shapedf = shapedf[['lat', 'lon']]
	shapedf['polys'] = get_shapefiles(filename)
	mergedf = latlon.merge(shapedf, left_on=['lat', 'lon'],
					   	   right_on=['lat', 'lon'])
	return mergedf


def make_shapefiles(latlon, polys, cnum):
	"""
    Creates merged polygons according to cluster number.
    Combines convex hull of lat/lon points with census shapefile polygons.

    Args:
        latlon : dataframe with 'lat' and 'lon' features
        polys : list of shapely polygons from merge_shapefiles
        	list-like, same length as latlon
        cnum : cluster number for each lat/lon point, list-like,
        	same length as latlon
    Returns:
        list of shapely polygons, one for each unique cluster number
    """
    # convert polygon list into numpy array
	parr = pd.Series(polys)

	# mask to eliminate invalid polygons (self-crossing edges, interior loops)
	validbool = map(lambda x: x.is_valid, parr)

	df = latlon.copy()
	df['cnum'] = cnum

	# Create polygons from convex hull-like estimation
	cxpolys = make_polys(df)

	# Create neighborhoods as merged census shapefiles
	# fill in gaps by merging with convex hull
	neibs = []
	for i, c in enumerate(cnum.unique()):
	    sub = parr[(np.array(cnum) == c) & (np.array(validbool))]
	    group = shapely.ops.cascaded_union(list(sub))
	    neibs.append(group.union(cxpolys[i]))

	# eliminate overlaps, take shape set difference with each other shape
	fneibs = []
	ln = len(neibs)
	for i in range(ln):
	    fneibs.append(neibs[i])
	    for j in range(i+1, ln):
	        fneibs[i] = fneibs[i].difference(neibs[j])

	return fneibs


def alpha_shape(points, alpha):
    """

    ** function from
    http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/ **
    
    Compute the alpha shape (concave hull) of a set
    of points.
 
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
 
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
 
    coords = np.array([point.coords[0]
                       for point in points])
 
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
 
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
 
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
 
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c)) + 1e-10
        circum_r = a*b*c/(4.0*area)
 
        # Here's the radius filter.
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
 
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


def make_polys(df):
    '''
    df: lat, lon, cnum
    returns: list of polygons
    '''
	"""
    Creates merged polygons according to cluster number from census polygons
    using a convex hull-based approach.

    Args:
        df : dataframe with 'lat', 'lon', and 'cnum';
        	latitude/longitude coordinates, assigned cluster number
    Returns:
        list of shapely polygons, one for each unique cluster number
    """
    gs = df.cnum.unique()

    polys = []
    for g in gs:
        groupn = df[df.cnum == g]
        # convert lat/lon into collection of point coordinates
        points = geometry.MultiPoint(zip(groupn.lon, groupn.lat))
        if len(points) > 2:  # required to make a polygon
            polygon, edge_points = alpha_shape(points, alpha=300)
        else:
            polygon = points
        polys.append(polygon)
    return polys





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

import scipy.interpolate
from scipy.interpolate import Rbf
from sklearn.preprocessing import StandardScaler

from code.makedbs import get_db

"""
Functions to turn data in database into actionable features
"""


def bin_interpolate(datax, datay, dataz, interpx, interpy, smooth=0):
    """
    Bin location-specific data via interpolation

    Args:
        datax, datay, dataz : x, y, z coordinates of raw data
        interpx, interpy : x, y coordinates of destination bins
        smooth : level of smoothing. 0 = no smoothing
    Returns:
        interpolated z-values
    """
    rbf = scipy.interpolate.Rbf(datax, datay, dataz,
                                function='linear', smooth=smooth)
    interpz = rbf(interpx, interpy)
    return interpz


def window(df, latmin, latmax, lonmin, lonmax):
    """
    Window down a dataframe of latitude and longitude columns
    within specified latitude and longitude boundaries

    Args:
        df : pandas dataframe to be windowed with 'lat' and 'lon' columns
        latmin, latmax, lonmin, lonmax : latitude and longitude min/max, float
    Returns:
        windowed dataframe
    """
    df = df[df.lat > latmin]
    df = df[df.lat < latmax]
    df = df[df.lon > lonmin]
    df = df[df.lon < lonmax]
    return df


def dist(lat1, lon1, lat2, lon2):
    """
    Calculate distance between points via triangulation

    Args:
        lat1, lon1 : latitude/longitude of start point, floats
        lat2, lon2 : latitude/longitude of destination point, floats
    Returns:
        distance, float
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return np.sqrt(dlat**2 + dlon**2)


def dist_by_node(node1, node2, df):
    """
    Calculate distance using node numbers

    Args:
        node1, node2 : start, end node number for calculation
        df : pandas dataframe with index corresponding to node number
    Returns:
        distance, float
    """
    lat1, lon1 = df.ix[node1][['lat', 'lon']]
    lat2, lon2 = df.ix[node2][['lat', 'lon']]
    return dist(lat1, lon1, lat2, lon2)


def fdist(f1, f2):
    """
    Feature Distance. Computes Euclidean distance between two feature vectors.

    Args:
        f1, f2 : feature vector, numpy array or pandas Series
    Returns:
        feature distance, float
    """
    return np.linalg.norm(f1-f2)**2


def fdist_by_node(n1, n2, df):
    """
    Calculate feature distance using node numbers

    Args:
        n1, n2 : start, end node number for calculation
        df : pandas dataframe indexed by node number
    Returns:
        distance, float
    """
    f1 = df.ix[n1]
    f2 = df.ix[n2]
    return fdist(f1, f2)


def angle(testnode, neib1, neib2):
    """
    Calculate the angle between the 2 edges connecting 3 nodes

    Args:
        testnode : central node, pandas dataframe/Series
            with 'lat' and 'lon' features
        neib1, neib2 : connected nodes, pandas dataframe/Series
            with 'lat' and 'lon' features
    Returns:
        angle in degrees, float [0, 180]
    """
    v1 = testnode[['lat', 'lon']] - neib1[['lat', 'lon']]
    v2 = testnode[['lat', 'lon']] - neib2[['lat', 'lon']]
    num = v1.dot(v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(num / denom) * 180 / np.pi


def find_closest(testnode, df, nneibs=4, anglelim=45):
    """
    Identify the closest neighbors to a node restricted by number and angle
    between adjacent nodes

    Args:
        testnode : the central node, pandas dataframe/Series
            with 'lat' and 'lon' features
        df : dataframe of nodes with 'lat' and 'lon' location features
            indexed by node number
        nneibs : maximum number of neighbors, int
        anglelim : minimum angle allowed between node edges to testnode
    Returns:
        list of closest nodes by node number, list of ints
    """
    # only allow nodes within certain lat/lon range
    neibs = window(df,
                   testnode.lat - 0.005,
                   testnode.lat + 0.005,
                   testnode.lon - 0.003,
                   testnode.lon + 0.003)

    neibs = neibs[['lat', 'lon']]
    # remove the testnode from the list of neighbors
    neibs.drop(testnode.name, axis=0, inplace=True)

    # no neighbors within range
    if neibs.shape[0] == 0:
        return []

    # calculate distance from testnode to all surrounding nodes and sort
    neibs['dist'] = neibs.apply(lambda x: dist(x.lat, x.lon,
                                               testnode.lat, testnode.lon),
                                axis=1)
    neibs = neibs.sort_values('dist')

    # iterate through list until nneibs is reached
    closeix = [neibs.index[0]]
    i = 0
    while len(closeix) < nneibs:
        i += 1
        if len(neibs.index) < (i + 1):
            break
        idx = neibs.index[i]
        angs = [angle(testnode, neibs.ix[idx], neibs.ix[j]) for j in closeix]
        if np.all(np.array(angs) > anglelim):
            closeix.append(idx)
    return closeix


class featurizer():
    """Featurizer class. Generates and operates on neighborhood features

    Args:
        None
    Returns:
        featurizer class object
    """

    def __init__(self):
        self.features = pd.DataFrame()
        self.fsmooth = pd.DataFrame()

        # San Francisco city boundaries
        self.latmin = 37.70784
        self.latmax = 37.8195
        self.lonmin = -122.5185
        self.lonmax = -122.35454

        self.latbins = np.linspace(self.latmin, self.latmax, 101)
        self.lonbins = np.linspace(self.lonmin, self.lonmax, 101)

        self.shapefile = self.window(get_db('usc_shapefile'))

        self.nodelat = self.shapefile.lat
        self.nodelon = self.shapefile.lon

        self.edges = None

        # list of features available, feature names
        self.allfeatures = ['taxable_value', 'grocery', 'restaurant',
                            'retail', 'ncrimes', 'sgnf',
                            'avg_hh_size', 'population', 'walkscore']

        # list of all sql tables used
        self.alltables = ['assessment', 'business', 'sfpd',
                          'usc_age_gender', 'usc_household',
                          'usc_pop', 'walkscore']

        # Formatted feature name dict
        self.featurenames = {'taxable_value': 'Property Value',
                             'grocery': 'Grocery',
                             'restaurant': 'Restaurants',
                             'retail': 'Retail',
                             'ncrimes': 'Crime',
                             'sgnf': 'Female:Male ratio',
                             'avg_hh_size': 'Household Size',
                             'population': 'Population',
                             'walkscore': 'Walkscore'}

        # smoothing level for each feature
        self.smoothing = {'taxable_value': 0.01,
                          'grocery': 0.1,
                          'restaurant': 0.01,
                          'retail': 0.3,
                          'ncrimes': 0.1,
                          'sgnf': 0.01,
                          'avg_hh_size': 0.1,
                          'population': 1,
                          'walkscore': 0}

    def window(self, df):
        """
        Window down a dataframe of latitude and longitude columns
            within latitude and longitude boundaries specified in
            object

        Args:
            df : pandas dataframe to be windowed with 'lat' and 'lon' columns
        Returns:
            windowed dataframe
        """
        df = df[df.lat > self.latmin]
        df = df[df.lat < self.latmax]
        df = df[df.lon > self.lonmin]
        df = df[df.lon < self.lonmax]
        df = df.reset_index()
        return df

    def binlatlon(self, df):
        """
        Bin dataframe lat/long locations according to object's .latbins
            and .lonbins arrays via pandas' cut method

        Args:
            df : pandas dataframe to be windowed with 'lat' and 'lon' columns
        Returns:
            binned dataframe
        """
        df['lat_cut'] = pd.cut(df.lat, self.latbins,
                               labels=self.latbins[1:])
        df['lon_cut'] = pd.cut(df.lon, self.lonbins,
                               labels=self.lonbins[1:])
        return df

    def set_limits(self, latmin, latmax, lonmin, lonmax):
        """
        Set object's lat/lon limits

        Args:
            latmin, latmax, lonmin, lonmax : latitude/longitude
                min/max limits, floats
        Returns:
            None
        """
        self.latmin = latmin
        self.latmax = latmax
        self.lonmin = lonmin
        self.lonmax = lonmax

    def set_bin_resolution(self, npoints):
        """
        Set object's lat/lon bin size within the object's lat/lon limits

        Args:
            npoints : number of points between lat/lon min/max, int
        Returns:
            None
        """
        self.latbins = np.linspace(latmin, latmax, npoints)
        self.lonbins = np.linspace(lonmin, lonmax, npoints)

    def plot(self, featurelist):
        """
        Plots selected features

        Args:
            featurelist : list of names of features to plot.
                available names in self.allfeatures, list of strings
        Returns:
            None
        """
        nplots = len(featurelist)
        plt.figure(figsize=(16, 16*nplots))
        for i in xrange(1, nplots + 1):
            plt.subplot(nplots, 1, i)
            plt.scatter(self.fsmooth.lon, self.fsmooth.lat,
                        c=self.fsmooth[featurelist[i-1]], linewidths=0)
            plt.colorbar()
            plt.axis('equal')
            plt.margins(0)
            ax = plt.gca()
            ax.set_axis_bgcolor('black')
            ax.get_xaxis().get_major_formatter().set_useOffset(False)
            plt.title(featurelist[i-1])
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')

    def add_features(self, flist, verbose=False):
        """
        Load and featurize selected features

        Args:
            featurelist : list of names of features to plot.
                available names in self.allfeatures, list of strings
            verbose : whether to print status to console, boolean
        Returns:
            None
        """

        for f in flist:
            if verbose:
                print 'loading ', f
                sys.stdout.flush()

            # load database table
            df1 = get_db(f)

            # merge in lat/lon for census data from shapefile
            if f in ['usc_age_gender', 'usc_household', 'usc_pop']:
                df1 = df1.merge(self.shapefile, left_on='id2',
                                right_on='geoid')
            df1 = self.window(df1)

            # handling each table type
            if f == 'assessment':
                df1 = self.binlatlon(df1)
                df1 = df1[['lat_cut', 'lon_cut', 'taxable_value']]
                df1 = df1.groupby(['lat_cut', 'lon_cut']).mean()
                df1 = df1.reset_index().dropna()
                df1.columns = ['lat', 'lon', 'taxable_value']

            elif f == 'business':
                df1 = self.binlatlon(df1)
                df1 = df1[['lat_cut', 'lon_cut', 'category']]
                df1['count'] = 1
                df1 = df1.groupby(['lat_cut', 'lon_cut', 'category']).count()
                df1 = df1.reset_index().dropna()
                df2 = df1.pivot(columns='category',
                                values='count').fillna(0)
                df1 = df1.merge(df2, left_index=True, right_index=True)
                df1.drop('category', axis=1, inplace=True)
                df1 = df1.groupby(['lat_cut', 'lon_cut']).sum().reset_index()
                df1 = df1[['lat_cut', 'lon_cut', 'grocery',
                           'restaurant', 'retail']]
                df1.columns = ['lat', 'lon', 'grocery',
                               'restaurant', 'retail']

            elif f == 'sfpd':
                df1 = self.binlatlon(df1)
                df1['ncrimes'] = 1
                df1 = df1.groupby(['lat_cut', 'lon_cut']).count()
                df1 = df1.dropna().reset_index()
                df1 = df1[['lat_cut', 'lon_cut', 'ncrimes']]
                df1.columns = ['lat', 'lon', 'ncrimes']

            elif f == 'usc_age_gender':
                df1['sgnf'] = (2 * df1.f / df1.total).fillna(0) - 1
                df1 = df1[['lat', 'lon', 'sgnf']]

            elif f == 'usc_household':

                # calc average household size
                total_p = 0
                p_range = range(1, 8)
                for i in p_range:
                    col = 'p' + str(i)
                    total_p += df1[col] * i
                av_p = total_p / df1.total
                df1['avg_hh_size'] = av_p
                df1.fillna(0, inplace=True)

                df1 = df1[['lat', 'lon', 'avg_hh_size']]

            elif f == 'usc_pop':
                df1 = df1[['lat', 'lon', 'total']]
                df1.columns = ['lat', 'lon', 'population']

            elif f == 'walkscore':
                df1 = df1[['lat', 'lon', 'walkscore']]

            # append results to final data frame
            for col in df1.columns[2:]:
                finterp = bin_interpolate(df1.lon, df1.lat, df1[col],
                                          self.nodelon, self.nodelat)
                finterp = pd.Series(finterp, name=col)

                if self.features.shape == (0, 0):
                    self.features = pd.concat((self.nodelat, self.nodelon,
                                               finterp), axis=1)
                else:
                    self.features = pd.concat((self.features, finterp),
                                              axis=1)

    def smooth_features(self):
        """
        Smooth features according to self.smoothing levels via Rbf
            and scale to zero mean and unit standard deviation.

        Args:
            None
        Returns:
            None
        """
        self.fsmooth = self.features.copy()
        cols = self.features.drop(['lat', 'lon'], axis=1).columns

        if 'sgnf' in cols:
            # scale gender ratio by population
            if 'population' not in cols:
                self.add_features(['usc_pop'])

        if 'taxable_value' in cols:
            # log transform taxable value
            self.fsmooth.taxable_value[self.fsmooth.taxable_value < 0] = 0
            self.fsmooth.taxable_value = np.log(self.fsmooth.taxable_value+1)

        for col in cols:
            rbf = Rbf(self.features.lon, self.features.lat,
                      self.fsmooth[col], function='linear',
                      smooth=self.smoothing[col])
            self.fsmooth[col] = rbf(self.features.lon, self.features.lat)

        # scale to zero mean, unit standard deviation
        ssc = StandardScaler()
        self.fsmooth.iloc[:, 2:] = ssc.fit_transform(self.fsmooth.iloc[:, 2:])

    def make_edges(self):
        """
        Generate list of nodes connected to each node for graph analysis

        Args:
            None
        Returns:
            None
        """

        def edgelambda():
            return find_closest(x, df)

        self.edges = self.features.apply(edgelambda, axis=1)


if __name__ == '__main__':

    f = featurizer()

    print 'Making features'
    f.add_features(f.alltables, verbose=True)

    print 'Smoothing features'
    f.smooth_features()

    print 'Making edges'
    f.make_edges()

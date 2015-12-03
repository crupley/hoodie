# Codebook

Organization of code files and requirements.

## Files and contents

Filename | Function
:--|:--
makedbs.py | Functions associated with extracting raw data, cleaning, transforming, and inserting into and extracting from the database.
featurize.py | Functions to turn data in database into actionable features.
shapefiles.py | Collection of functions for loading, manipulating, and merging shapefiles.
graphreduce.py | Collection of functions for building and analyzing a graph model.
clusterize.py | Final function set for creating neighborhood clusters.



## External dependencies

The following python packages must be installed for everything to function.

* pandas - dataframe functionality
* numpy - mathematical operations and array manipulation
* matplotlib - for some plotting and colormaps
* scipy - for interpolation and smoothing (Rbf, Delaunay)
* sklearn - scaling and similarity (StandardScaler, pairwise_distances)
* psycopg2 - connecting to postgres database
* sqlalchemy - bridge between pandas and postgres database
* [pyshp](https://pypi.python.org/pypi/pyshp) - for loading data from census shapefiles
* [shapely](https://pypi.python.org/pypi/Shapely) - for manipulating and joining shapefiles (polygon, mapping, cascaded_union, polygonize)
* [graph-tool](https://graph-tool.skewed.de/) - a fast graph network analysis package
* [networkx](https://networkx.github.io/) - additional tools for working with graph networks
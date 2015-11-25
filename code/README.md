# Codebook

## Files and contents

Filename | Function
:--|:--
makedbs.py | Functions associated with extracting raw data, cleaning, transforming, and inserting into and extracting from the database.
featurize.py | Functions to turn data in database into actionable features.
shapefiles.py | Collection of functions for loading, manipulating, and merging shapefiles.
graphreduce.py | Collection of functions for building and analyzing a graph model.
clusterize.py | Final function set for creating neighborhood clusters.



## External dependencies

* pandas
* numpy
* matplotlib
* scipy: Rbf, Delaunay
* sklearn: StandardScaler, pairwise_distances
* psycopg2
* sqlalchemy
* pyshp
* shapely: polygon, mapping, cascaded_union, polygonize
* graph-tool
* networkx
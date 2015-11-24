# Codebook

## Files and contents

Filename | Function
:--|:--
makedbs.py | Functions associated with extracting raw data, cleaning, transforming, and inserting into and extracting from the database.
featurize.py | Functions to turn data in database into actionable features
shapefiles.py | Collection of functions for loading, manipulating, and merging shapefiles.


## Dependencies

* pandas
* numpy
* matplotlib
* scipy: Rbf, Delaunay
* sklearn: StandardScaler
* psycopg2
* sqlalchemy
* pyshp
* requests
* shapely: polygon, mapping, cascaded_union, polygonize
# hoodieSF

hoodieSF was created on the premise that neighborhood boundaries in a city are fluid and can vary based on what factors you care about. Are they defined by the demographics of the people living there, or is it the types of shops and restaurants? How much does cost of living matter? What about crime, or population density? Using a variety of location-specific data, hoodieSF lets you compare and contrast different areas in the city of San Francisco based on what features matter to you.

You can can use the live project at [https://crupley.github.io/hoodie/](https://crupley.github.io/hoodie/)

## What it does

hoodieSF is a tool for visualizing and discovering neighborhoods in San Francisco.  The San Francisco Planning Department officially identifies 36 neighborhoods, but those neighborhood boundaries are static, possibly outdated, and possibly irrevalent to what you would consider defines the character of a neighborhood.

You can choose any combination of up to three of the following features:

* Property Value - average property value
* Grocery Stores - density of grocery stores
* Restaurants - density of restaurants
* Retail Stores - density of retail stores
* Crime - density of reported crime events
* Gender - Female/male ratio
* Household Size - average household size
* Population - population density
* Walkscore - average

Once you select your features, click the 'Draw!' button and a new set of neighborhoods will be loaded.

![raw map](/images/web-app-load.png)

After you have a set of features you would like to explore, you can then begin to interact with the map. Clicking on a particular neighborhood with cause each neighborhood to be colored according to how similar it is to the one you clicked; green is most similar, red is least. A bar chart of the features you selected is also displayed for the selected neighborhood in the sidebar so you see what makes your neighborhoods similar.

![color map](/images/web-app-clicked.png)


## How it works

In order to determine the new neighborhood boundaries, I use a blend of two different clustering methods; kmeans clustering and edge betweenness from graph theory. Creation of the model is as follows,

* All of the data from different sources is aggregated, binned (via interpolation), and smoothed. The bins used are defined by the US Census-defined blocks of which there are about 7,300 in the city.

![Binned population](/images/datapopulation.png)

* A graph is then created by creating connections (edges) between adajacent data bins (nodes). The graph looks like this:

![graph](/images/plotedges.png)

* The edges in the graph are then weighted according to how similar the connected nodes are. The weight is calculated as the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) between the feature vectors of the two nodes. Since the subset of features depends on the user's input, a different graph is constructed for each possible combination. There are 9 possible features and choosing 1, 2, or 3 features leads to 129 combinations. A final unweighted graph is thrown in to make it an even 130.

* This weighted graph is then reduced, borrowing the [Girvan-Newman algorithm](https://en.wikipedia.org/wiki/Girvan%E2%80%93Newman_algorithm) from graph theory. In the algorithm, edges are removed sequentially according to how 'connected' they are until a group of nodes is completely separated from the rest; a new neighborhood is born! When the similarity weighting is added, the edges now need to have a combination of high connectivity and low similarity to be removed. In this way, a border is drawn between dissimilar chunks of the city.

* Finally, in order to determine the optimal number of neighborhoods, each time a neighborhood is split, I examine how similar the blocks in that neighborhood are to the neighborhood average using a metric called Within Cluster Sum of Squares Error (WCSSE). As the clusters get smaller, the blocks in them will be more similar to the cluster mean. The optimal value is chosen where the rate of decrease of the WCSSE starts to level off. After inspecting a subset of the feature sets, all of them appeared to have an optimal level between 20 to 30 neighborhoods. For consistency, I then fixed the number of neighborhoods at 25 for all combinations.

### Under the hood

All of the data analysis for this project was performed in [python](https://www.python.org/) (v2). The key python packages used include NumPy, pandas, and graph-tool. The visualization and web interface was built on the [Google Maps Javascript API](https://developers.google.com/maps/documentation/javascript/) and the data was plotted using [Google Charts API](https://developers.google.com/chart/).


## Data Sources

Data for the project came from 3 primary sources:

1. The [US Census](http://www.census.gov/) provided data for Population, Household Size, and Age/Gender (2010 census)
2. [SF OpenData](https://data.sfgov.org/) provided data for Crime, Registered Businesses, and Property Tax Assessments
3. [Walkscore.com](https://www.walkscore.com/) provided walkscore data; a measure of how walkable a location is.

More information, including links to all sources, can be found in the [databook](data/databook.md)

## Repo Structure

This repo is organized as follows

* code - python code used to produce these results
* data - file structure for data used. 
* images - image files for reference
* intermediate - storage location for intermediate results during calculation of the model
* results - final results of the model
* web - code and data required for web app

## Code

Description of code and package dependencies can be found in the [code readme](code/README.md)

## Reproduction of Results

If you are interested in reproducing the results here, you can do so by following these instructions. All steps should be executed from the root folder of this repo ('hoodie/')

1. Clone this repo

1. Download raw data from sources as outlined in the [databook](data/databook.md) and place in the appropriate locations.

2. Setup postgres database:

    * Create
    
        ```bash
        $ psql
        # CREATE DATABASE hoodie;
        ```
        
    * Make schema

        ```bash
        $ psql hoodie < data/assessment/assessment.sql
        $ psql hoodie < data/business/business.sql
        $ psql hoodie < data/sfpd/sfpd.sql
        $ psql hoodie < data/uscensus/uscensus.sql
        $ psql hoodie < data/walkscore/walkscore.sql
        ```

3. Install dependencies

    * Dependencies are outlined in the [code readme](code/README.md)

4. execute main.py:

    ```
    $ python code/main.py
    ```
    
There is a considerable amount of computation involved; especially for the graph-reduction portion. You may want to start with just a subset of the feature combinations.

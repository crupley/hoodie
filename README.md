# hoodieSF

hoodieSF was created on the premise that neighborhood boundaries in a city are fluid and can vary based on what factors you care about. Are they defined by the demographics of the people living there, or is it the types of shops and restaurants? How much does cost of living matter? What about crime, or population density? Using a variety of location-specific data, hoodieSF lets you compare and contrast different areas in the city of San Francisco based on what features matter to you.

You can can use the live project at [http://hoodiesf.com](http://hoodiesf.com)

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

<2 neighborhoods image>

After you have a set of features you would like to explore, you can then begin to interact with the map. Clicking on a particular neighborhood with cause each neighborhood to be colored according to how similar it is to the one you clicked; green is most similar, red is least. A bar chart of the features you selected is also displayed for the selected neighborhood in the sidebar so you see what makes your neighborhoods similar.

<2 similarity image>



## How it works

In order to determine the new neighborhood boundaries, I use a blend of two different clustering methods; kmeans clustering and edge betweenness from graph theory.





## Data Sources

Data for the project came from 3 primary sources:

1. The [US Census](http://www.census.gov/) provided data for Population, Household Size, and Age/Gender (2010 census)
2. [SF OpenData](https://data.sfgov.org/) provided data for Crime, Registered Businesses, and Property Tax Assessments
3. [Walkscore.com](https://www.walkscore.com/) provided walkscore data; a measure of how walkable a location is.

More information, including links to all sources, can be found in the [databook](data/databook.md)

## Repo Structure

## Code

## Reproduce

1. Download raw data from sources as outlined in the [databook](data/databook.md)

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
    $ python main.py
    ```
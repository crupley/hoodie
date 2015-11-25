# hoodieSF

hoodieSF was created on the premise that neighborhood boundaries in a city are fluid and can vary based on what factors you care about. Are they defined by the demographics of the people living there, or is it the types of shops and restaurants? How much does cost of living matter? What about crime, or population density? Using a variety of location-specific data, hoodieSF lets you compare and contrast different areas in the city of San Francisco based on what features matter to you.

## What it does

## How it works

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
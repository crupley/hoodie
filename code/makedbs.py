import numpy as np
import pandas as pd
import re

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

from pyshp import shapefile

"""
Functions associated with extracting raw data, cleaning, transforming,
and inserting into and extracting from the database.
"""


def db_insert(df, q_string):
	"""
    Insert pandas dataframe into the neighborhood database using
    accompanying query string.

    Args:
    	df : pandas dataframe formatted to match database table
    		columns should match columns in query string and table
    	q_string : string. sql query string for database insertion
    		The query string takes the following form:

    		'INSERT INTO <table name> (<column name1>, <column name2>, ...)
			VALUES (%s)'
    Returns:
    	None
    """

    # create a '%s' for each column to be inserted
	n_cols = df.shape[1]
	esses = '%s, ' * (n_cols - 1) + '%s'
	q_string = q_string % esses

	# connect to hoodie database
	conn = psycopg2.connect(dbname='hoodie', user='postgres', host='/tmp')
	c = conn.cursor()

	# insert data in dataframe one row at a time, write every 100 lines
	for idx in df.index:
		values = df.ix[idx].values
		c.execute(q_string, values)
		if idx % 100 == 0: conn.commit();
	conn.commit()
	conn.close()

def get_db(table_name):
	"""
    Fetch data from database as a pandas dataframe

    Args:
    	table_name : string. sql table name to fetch from database

    Returns:
    	pandas dataframe from sql table
    """

    # parameters
	db = {'drivername': 'postgres',
		  'host': 'localhost',
		  'username': 'postgres',
		  'database': 'hoodie'}

	engine = create_engine(URL(**db))
	with engine.connect() as conn, conn.begin():
		df = pd.read_sql_table(table_name, conn)
	return df

"""
Functions for handling each individual dataset. Most have a function
to load the data, clean it, and then insert it into the database.
"""

### assessment data
def make_assessment():
	"""
    Build 'assessment' data database

    Args:
    	None

    Returns:
    	None
    """
    fn = 'data/assessment/Secured_Property_Assessment_Roll_FY13_Q4.csv'
	df = pd.read_csv(fn)

	q_string = '''
		INSERT INTO assessment_raw (Situs_Address,
									Situs_Zip,
									APN,
									RE,
									RE_Improvements,
									Fixtures_Value,
									PP_Value,
									District,
									Taxable_Value,
									geom)
		VALUES (%s)'''

	# insert raw data
	db_insert(df, q_string)

	# clean data; remove unused columns, convert datatype
	df.drop('Fixtures_Value', axis = 1, inplace = True)
	df.RE = df.RE.apply(lambda x: x.strip('$')).astype('float')
	df.RE_Improvements = df.RE_Improvements.apply(lambda x: x.strip('$'))
	df.RE_Improvements = df.RE_Improvements.astype('float')
	df.PP_Value = df.PP_Value.apply(lambda x: x.strip('$')).astype('float')
	df.Taxable_Value = df.Taxable_Value.apply(lambda x: x.strip('$'))
	df.Taxable_Value = df.Taxable_Value.astype('float')

	df['lat'] = df.geom.apply(lambda x: eval(x)[0])
	df['lon'] = df.geom.apply(lambda x: eval(x)[1])
	df.drop('geom', axis = 1, inplace = True)

	# insert cleaned data into db
	q_string = '''
		INSERT INTO assessment (Situs_Address,
								Situs_Zip,
								APN,
								RE,
								RE_Improvements,
								PP_Value,
								District,
								Taxable_Value,
								lat,
								lon)
		VALUES (%s)'''

	db_insert(df, q_string)

	return

### Business data
def getlatlon(v):
	"""
	Helper function for clean_business. Extracts latitude, longitude
	tuple from address cell.

    Args:
    	v : formatted address, string of form '(<lat>, <lon>)'

    Returns:
    	latitude, longitude, float tuple			
	"""
	
	# check for nan
	if type(v) == float: return 0, 0
	if v == 'NaN': return 0, 0
	s = v.split('\n')[-1]

	# check for missing lat/lon
	if len(s) == 0: return 0, 0

	# convert string to tuple
	return eval(s)

def load_business():
	"""
	Load the business dataset and return it in a dataframe

    Args:
		None
    Returns:
    	'business' data, pandas dataframe
	"""
	fn = 'data/business/Registered_Business_Locations_-_San_Francisco.csv'
	df = pd.read_csv(fn)
	return df

def make_business():
	'''
	INPUT: None
	OUTPUT: None
	Inserts data from the business dataset into project database
	both in raw form and cleaned form
	'''

	# insert raw data
	df = load_business()
	q_string = '''
		INSERT INTO business_raw (Location_ID,
								  Business_Account_Number,
								  Ownership_Name,
								  DBA_Name,
								  Street_Address,
								  City,
								  State,
								  Zip_Code,
								  Business_Start_Date,
								  Business_End_Date,
								  Location_Start_Date,
								  Location_End_Date,
								  Mail_Address,
								  Mail_City_State_Zip,
								  Class_Code,
								  PBC_Code,
								  Business_Location)
		VALUES (%s)'''

	# insert raw data
	# db_insert(df, q_string)

	### clean data

	df = get_db('business_raw')
	df = clean_business(df)

	q_string = '''
	INSERT INTO business (ownership_name,
						  dba_name,
						  street_address,
						  city,
						  state,
						  zip_code,
						  class_code,
						  pbc_code,
						  lat,
						  lon,
						  major_class,
						  minor_class,
						  category)
	VALUES (%s)'''

	# insert raw data
	db_insert(df, q_string)
	return

def clean_business(df):
	### clean data

	df.columns = map(lambda x: x.lower().replace(' ', '_'), df.columns)

	# drop rows where location has an end date
	df = df[df.location_end_date == 'NaN']

	df['lat'] = df.business_location.apply(lambda x: getlatlon(x)[0])
	df['lon'] = df.business_location.apply(lambda x: getlatlon(x)[1])

	major_names = {'00': 'Fixed Place of Business',
				   '01': 'Commission Merchant or Broker (non-durable goods)',
				   '02': 'General Contractors & Operative Builders',
				   '03': 'Hotels, Apartments',
				   '04': 'Coin-Operated Laundries, Dry Cleaning',
				   '05': 'Credit Agencies, Lending Institutions',
				   '06': 'Personal Property/Equipment Rental & Leasing',
				   '07': 'Other Business Services',
				   '08': 'Retail Sales',
				   '09': 'Public Warehousing/Transportation/Storage/Freight Forwarding',
				   '10': 'Communication Services; Utilities (Gas/Electric/Steam/Railroad)',
				   '11': 'Transporting Persons for Hire',
				   '12': 'Trucking/Hauling',
				   '13': 'Wholesale Sales',
				   '15': 'Architectural and Engingeering Services',
				   '16': 'Non-Profit Garage Corporations',
	               'n.a.': 'n.a.',
	               'NaN': 'n.a.'}
	df['major_class'] = df['class_code'].replace(major_names)

	df['pbc_code'] = df['pbc_code'].replace(['n.a.', 'NaN'], '8888')
	df['pbc_code'] = df['pbc_code'].astype('int')




	# drop rows with missing lat/lon
	df = df[df.lat != 0] # ~5000

	# add pbc code descriptions
	codes = pd.read_csv('data/business/Principal_Business_Code__PBC__List.csv')
	codes.set_index('Business_Minor_Class', inplace = True)
	df['minor_class'] = df['pbc_code'].replace(codes.Description)
	# df = df.merge(codes, how='left', left_on='pbc_code', right_on='Business_Minor_Class')

	# add category column
	df['category'] = '0'
	df.category.loc[df.class_code == '04'] = 'laundry'
	for cl in ['n.a.', '00', '01', '02', '03', '05', '06', '09',
			   '10', '11', '12', '13', '15', '16']:
		df.category.loc[df.class_code == cl] = 'other'

	keywords = [('BUSINESS SERVICES', 'other'),
				('DRINKING PLACES', 'bars'),
				('RESTAURANT', 'restaurant'),
				('GROCERY', 'grocery'),
				('FOOD', 'grocery'),
				('MARKET', 'grocery'),
				('MEDICAL', 'medical'),
				('OFFICE', 'medical'),
				('HOSPITAL', 'medical'),
				('HEALTH', 'medical'),
				('RETAIL', 'retail'),
				('APPAREL', 'retail'),
				('GIFT', 'retail'),
				('JEWELRY', 'retail'),
				('THEATER', 'entertainment'),
				('BARBER', 'personal_care'),
				('BEAUTY', 'personal_care'),
				('HOTEL', 'hotel'),
				('PARKING', 'parking')]

	for key, value in keywords:
		df.category[df.minor_class.str.contains(key, na=False)] = value
	df.category[df.category == '0'] = 'other'




	# drop unused columns
	df.drop(['location_id',
         'business_account_number',
         'mail_address',
         'mail_city_state_zip',
         'business_start_date',
         'business_end_date',
         'location_start_date',
         'location_end_date',
         'business_location'],
         axis = 1, inplace=True)

	return df

def load_sfpd():
	df = pd.read_csv('../data/sfpd/SFPD_Incidents_-_from_1_January_2003.csv')
	return df

def clean_sfpd(df):

	df.rename(columns={'X': 'lon', 'Y': 'lat'}, inplace=True)

	df['datetime'] = df.Date + ' ' + df.Time
	df.drop(['IncidntNum',
			 'DayOfWeek',
			 'Resolution',
			 'Location',
			 'PdId',
			 'Date',
			 'Time'], axis = 1, inplace = True)


	return df

def make_sfpd():
	'''
	INPUT: None
	OUTPUT: None
	Inserts data from the business dataset into project database
	both in raw form and cleaned form
	'''

	# insert raw data
	df = load_sfpd()
	q_string = '''
		INSERT INTO sfpd_raw (IncidntNum,
							  Category,
							  Descript,
							  DayOfWeek,
							  Date,
							  Time,
							  PdDistrict,
							  Resolution,
							  Address,
							  X,
							  Y,
							  Location,
							  PdId)
		VALUES (%s)'''

	db_insert(df, q_string)

	# insert clean data
	df = clean_sfpd(df)
	q_string = '''
		INSERT INTO sfpd (Category,
						  Descript,
						  PdDistrict,
						  Address,
						  lon,
						  lat,
						  datetime)
		VALUES (%s)'''

	# insert raw data
	db_insert(df, q_string)

### US Census Data

def splitgeo(geo):
    s = geo.split(',')
    block = s[0].split(' ')[-1]
    blockgroup = s[1].split(' ')[-1]
    tract = s[2].split(' ')[-1]
    return block, blockgroup, tract

def load_usc_age_gender():
	df = pd.read_csv('data/uscensus/p12/DEC_10_SF1_P12.csv', skiprows=1)
	return df

def clean_usc_age_gender(df):
	df['geovalues'] = df.Geography.apply(splitgeo)
	cols = ['Block', 'Block_Group', 'Tract']
	for n, col in enumerate(cols[::-1]):
		df.insert(0, col, df.geovalues.apply(lambda x: x[n]))


	df.drop(['Id',
			 'Geography',
			 'geovalues'], axis = 1, inplace = True)

	cnames = df.columns
	cnames = cnames.str.replace(': - ', '_')
	cnames = cnames.str.replace('Male', 'M')
	cnames = cnames.str.replace('Female', 'F')
	cnames = cnames.str.replace(' to ', '_')
	cnames = cnames.str.replace(' years', '')
	cnames = cnames.str.replace(' and ', '_')
	cnames = cnames.str.replace(':', '')
	cnames = cnames.str.replace('-over', '+')
	cnames = cnames.str.replace('Under ', 'U')
	df.columns = cnames

	return df

def make_usc_age_gender():
	df = load_usc_age_gender()
	df = clean_usc_age_gender(df)
	q_string = '''
		INSERT INTO usc_age_gender (Block,
									Block_Group,
									Tract,
									Id2,
									Total,
									M,
									M_U5,
									M_5_9,
									M_10_14,
									M_15_17,
									M_18_19,
									M_20,
									M_21,
									M_22_24,
									M_25_29,
									M_30_34,
									M_35_39,
									M_40_44,
									M_45_49,
									M_50_54,
									M_55_59,
									M_60_61,
									M_62_64,
									M_65_66,
									M_67_69,
									M_70_74,
									M_75_79,
									M_80_84,
									M_85_over,
									F,
									F_U5,
									F_5_9,
									F_10_14,
									F_15_17,
									F_18_19,
									F_20,
									F_21,
									F_22_24,
									F_25_29,
									F_30_34,
									F_35_39,
									F_40_44,
									F_45_49,
									F_50_54,
									F_55_59,
									F_60_61,
									F_62_64,
									F_65_66,
									F_67_69,
									F_70_74,
									F_75_79,
									F_80_84,
									F_85_over)
		VALUES (%s)'''

	db_insert(df, q_string)

def load_usc_household():
	df = pd.read_csv('data/uscensus/h13/DEC_10_SF1_H13.csv', skiprows=1)
	return df

def clean_usc_household(df):
	df['geovalues'] = df.Geography.apply(splitgeo)
	cols = ['Block', 'Block_Group', 'Tract']
	for n, col in enumerate(cols[::-1]):
		df.insert(0, col, df.geovalues.apply(lambda x: x[n]))


	df.drop(['Id',
			 'Geography',
			 'geovalues'], axis = 1, inplace = True)

	cnames = df.columns
	cnames = cnames.str.replace('-person household', '')
	cnames = cnames.str.replace('-or-more', '')
	cnames = cnames.str.replace(':', '')

	# prepend numeric entries
	fnames = []
	for name in cnames:
	    pp = ''
	    if re.match('[0-9]', name): pp = 'p'
	    fnames.append(pp + name)
	df.columns = fnames

	return df

def make_usc_household():
	df = load_usc_household()
	df = clean_usc_household(df)
	q_string = '''
		INSERT INTO usc_household (Block,
								   Block_Group,
								   Tract,
								   Id2,
								   Total,
								   p1,
								   p2,
								   p3,
								   p4,
								   p5,
								   p6,
								   p7)
		VALUES (%s)'''

	db_insert(df, q_string)

def load_usc_pop():
	df = pd.read_csv('data/uscensus/p1/DEC_10_SF1_P1.csv', skiprows=1)
	return df

def clean_usc_pop(df):
	df['geovalues'] = df.Geography.apply(splitgeo)
	cols = ['Block', 'Block_Group', 'Tract']
	for n, col in enumerate(cols[::-1]):
		df.insert(0, col, df.geovalues.apply(lambda x: x[n]))


	df.drop(['Id',
			 'Geography',
			 'geovalues'], axis = 1, inplace = True)

	return df

def make_usc_pop():
	df = load_usc_pop()
	df = clean_usc_pop(df)
	q_string = '''
		INSERT INTO usc_pop (Block,
							 Block_Group,
							 Tract,
							 Id2,
							 Total)
		VALUES (%s)'''

	db_insert(df, q_string)


def load_usc_shapefile():
	fn = 'data/uscensus/tl_2010_06075_tabblock10/tl_2010_06075_tabblock10.dbf'
	rdf = sf_to_df(fn)
	return rdf


def make_usc_shapefile():
	df = load_usc_shapefile()
	q_string = '''
		INSERT INTO usc_shapefile (state,
								   county,
								   tract,
								   block,
								   geoid,
								   name,
								   mtfcc,
								   land_area,
								   water_area,
								   lat,
								   lon)
		VALUES (%s)'''

	db_insert(df, q_string)


# Walkscore
# imported in scrape.py

def clean_walkscore(df):

	df['walkscore'] = df.walkscore.astype('int')

	# remove duplicated points
	df.drop_duplicates(['snapped_lat', 'snapped_lon'], inplace = True)

	df.drop(['help_link',
			 'logo_url',
			 'more_info_icon',
			 'more_info_link',
			 'status',
			 'ws_link'], axis = 1, inplace = True)

	df = df[['snapped_lat', 'snapped_lon', 'walkscore', 'description',
         	 'updated', 'searched_lat', 'searched_lon']]
	df.columns = ['lat', 'lon', 'walkscore', 'description',
         	 	  'updated', 'searched_lat', 'searched_lon']
	
	return df


def make_walkscore():
	df = get_db('walkscore_raw')
	df = clean_walkscore(df)
	q_string = '''
		INSERT INTO walkscore (lat,
							   lon,
							   walkscore,
							   description,
							   updated,
							   searched_lat,
							   searched_lon)
		VALUES (%s)'''

	db_insert(df, q_string)


def sf_to_df(filename):
	'''
	INPUT
		filename: string
	OUTPUT
		shapefile records; pandas DataFrame

	Converts shapefile records from US Census 2010 to dataframe.
	'''
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


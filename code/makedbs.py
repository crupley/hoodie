# Ingest data from sources into standardized format and store in db
import numpy as np
import pandas as pd

import psycopg2

import pdb


def db_insert(df, q_string):
	# insert raw data into db
	n_cols = df.shape[1]
	esses = '%s, ' * (n_cols - 1) + '%s'
	q_string = q_string % esses

	conn = psycopg2.connect(dbname='hoodie', user='postgres', host='/tmp')
	c = conn.cursor()

	for idx in df.index:
		values = df.ix[idx].values
		c.execute(q_string, values)
		if idx % 100 == 0: conn.commit();
	conn.commit()
	conn.close()

# assessment data
def make_assessment():
	df = pd.read_csv('../data/assessment/Secured_Property_Assessment_Roll_FY13_Q4.csv')

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

	# clean data
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

def getlatlon(v):
	'''
	INPUT: formatted address, string
	OUTPUT: latitude, longitude, tuple

	Helper function for clean_business. Extracts latitude, longitude
	tuple from address cell
	'''

	# check for nan
	if type(v) == float: return 0, 0
	s = v.split('\n')[-1]

	# check for missing lat/lon
	if len(s) == 0: return 0, 0

	# convert string to tuple
	return eval(s)

def load_business():
	'''
	INPUT: None
	OUTPUT: business data, DataFrame
	Function to load the business dataset and return it in a dataframe
	'''
	fn = '../data/business/Registered_Business_Locations_-_San_Francisco.csv'
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
	db_insert(df, q_string)

	### clean data

	df = clean_business(df)

	q_string = '''
	INSERT INTO business (Ownership_Name,
						  DBA_Name,
						  Street_Address,
						  City,
						  State,
						  Zip_Code,
						  major_class,
						  minor_class,
						  lat,
						  lon)
	VALUES (%s)'''

	# insert raw data
	db_insert(df, q_string)
	return

def clean_business(df):
	### clean data

	# drop rows where location has an end date
	df = df[pd.isnull(df.Location_End_Date)]

	major_names = {'00': 'fixed place of business',
	               '01': 'agent, broker',
	               '02': 'construction',
	               '03': 'hotel',
	               '04': 'laundry',
	               '05': 'credit',
	               '06': 'equipment rental',
	               '07': 'mining, entertainment, medical',
	               '08': 'clothing, furniture, auto, insurance',
	               '09': 'warehouse',
	               '10': 'utility',
	               '11': 'taxi',
	               '12': 'trucking',
	               '13': 'industrial supply',
	               '15': 'arcitectural',
	               '16': 'garage',
	               '18': 'firearms',
	               'n.a.': 'n.a.'}
	df['major_class'] = df['Class Code'].replace(major_names)
	df['minor_class'] = 0


	df['lat'] = df.Business_Location.apply(lambda x: getlatlon(x)[0])
	df['lon'] = df.Business_Location.apply(lambda x: getlatlon(x)[1])

	# drop rows with missing lat/lon
	df = df[df.lat != 0] # ~5000

	# drop unused rows
	df.drop(['Location_ID',
         'Business_Account_Number',
         'Mail_Address',
         'Mail_City_State_Zip',
         'Business_Start_Date',
         'Business_End_Date',
         'Location Start Date',
         'Location_End_Date',
         'Class Code',
         'PBC Code',
         'Business_Location'], axis = 1, inplace=True)

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


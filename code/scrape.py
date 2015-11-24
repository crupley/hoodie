import numpy as np
import pandas as pd

import psycopg2
import requests


def scrape_walkscore(lat, lon):
	'''
	INPUT
		lat, lon: latitude and longitude; float
	OUTPUT
		stores results in postgres database

	API query to walkscore.com for walkscore data nearest to a particular
	latitude and longitude. Walkscore snaps to a grid with 500 ft. spacing.
	'''
	with open('/Users/crupley/.api/walkscore.txt') as f:
	    wskey = f.readline().strip()

	requrl = 'http://api.walkscore.com/score'
	payload = {'wsapikey': wskey,
			   'lat': lat,
			   'lon': lon,
			   'format': 'json'}

	response = requests.get(requrl, params = payload)

	if response.json()['status'] != 1:
		print 'Server response error, code:', response.json()['status']
		return

	data = pd.Series(response.json())
	data['searched_lat'] = lat
	data['searched_lon'] = lon

	ncols = data.shape[0]
	q_string = 	'''
		INSERT INTO walkscore_raw (description,
								   help_link,
								   logo_url,
								   more_info_icon,
								   more_info_link,
								   snapped_lat,
								   snapped_lon,
								   status,
								   updated,
								   walkscore,
								   ws_link,
								   searched_lat,
								   searched_lon)
			VALUES (%s)''' % ('%s, ' * (ncols - 1) + '%s')

	conn = psycopg2.connect(dbname='hoodie', user='postgres', host='/tmp')
	c = conn.cursor()

	c.execute(q_string, data.values)

	conn.commit()
	conn.close()

	return
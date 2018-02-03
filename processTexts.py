import messengerScraper
import json

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
  import Features, EntitiesOptions, KeywordsOptions

watsonCreds = 'ibm-key.json'
minInterval = 10

def initWatson():
	with open(watsonCreds, 'r') as f:
		creds = json.load(f)
	return NaturalLanguageUnderstandingV1(
		username = creds['username'],
		password = creds['password'],
		version = '2017-02-27')

if __name__ == "__main__":
	natural_language_understanding = initWatson()
	# name, msgs = messengerScraper.scrapeAll('data');

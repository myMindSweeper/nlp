import messengerScraper
import datetime
import json

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
  import Features, EntitiesOptions, KeywordsOptions

watsonCreds = 'ibm-key.json'
clumpMins = 20
endPunctuation = set(['!', ',', '.', ';', '?'])

def initWatson():
	with open(watsonCreds, 'r') as f:
		creds = json.load(f)
	return NaturalLanguageUnderstandingV1(
		username = creds['username'],
		password = creds['password'],
		version = '2017-02-27')

def makeClumps(name, convos):
	allClumps = []
	userClumps = []
	for convo in convos:
		allMsgs = []
		userMsgs = []
		clumpEndTime = convo[0]['time'] + datetime.timedelta(minutes=clumpMins)
		for msg in convo:
			if msg['time'] > clumpEndTime:
				if len(allMsgs) > 0:
					allClumps.append(clumpMsgs(allMsgs))
					allMsgs = []
					if len(userMsgs) > 0:
						userClumps.append(clumpMsgs(userMsgs))
						userMsgs = []
				clumpEndTime = msg['time'] + datetime.timedelta(minutes=clumpMins)
			allMsgs.append(msg)
			if msg['user'] == name:
				userMsgs.append(msg)
		if len(allMsgs) > 0:
			allClumps.append(clumpMsgs(allMsgs))
			if len(userMsgs) > 0:
				userClumps.append(clumpMsgs(userMsgs))
	return allClumps, userClumps

def clumpMsgs(msgs):
	clump = {'time': msgs[0]['time'], 'user': msgs[0]['user'], 'text': ''}
	for msg in msgs:
		clump['text'] += msg['text']
		if msg['text'][-1:] not in endPunctuation:
			clump['text'] += '. '
		else:
			clump['text'] += ' '
	return clump

if __name__ == "__main__":
	name, convos = messengerScraper.scrapeAll('data')
	allClump, userClump = makeClumps(name, convos)
	for clump in userClump:
		print(clump)

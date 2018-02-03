import messengerScraper
import datetime
import json, csv
import re

from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 \
	import Features, EmotionOptions, SentimentOptions, KeywordsOptions

watsonCreds = 'ibm-key.json'
clumpMins = 20
endPunctuation = set(['!', ',', '.', ';', '?'])
regexAbb = {
	'aka': 'also known as',
	'btw': 'by the way',
	'bc': 'because',
	'fyi': 'for your information',
	'idk':"I don't know",
	'imo': 'in my opinion',
	'omg': 'oh my gosh',
	'omfg': 'oh my gosh',
	'tba': 'to be announced',
	'tbd': 'to be decided',
	'thx':'thanks',
	'wtf':'what the heck'}

# get instance of Watson NaturalLanguageUnderstanding
def initWatson():
	with open(watsonCreds, 'r') as f:
		creds = json.load(f)
	return NaturalLanguageUnderstandingV1(
		username = creds['username'],
		password = creds['password'],
		version = '2017-02-27')

# combine conversations into larger clumps for watson processing
def makeClumps(name, convos):
	clumped = []
	for convo in convos:
		allMsgs = []
		userMsgs = []
		clumpEndTime = convo[0]['time'] + datetime.timedelta(minutes=clumpMins)
		for msg in convo:
			if msg['time'] > clumpEndTime:
				if len(userMsgs) > 0:
					clumped.append((clumpMsgs(name, allMsgs), clumpMsgs(name, userMsgs)))
				allMsgs = []
				userMsgs = []
				clumpEndTime = msg['time'] + datetime.timedelta(minutes=clumpMins)
			allMsgs.append(msg)
			if msg['user'] == name:
				userMsgs.append(msg)
		if len(userMsgs) > 0:
			clumped.append((clumpMsgs(name, allMsgs), clumpMsgs(name, userMsgs)))
	return clumped

# concatenate msgs together into one with correct user attribution
def clumpMsgs(name, msgs):
	foundUser = False
	clump = {'time': msgs[0]['time'], 'user': name, 'text': ''}
	for msg in msgs:
		clump['text'] += msg['text']
		if msg['text'][-1:] not in endPunctuation:
			clump['text'] += '. '
		else:
			clump['text'] += ' '
		if not foundUser and msg['user'] != name:
			clump['user'] = msg['user']
			foundUser = True
	return clump

# returns information for stats package given clumps to analyze with Watson
def analyzeClumps(clumps):
	natural_language_understanding = initWatson()
	data = []
	for allClump, userClump in clumps:
		userResp = natural_language_understanding.analyze(
			text = userClump['text'],
			features = Features(
				sentiment = SentimentOptions(), 
				emotion = EmotionOptions()),
			language='en')
		allResp = natural_language_understanding.analyze(
			text = allClump['text'],
			features = Features(
				keywords = KeywordsOptions()),
			language = 'en')
		score = riskScore(userClump, userResp)
		keywords = relevantKeywords(allResp)
		data.append({
			'time': allClump['time'], 
			'user': allClump['user'], 
			'score': score, 
			'keywords': keywords})
	return data

# calculates a risk score for a message
def riskScore(clump, response):
	emotions = response['emotion']['document']['emotion']
	sentiment = response['sentiment']['document']
	if sentiment['label'] == 'positive':
		return 3.5 + emotions['anger'] - emotions['sadness'] + 1.5 * emotions['joy']
	elif sentiment['label'] == 'negative':
		return 5.0 - 4.0 * (emotions['anger'] + emotions['sadness'])
	else:
		return 4.0
	return emotions['sadness'] + emotions['fear'] + emotions['anger'] - emotions['joy']

# extracts keywords and their relevance from a keyword response
def relevantKeywords(response):
	return [{'term': term['text'], 'relevance': term['relevance']} \
			for term in response['keywords']]

# preprocess conversation to normalize text
def preprocess(convo):
	return {
		'time': convo['time'],
		'user': convo['user'],
		'text': re.sub(
			r'\S+', 
			lambda g: regexAbb[g.group(0).lower()] if g.group(0) in regexAbb else g.group(0), 
			convo['text'])
	}

# calculates risk scores and writes them to csv file along with associated meta-data
def writeDataToFile(name, convos, ordered, file):
	if not ordered:
		convos.sort(key = lambda dic: dic['time'])
	convos = [[preprocess(msg) for msg in convo] for convo in convos]
	data = analyzeClumps(makeClumps(name, convos))
	with open(file, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(data)):
			curr = data[i]
			writer.writerow([i // 10, curr['time'], curr['score'], curr['user'], curr['keywords']])

if __name__ == "__main__":
	name, convos = messengerScraper.scrapeAll('data')
	writeDataToFile(name, convos, True, 'data/data.csv')

import messengerScraper
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
	'wtf':'what the heck',
	'lol': 'haha',
	'ppl': 'people',
	'brb': 'be right back' }

# get stopwords from file
def loadStopWords(file):
	stopwords = set()
	with open(file) as f:
		for line in f:
			stopwords.add(line.lower().rstrip())
	return stopwords

# get instance of Watson NaturalLanguageUnderstanding
def initWatson():
	with open(watsonCreds, 'r') as f:
		creds = json.load(f)
	return NaturalLanguageUnderstandingV1(
		username = creds['username'],
		password = creds['password'],
		version = '2017-02-27')

# combine conversations into larger clumps for watson processing
def makeClumps(convos):
	clumped = []
	for convo in convos:
		allMsgs = []
		userMsgs = []
		clumpEndTime = convo['msg_list'][0]['date'] + clumpMins * 60
		for msg in convo['msg_list']:
			if msg['date'] > clumpEndTime:
				if len(userMsgs) > 0:
					clumped.append((
						concatToClump(convo['person'], allMsgs), 
						concatToClump(convo['person'], userMsgs)))
				allMsgs = []
				userMsgs = []
				clumpEndTime = msg['date'] + clumpMins * 60
			allMsgs.append(msg)
			if msg['user_speaking']:
				userMsgs.append(msg)
		if len(userMsgs) > 0:
			clumped.append((
				concatToClump(convo['person'], allMsgs), 
				concatToClump(convo['person'], userMsgs)))
	return clumped

# concatenate msgs together into one with correct user attribution
def concatToClump(user, msgs):
	clump = {'time': msgs[0]['date'], 'user': user, 'text': ''}
	for msg in msgs:
		clump['text'] += msg['body']
		if msg['body'][-1:] not in endPunctuation:
			clump['text'] += '. '
		else:
			clump['text'] += ' '
	return clump

# returns information for stats package given clumps to analyze with Watson
def analyzeClumps(clumps, stopWords):
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
		keywords = getKeywords(allResp, stopWords)
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

# extracts relevant keywords from the text
def getKeywords(response, stopWords):
	words = [{'term': term['text'], 'relevance': term['relevance']} \
			for term in response['keywords']]
	keywords = []
	for word in words:
		text = word['term'].lower()
		if text not in stopWords:
			keywords.append({'term': text, 'relevance': word['relevance']})
	return keywords

# preprocess conversation to normalize text
def preprocess(convo):
	for i in range(len(convo['msg_list'])):
		target = convo['msg_list'][i]
		target['body'] = re.sub(r'\S+', 
				lambda g: regexAbb[g.group(0).lower()] if g.group(0) in regexAbb else g.group(0), 
				target['body'])
	return convo

# calculates risk scores and writes them to csv file along with associated meta-data
def writeDataToFile(convos, file):
	stopWords = loadStopWords('stopwords.txt')
	loads = [preprocess(load) for load in json.loads(convos)]
	data = sorted(analyzeClumps(makeClumps(loads), stopWords), key = lambda x: x['time'])
	with open(file, 'w') as f:
		writer = csv.writer(f)
		for i in range(len(data)):
			curr = data[i]
			writer.writerow([i // 10, curr['time'], curr['score'], curr['user'], curr['keywords']])

if __name__ == "__main__":
	convos = messengerScraper.scrapeAll('data')
	writeDataToFile(convos, 'data.csv')
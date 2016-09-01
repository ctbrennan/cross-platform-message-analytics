import sys
#sys.path.insert(0, '/home/connor/Documents/prj/parser/FB-Message-Parser/')
import fb_parser
import imessage_export
from collections import OrderedDict
import json
from datetime import datetime
from wordcloud import WordCloud
import calendar
import os, os.path #parseAllThreads
import nltk
import xml.etree.ElementTree as ET #parseSMS
import itertools
from titlecase import titlecase #for name/alias matching
from sklearn.metrics.pairwise import cosine_similarity #finding similarity between two texts, requires scipy
from sklearn.feature_extraction.text import TfidfVectorizer #make tf-idf matrix

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
import numpy as np
import re #remove emojis
import twython
#=====================================================================================================
#                                  Parsing Messages and Contact Info
#=====================================================================================================
me = None
personDict = {} #name -> dateTime -> tuple of messages
fullTextDict = {} #dateTime -> tuple of everyone's messages
vCardDict = {} #phone number/email -> name
aliasDict = {} #alias1 -> alias2, alias2 -> alias1
STOPWORDS = set([x.strip() for x in open(os.path.join(os.getcwd(),'stopwords.txt')).read().split('\n')])
STOPWORDS.add('none')
"""
For parsing all Facebook messages into dictionaries
"""
def parseFBMessages(confident):
	if not os.path.isfile('messages.json') :
		with open('messages.htm', "r") as f:
			chat = fb_parser.html_to_py(f)
			# Dump to json to prove works:
			fb_parser.py_to_json(chat)
	jsonFile = json.loads(open('messages.json').read())
	newPersonDict = {}
	newFullTextDict = {}
	for person in jsonFile['threads']:
		for message in person['messages']:
			sender = titlecase(message['sender'])
			text = message['text']
			dateFormatted = datetime.strptime(message['date_time'], '%A, %B %d, %Y at %I:%M%p ')
			addToNewDict(newPersonDict, dateFormatted, text, sender)
			addToNewDict(newFullTextDict, dateFormatted, text)
	if 'y' in input("Enter 'y' if you would like to match duplicate names on Facebook"):
		matchDuplicates(newPersonDict)
	mergeAndSortPersonDict(newPersonDict, confident)
	mergeAndSortFullTextDict(newFullTextDict)

"""
Parsing .xml file containing all sms ("Super Backup" for Android)
"""
def parseSMS(me, confident):
	def parseSuperBackup():
		tree = ET.parse('allsms.xml') #fix this later
		root = tree.getroot()
		newPersonDict = {}
		newFullTextDict = {}

		newNames = []
		notFound = []
		for message in root:
			phoneNumber = formatPhoneNumber(message.attrib['address'])
			if message.attrib['type'] == '2':
				sender = me
			elif message.attrib['name']:
				sender = titlecase(message.attrib['name'])
			elif phoneNumber in vCardDict.keys():
				sender = titlecase(vCardDict[phoneNumber])
				if sender not in newNames:
					newNames.append(sender)
			else:
				continue #don't add plain phone numbers
				
			date = message.attrib['time']
			text = message.attrib['body']
			dateFormatted = datetime.strptime(date, '%b %d, %Y %I:%M:%S %p') #"Jul 10, 2016 8:28:10 PM"
			addToNewDict(newPersonDict, dateFormatted, text, sender)
			addToNewDict(newFullTextDict, dateFormatted, text)
		if 'y' in input("Enter 'y' if you would like to match duplicate names from Android SMS"):
			matchDuplicates(newPersonDict)
		mergeAndSortPersonDict(newPersonDict, confident)
		mergeAndSortFullTextDict(newFullTextDict)

	parseVCF()
	parseSuperBackup()
	return

"""
For parsing all imessage threads into dictionaries
"""
def parseAllThreads(folder=None):
	global vCardDict
	newPersonDict = {}
	newFullTextDict = {}
	def parseThread(me, fileName):
		if vCardDict == {}:
			parseVCF()
		jsonFile = json.loads(open(fileName).read())
		number = formatPhoneNumber(jsonFile['messages'][0]['handle_id'])
		if number in vCardDict.keys():
			person = vCardDict[number]
		else:
			return 1
		for message in jsonFile['messages']:
			fromMe = message['is_from_me']
			date = message['date']
			text = message['text']
			sender = me if fromMe else titlecase(person)
			if sender not in newPersonDict.keys():
				newPersonDict[sender] = {}
			dateFormatted = datetime.strptime(date, '%Y-%m-%d %H:%M:%S') #"2016-01-13 23:36:32"
			
			addToNewDict(newPersonDict, dateFormatted, text, sender)
			addToNewDict(newFullTextDict, dateFormatted, text)	
		return 0

	if not folder or not os.path.exists(folder):
		wantsToParse = True if 'y' in input("Enter 'y' if you would like to parse your iMessageDatabase (please make a backup first)") else False
		if not wantsToParse:
			return
		folder = folder if folder else "./chatparsed/threads/"
		for file in os.listdir(os.getcwd()): #for file in working directory
			if file.endswith(".db"):
				sqlPath = file
				break
		#imessage_export.main("-i " + sqlPath, "-o " + folder)
		imessage_export.main(sqlPath, folder)
	for root, _, files in list(os.walk(folder)):
		for f in files:
			fullpath = os.path.join(root, f)
			parseThread(me, fullpath)
	if 'y' in input("Enter 'y' if you would like to match duplicates from iMessage"):
		matchDuplicates(newPersonDict)
	mergeAndSortPersonDict(newPersonDict)
	mergeAndSortFullTextDict(newFullTextDict)

"""
Parses file of all vcard data into dictionary mapping phone number/email to name.
All vcards must be in same file. Handles VCF versions 2.x and 3.x
"""
def parseVCF():
	global vCardDict
	def parseVCF3():
		for line in vcfFile:
			if line.startswith('FN'):
				currName = line[3:len(line)-1]
			elif line.startswith('TEL;') or line.startswith('EMAIL'):
				index = line.find('pref:')
				number = line[index + 5:]
				if index == -1:
					index = numer.find(":")
					number = number[index + 1:]
				number = number.rstrip()
				number = formatPhoneNumber(number) #trying this out
				if currName:
					vCardDict[number] = currName
				else: #currName is still None, haven't found a name yet
					vCardDict[number] = number
	def parseVCF2():
		for line in vcfFile:
			if line.startswith('FN'):
				currName = line[3:len(line)-1]
			elif line.startswith('TEL;') or line.startswith('EMAIL'):
				index = line.find(':')
				number = line[index+1:]
				number = number.rstrip()
				number = formatPhoneNumber(number) #trying this out
				if currName:
					vCardDict[number] = currName
				else:#currName is still None, haven't found a name yet
					vCardDict[number] = number

	currName = None #necessary in case first contact has no name associated with it
	vcfFile = open('allme.vcf', 'r')#need to change to below, eventually
	"""
	#only works if there's only one .vcf file		
	for file in os.listdir(os.getcwd()): #for file in working directory
			if file.endswith(".vcf"):
				vcfFile = file
				break
	"""
	i = 0
	for line in vcfFile: #hacky, consider changing
		i += 1
		if line.startswith('VERSION'):
			if '3.' in line:
				parseVCF3()
			else:
				parseVCF2()
"""
Adds a new message entry to the new dictionary. Handles both fullTextDict and personDict.
"""
def addToNewDict(newDict, dateFormatted, text, sender = None):
	if not areEnglishCharacters(sender):
		return
	if sender is None: #newFullTextDict
		if dateFormatted in newDict.keys():
			if text not in newDict[dateFormatted]:
				currLst = list(newDict[dateFormatted])
				currLst.append(text)
				newDict[dateFormatted] = tuple(currLst)
		else:
			newDict[dateFormatted] = tuple([text])
	else: #newPersonDict
		if sender not in newDict.keys():
			newDict[sender] = {}
		if dateFormatted in newDict[sender].keys():
			if text not in newDict[sender][dateFormatted]:
				currLst = list(newDict[sender][dateFormatted])
				currLst.append(text)
				newDict[sender][dateFormatted] = tuple(currLst)
		else:
			newDict[sender][dateFormatted] = tuple([text])

"""
Merges a newDict, which is newly parsed from some platform, with the dictionary that already exists
"""
def mergeAndSortPersonDict(newDict, confident):
	global personDict
	if personDict == {}:
		personDict = newDict
		for name in personDict:
			personDict[name] = OrderedDict(sorted(personDict[name].items(), key=lambda t: t[0]))
		return

	for name in newDict:
		if name not in personDict.keys():
			if name not in aliasDict.keys():
				matchAlias(personDict.keys(), name, newDict, confident)
			trueName = aliasDict[name]
			if trueName in aliasDict.keys():
				trueName = aliasDict[trueName]

		else:
			trueName = name
		for date in newDict[name]:
			if trueName not in personDict.keys():
				personDict[trueName] = {}
			if date not in personDict[trueName]:
				personDict[trueName][date] = newDict[name][date]
			else:
				personDict[trueName][date] = combineMessageTuples(personDict[trueName][date], newDict[name][date])
		personDict[trueName] = OrderedDict(sorted(personDict[trueName].items(), key=lambda t: t[0]))
def mergeAndSortFullTextDict(newDict):
	global fullTextDict
	if fullTextDict == {}:
		fullTextDict = newDict
		fullTextDict = OrderedDict(sorted(fullTextDict.items(), key=lambda t: t[0]))
		return

	for date in newDict:
		if date not in fullTextDict:
			fullTextDict[date] = newDict[date]
		else:
			fullTextDict[date] = combineMessageTuples(fullTextDict[date], newDict[date])
	fullTextDict = OrderedDict(sorted(fullTextDict.items(), key=lambda t: t[0]))


#=====================================================================================================
#                                       Combining Contacts
#=====================================================================================================
"""
add to alias dictionary a mapping from each name in otherNames to a name that's in existing names,
or adds a new name if no good match is found.
Step 1: compile list of possible matches by using minEditDistance (need to deal with middle names, non-English characters, initials for last names, shortened first names)
Step 2: If there are a few possible matches, sort possible matches using elements of writing style, and ask user to confirm
"""
def matchAliases(existingNames, otherNames, otherNamesDict, confident):
	CUTOFFSCORE = 2 #play around with this
	for otherName in otherNames:
		candidates = possMatches(otherName, existingNames) #list of possible matches (determined by small edit distance)
		topCandidate, bestScore = candidates[0]
		correctMatch = False
		if not confident and bestScore < CUTOFFSCORE:
			if otherName.isdigit(): #phone number
				aliasDict[otherName] = otherName
			#if candidates[1][1] >= bestScore - 1: #multiple best matches within 1 of eachother
			elif candidates[1][1] == bestScore: #multiple best matches equal to eachother	
				writingStyleSimilarityDict = {} #candidate existingName -> similarity to otherName 
				toCompare = [candidates[0][0]]
				for candidate in candidates:
					if candidate[1] == bestScore:
						writingStyleSimilarityDict[candidate[0]] = writingStyleMatchScore(otherName, otherNamesDict, candidate[0])
				topCandidates = sorted(writingStyleSimilarityDict.keys(), key = lambda x: -writingStyleSimilarityDict[x])
				i = 0
				while not correctMatch and i < len(topCandidates):
					topCandidate = topCandidates[i]
					correctMatch = True if 'y' in input("Enter 'y' if " + otherName + " should be matched with " + topCandidate + ": ") else False
					i += 1	
			else:
				correctMatch = True if 'y' in input("Enter 'y' if " + otherName + " should be matched with " + topCandidate + ": ") else False
			if correctMatch:
				aliasDict[otherName] = topCandidate
			else:
				aliasDict[otherName] = titlecase(otherName)
		elif confident:
			aliasDict[otherName] = topCandidate
		else:
			aliasDict[otherName] = titlecase(otherName)

def matchAlias(existingNames, otherName, otherNamesDict, confident):
	matchAliases(existingNames, [otherName], otherNamesDict, confident)
	return aliasDict[otherName]

def matchDuplicates(newDict):
	tried = {}
	CUTOFFSCORE = 1 #play around with this
	for name in newDict:
		if name in aliasDict.values():
			continue
		candidates = possMatches(name, newDict.keys(), 3, False) #list of possible matches (determined by small edit distance)
		correctMatch = False
		i = 0
		while not correctMatch and i < len(candidates) and candidates[i][1] <= CUTOFFSCORE:
			topCandidate = candidates[i][0]
			pair1 = (topCandidate, name)
			if pair1 in tried.keys():
				i += 1
				continue
			pair2 = (name, topCandidate)
			tried[pair1] = True
			tried[pair2] = True
			if topCandidate != name:
				correctMatch = True if 'y' in input("Enter 'y' if " + topCandidate + " is a duplicate of " + name + " on the same platform: ") else False
			i += 1
		if correctMatch:
			aliasDict[name] = topCandidate
	return

def possMatches(name, existingNames, number=5, diffDics = True):
	if name in existingNames and diffDics:
		return [(name,0)]
	similarityScores = {} #existing name -> min edit distance
	for existingName in existingNames:
		score = nameSimilarityScore(name, existingName)
		similarityScores[existingName] = score
	sortedByScore = sorted(similarityScores.items(), key = lambda x: x[1])[:number] #?
	return sortedByScore

def nameSimilarityScore(w1, w2):
	def minEditDistance(partW1, partW2):
		def diff(a, b):
			return 0 if a == b else 1
		if len(partW1) == 0 or len(partW2)==0:
			return max(len(partW1), len(partW2))
		table = [[0 for _ in range(len(partW2))] for _ in range(len(partW1))]
		for i in range(0, len(partW1)):
			table[i][0] = i
		for j in range(1, len(partW2)):
			table[0][j] = j
		for i in range(1, len(partW1)):
			for j in range(1, len(partW2)):
				table[i][j] = min(table[i-1][j] + 1, table[i][j-1] + 1, table[i-1][j-1] + diff(partW1[i], partW2[j]))
		return table[len(partW1) - 1][len(partW2) - 1]
	
	FIRSTNAMEMATCHSCORE = .1 #play around with this value
	allPartScore = 0
	splitW1 = w1.split(" ")
	splitW2 = w2.split(" ")
	if splitW1[0] == splitW2[0] or splitW1[0].startswith(splitW2[0]) or splitW2[0].startswith(splitW1[0]): #first name match, cor one is a prefix of the other
		if splitW1[len(splitW1)-1] == splitW2[len(splitW2)-1]: #first and last name match
			return 0
		else:
			if len(splitW1) == 1 or len(splitW2) == 1:#one of the names is just a first name
				return FIRSTNAMEMATCHSCORE
			else: #both are more than just a first name
				splitW1 = splitW1[1:]
				splitW2 = splitW2[1:]
	w1 = " ".join(splitW1)
	w2 = " ".join(splitW2)
	return minEditDistance(w1, w2)
"""
computes the cosine similarity of the tfidf vectors formed from all the words the unknown person and possible match have typed
"""
def writingStyleMatchScore(otherName, otherNamesDict, possibleExistingMatch):
	#http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/
	existingNameText = " ".join(fullMessageList(possibleExistingMatch))
	otherNameText = " ".join(fullMessageList(otherName, otherNamesDict))
	tfidf_vectorizer = TfidfVectorizer(min_df=1)
	tfidf_matrix = tfidf_vectorizer.fit_transform(tuple([existingNameText, otherNameText]))
	similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0]
	return similarity

#=====================================================================================================
#                                        Analytics/Fun
#=================================================== ==================================================

def wordCloud(person):
	text = getAllMessagesAsString(person)

	# Generate a word cloud image
	wordcloud = WordCloud().generate(text)

	# Display the generated image:
	# the matplotlib way:
	import matplotlib.pyplot as plt
	# take relative word frequencies into account, lower max_font_size
	wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text)
	plt.figure()
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()

def mostCommonNGrams(n, number, person = None):
	"""
	get common phrases of length n
	"""
	from nltk import ngrams
	if person:
		wordList = fullWordListPerson(person)
	else:
		wordList = fullWordList()
	grams = list(ngrams(wordList, n))
	counts = {}
	for gram in grams:
		if gram not in counts:
			counts[gram] = 1
		else:
			counts[gram] += 1
	return sorted(counts.keys(), key = lambda x: -counts[x])[:number]

def plotTopFriends(number = 15):
	import matplotlib.pyplot as plt
	earliestDateTime = min(fullTextDict.keys())
	earliestYear = earliestDateTime.year
	lastDateTime = max(fullTextDict.keys())
	lastYear = lastDateTime.year
	messageCount = {} #key = person, value = [month1count, month2count, ...]
	topFriendsList = topFriends(number)
	for friend in topFriendsList:
		messageCount[friend] = []
	for year in range(earliestYear, lastYear+1):
		for month in range(1, 13):
				monthStart = datetime(int(year), int(month), 1)
				monthEnd = datetime(int(year), int(month), calendar.monthrange(int(year),int(month))[1])
				for friend in topFriendsList:
					messageCount[friend].append((monthStart, numMessagesMonth(friend, monthStart, monthEnd)))
	for friend, mcTups in sorted(messageCount.items(), key= lambda x: -sum([count for month, count in x[1]])):
		counts = [count for (monthStart, count) in mcTups]
		months = [monthStart for (monthStart, count) in mcTups]
		plt.plot(months, counts, label = friend)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.show()

def plotFriendSentiment(number = 15):
	import matplotlib.pyplot as plt
	earliestDateTime = min(fullTextDict.keys())
	earliestYear = earliestDateTime.year
	lastDateTime = max(fullTextDict.keys())
	lastYear = lastDateTime.year
	sentiments = {} #key = person, value = [month1sent, month2sent, ...]
	topFriendsList = topFriends(number)
	for friend in topFriendsList:
		sentiments[friend] = []
	for year in range(earliestYear, lastYear+1):
		for month in range(1, 13):
				monthStart = datetime(int(year), int(month), 1)
				monthEnd = datetime(int(year), int(month), calendar.monthrange(int(year),int(month))[1])
				for friend in topFriendsList:
					sentiments[friend].append((monthStart, personAvgSentiment(friend, month, year)))
	for friend, sentTups in sorted(sentiments.items(), key= lambda x: -sum([sent for month, sent in x[1]])):
		sents = [sent for (monthStart, sent) in sentTups]
		months = [monthStart for (monthStart, sent) in sentTups]
		plt.plot(months, sents, label = friend)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.show()

def mostSimilarFriends():
	return maxPairWritingStyleMatchScore(personDict.keys())

def maxPairWritingStyleMatchScore(people, number = 10):
	#http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/
	textList = []
	orderingDict = {} # i -> name
	scoreDict = {} #(personi, personj) -> score
	i = 0
	for p in people:
		pText = " ".join(fullMessageList(p))
		textList.append(pText)
		orderingDict[i] = p
		i += 1
	tfidf_vectorizer = TfidfVectorizer(min_df=1)
	tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(textList))
	for i in range(tfidf_matrix.shape[0]):
		for j in range(tfidf_matrix.shape[0]):
			if i < j and len(personDict[orderingDict[i]]) > 100 and len(personDict[orderingDict[j]]) > 100: #minimum of 100 messages for both people
				score = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[j:j+1])[0][0]
				if len(scoreDict) <= number: #number of pairs
					scoreDict[(i,j)] = score
				else:
					sortedScores = sorted(scoreDict.items(), key = lambda x: x[1])
					leastSim = sortedScores[0]
					if leastSim[1] < score:
						del scoreDict[leastSim[0]]
						scoreDict[(i,j)] = score
				# 	del scoreDict[sortedScores[0][0]]
				# 	scoreDict[(i,j)] = similarities[j]
	return [(orderingDict[i], orderingDict[j], score) for (i,j), score in sorted(scoreDict.items(), key = lambda x: -x[1])]

"""
Takes tf_idf vectors of top friends, identifies the most distinguishing features (PCA), and reduces the dimensionality while seeking to preserve distances 
between vectors, and projects the vectors onto 2D plane to show similarity between friends' word choice.
"""
def similarityPlot():
	import matplotlib.pyplot as plt
	from matplotlib import rcParams
	tfidf_vectorizer = TfidfVectorizer(min_df=1)
	names = friendsAboveMinNumMessages(200) + [me]
	data = []
	words = [] #ordering of words in tf_idf matrix
	wordsSet = set() #for faster lookup
	nameSet = set()
	for person in personDict:
		for name in person.split():
			nameSet.add(name)
			nameSet.add(name.lower())
	for i in range(len(names)):
		data.append(getAllMessagesAsString(names[i], False))
	tfidf_matrix = tfidf_vectorizer.fit_transform(data)
	featureNames = tfidf_vectorizer.get_feature_names()
	tfidf_arr = tfidf_matrix.toarray()
	for j in range(len(tfidf_arr[0])):
		word = tfidf_arr[0][j]
		if word not in wordsSet:
			words.append(word)
			wordsSet.add(j)
	#nmds = manifold.MDS(metric = True, n_components = N_DISTINGUISHING_FEATURES) 
	#npos = nmds.fit_transform(tfidf_matrix.toarray())
	clf = PCA(n_components=2)
	npos = clf.fit_transform(tfidf_arr)
	plt.scatter(npos[:, 0], npos[:, 1], marker = 'o', c = 'b', cmap = plt.get_cmap('Spectral')) #change colors
	for name, x, y in zip(names, npos[:, 0], npos[:, 1]):
		plt.annotate(
			name, 
			xy = (x, y), xytext = (-20, 20),
			textcoords = 'offset points', ha = 'right', va = 'bottom',
			bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
	fig, ax = plt.subplots()
	ax2 = ax.twinx()
	xAxisP = [featureNames[i] for i in np.argpartition(clf.components_[0], -50)[-50:] if featureNames[i] not in nameSet]
	yAxisP = [featureNames[i] for i in np.argpartition(clf.components_[1], -50)[-50:] if featureNames[i] not in nameSet]
	xAxisN = [featureNames[i] for i in np.argpartition(-clf.components_[0], -50)[-50:] if featureNames[i] not in nameSet]
	yAxisN = [featureNames[i] for i in np.argpartition(-clf.components_[1], -50)[-50:] if featureNames[i] not in nameSet]
	ax.set_xlabel("Most Postively influential words along x axis:\n" + ", ".join(xAxisP), fontsize=18)
	ax.set_ylabel("Most Postively influential words along y axis:\n" + ", ".join(yAxisP), fontsize=18)
	ax2.set_xlabel("Most Negatively influential words along x axis:\n" + ", ".join(xAxisN), fontsize=18)
	ax2.set_ylabel("Most Negatively influential words along y axis:\n" + ", ".join(yAxisN), fontsize=18)
	# xAxis = [featureNames[i] for i in np.argpartition(np.absolute(clf.components_[0]), -50)[-50:] if featureNames[i] not in nameSet]
	# yAxis = [featureNames[i] for i in np.argpartition(np.absolute(clf.components_[1]), -50)[-50:] if featureNames[i] not in nameSet]
	# for i in range(1, max(len(xAxis), len(yAxis)) ):
	# 	if i % 20 == 0 and i < len(xAxis):
	# 		xAxis[i] += "\n"
	# 	if i % 15 == 0 and i < len(yAxis):
	# 		yAxis[i] += "\n"
	# plt.xlabel("Most influential words along x axis:\n" + ", ".join(xAxis), fontsize=18)
	# plt.ylabel("Most influential words along y axis:\n" + ", ".join(yAxis), fontsize=18)
	rcParams.update({'figure.autolayout': True})
	plt.suptitle("Word-Usage Similarity Scatterplot", fontsize = 24, fontweight = 'bold')
	plt.show()

def similarityPlot3():
	import matplotlib.pyplot as plt
	from matplotlib import rcParams
	from mpl_toolkits.mplot3d import Axes3D
	from mpl_toolkits.mplot3d import proj3d
	tfidf_vectorizer = TfidfVectorizer(min_df=1)
	names = friendsAboveMinNumMessages(200) + [me]
	data = []
	words = [] #ordering of words in tf_idf matrix
	wordsSet = set() #for faster lookup
	nameSet = set()
	for person in personDict:
		for name in person.split():
			nameSet.add(name)
			nameSet.add(name.lower())
	for i in range(len(names)):
		data.append(getAllMessagesAsString(names[i], False))
	tfidf_matrix = tfidf_vectorizer.fit_transform(data)
	featureNames = tfidf_vectorizer.get_feature_names()
	tfidf_arr = tfidf_matrix.toarray()
	for j in range(len(tfidf_arr[0])):
		word = tfidf_arr[0][j]
		if word not in wordsSet:
			words.append(word)
			wordsSet.add(j)
	clf = PCA(n_components=3)
	npos = clf.fit_transform(tfidf_arr)
	visualize3DData(npos, clf, featureNames)

def visualize3DData (X, clf, featureNames):
	"""Visualize data in 3d plot with popover next to mouse position.

	Args:
		X (np.array) - array of points, of shape (numPoints, 3)
	Returns:
		None
	"""
	import matplotlib.pyplot as plt, numpy as np
	from mpl_toolkits.mplot3d import proj3d
	fig = plt.figure(figsize = (16,10))
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], depthshade = False, picker = True)
	names = friendsAboveMinNumMessages(200) + [me]
	data = []
	words = [] #ordering of words in tf_idf matrix
	wordsSet = set() #for faster lookup
	nameSet = set()
	for person in personDict:
		for name in person.split():
			nameSet.add(name)
			nameSet.add(name.lower())
	xAxis = [featureNames[i] for i in np.argpartition(np.absolute(clf.components_[0]), -50)[-50:] if featureNames[i] not in nameSet]
	yAxis = [featureNames[i] for i in np.argpartition(np.absolute(clf.components_[1]), -50)[-50:] if featureNames[i] not in nameSet]
	zAxis = [featureNames[i] for i in np.argpartition(np.absolute(clf.components_[2]), -50)[-50:] if featureNames[i] not in nameSet]
	ax.set_xlabel("Most influential words along x axis:\n" + ", ".join(xAxis), fontsize=18)
	ax.set_ylabel("Most influential words along y axis:\n" + ", ".join(yAxis), fontsize=18)
	ax.set_zlabel("Most influential words along z axis:\n" + ", ".join(zAxis), fontsize=18)
	def distance(point, event):
		"""Return distance between mouse position and given data point
		Args:
			point (np.array): np.array of shape (3,), with x,y,z in data coords
			event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
		Returns:
			distance (np.float64): distance (in screen coords) between mouse pos and data point
		"""
		assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape
		# Project 3d data space to 2d data space
		x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
		# Convert 2d data space to 2d screen space
		x3, y3 = ax.transData.transform((x2, y2))
		return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)
	def calcClosestDatapoint(X, event):
		""""Calculate which data point is closest to the mouse position.

		Args:
			X (np.array) - array of points, of shape (numPoints, 3)
			event (MouseEvent) - mouse event (containing mouse position)
		Returns:
			smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
		"""
		distances = [distance (X[i, 0:3], event) for i in range(X.shape[0])]
		return np.argmin(distances)
	labelDic = {}
	def annotatePlot(X, index):
		"""Create popover label in 3d chart

		Args:
			X (np.array) - array of points, of shape (numPoints, 3)
			index (int) - index (into points array X) of item which should be printed
		Returns:
			None
		"""
		# If we have previously displayed another label, remove it first
		if hasattr(annotatePlot, 'label'):
			annotatePlot.label.remove()
		# Get data point from array of points X, at position index
		if index not in labelDic:
			x2, y2, _ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())
			labelDic[index] = (x2, y2)
		x2, y2 = labelDic[index]
		annotatePlot.label = plt.annotate(names[index],
			xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
			bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
		fig.canvas.draw()
	def onMouseMotion(event):
		"""Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
		closestIndex = calcClosestDatapoint(X, event)
		annotatePlot (X, closestIndex)
	fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion
	plt.show()
#make messages searchable
#most similar friends in terms of words used
#make function that makes huge png of all messages (maybe in shape of picture)
#phraseCloud
#graph slang usage over time

	
#=====================================================================================================
#                                           Helpers/Utilities
#=====================================================================================================
def topFriends(number):
	temp = sorted(personDict.keys(), key=lambda x: -len(personDict[x]))
	i = 0
	topFriends = []
	for person in temp:
		if i < number:
			if person != me:
				topFriends.append(person)
				i += 1
		else:
			return topFriends

def topFriendsMonth(number, month, year): 
	monthStart = datetime(int(year), int(month), 1)
	monthEnd = datetime(int(year), int(month), calendar.monthrange(int(year),int(month))[1])
	temp = sorted(personDict.keys(), key= lambda x: -numMessagesMonth(x, monthStart, monthEnd))
	i = 0
	topFriends = []
	for person in temp:
		if i < number:
			if person != me:
				topFriends.append(person)
				i += 1
		else:
			return topFriends

def friendsAboveMinNumMessages(number = 100):
	temp = sorted(personDict.keys(), key=lambda x: -len(personDict[x]))
	topFriends = []
	for person in temp:
		if len(personDict[person]) >= number:
			if person != me:
				topFriends.append(person)
		else:
			return topFriends

def getAllMessagesAsString(personStr, includeSW = True):
	string = ""
	for messages in personDict[personStr].values():
		for message in messages:
			if not includeSW:
				messageWords = message.split()
				if type(messageWords) != None and len(messageWords) > 0:
					afterRemoval = [word for word in messageWords if word.lower() not in STOPWORDS]
					message = ' '.join(afterRemoval)
			string += message + " "
	return string

def numMessagesMonth(person, monthStart, monthEnd):
	count = 0
	for datetime in personDict[person]:
		if datetime >= monthStart and datetime <= monthEnd:
			count += 1
	return count
def fullMessageList(name, sourceDict=None):
	if not sourceDict:
		sourceDict = personDict[name]
	fullMessageList = []
	for messTups in sourceDict.values():
		if type(messTups) == dict:
			messTups = messTups.values()
		i = 0
		if i>0:
			break
		i += 1
		for message in messTups:
			if type(message) == tuple:
				for mess in message:
					fullMessageList.append(mess)
			else:
				fullMessageList.append(message)
	return fullMessageList
def fullMessageListMonth(name, month, year):
	messageLst = []
	monthStart = datetime(int(year), int(month), 1)
	monthEnd = datetime(int(year), int(month), calendar.monthrange(int(year),int(month))[1])
	for dt in personDict[name]:
		if dt >= monthStart and dt <= monthEnd:
			for message in personDict[name][dt]:
				messageLst.append(message)
	return messageLst
			
def fullWordList():
	fullWordList = []
	for messTup in fullTextDict.values():
		i = 0
		if i>0:
			break
		i += 1
		for mess in messTup:
			words = mess.split(" ")
			for word in words:
				fullWordList.append(word)
	return fullWordList

def fullWordListPerson(name, sourceDict=None):
	if not sourceDict:
		sourceDict = personDict[name]
	fullWordList = []
	for messTups in sourceDict.values():
		if type(messTups) == dict:
			messTups = messTups.values()
		i = 0
		if i>0:
			break
		i += 1
		for message in messTups:
			if type(message) == tuple:
				for mess in message:
					words = mess.split(" ")
					for word in words:
						fullWordList.append(word)
			else:
				words = message.split(" ")
				for word in words:
					fullWordList.append(word)
	return fullWordList

def slangList():
	from nltk.corpus import words
	wordSet = set(words.words())
	lst = [slang for slang in fullWordList() if slang not in wordSet and isSlang(slang) and slang[:len(slang) - 1] not in wordSet]
	return lst
def isSlang(word):
	wordExceptLast = word[:len(slang) - 1]
	if isProperNoun(word) or isProperNoun(wordExceptLast):
		return False
	if 'www' in word or "'" in word:
		return False
	return True
def isProperNoun(word):
	return titlecase(word) == word 
# def q():
# 	from nltk.metrics.association import QuadgramAssocMeasures
# 	quadgram_measures = QuadgramAssocMeasures()
# 	finder = QuadgramCollocationFinder.from_words(fullWordList())
# 	finder.nbest(quadgram_measures.pmi, 10)  # doctest: +NORMALIZE_WHITESPACE
def personAvgSentiment(person, month = None, year = None):
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	comp = 0
	sid = SentimentIntensityAnalyzer()
	if month:
		msgLst = fullMessageListMonth(person, month, year)
	else:	
		msgLst = fullMessageList(person)
	for message in msgLst:
		sentimentDict = sid.polarity_scores(message)
		comp += sentimentDict['compound']
	return comp/len(msgLst) if len(msgLst) != 0 else 0


def messageSentiment(message):
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	#may need to download vader_lexicon after calling nltk.download()
	sid = SentimentIntensityAnalyzer()
	return sid.polarity_scores(message)

"""
908-872-6993
+13106996932
1 (818) 884-9040
(510) 642-9255
dmg5233@lausd.net

"""
def formatPhoneNumber(pnStr):
	if '@' in pnStr or '#' in pnStr or pnStr == "": #or len(pnStr) == 0: #when/why does the length == 0? 
		return pnStr
	reformattedStr = ''.join(filter(lambda x: x.isdigit(), pnStr))
	if not reformattedStr:
		return 
	elif reformattedStr[0] == '1':
		return reformattedStr[1:]
	return reformattedStr


def combineMessageTuples(tup1, tup2):
	currLst1 = list(tup1)
	currLst2 = list(tup2)
	currLst1 += currLst2
	return tuple(currLst1)

def areEnglishCharacters(s):
	#http://stackoverflow.com/a/27084708
	if not s:
		return True
	try:
		emoji_pattern = re.compile("["
		u"\U0001F600-\U0001F64F"  # emoticons
		u"\U0001F300-\U0001F5FF"  # symbols & pictographs
		u"\U0001F680-\U0001F6FF"  # transport & map symbols
		u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
						   "]+", flags=re.UNICODE) #http://stackoverflow.com/a/33417311
		s = emoji_pattern.sub(r'', s)
		s.encode('ascii')
	except UnicodeEncodeError:
		return False
	else:
		return True



def main(username, confident):
	global me
	me = username
	parseFBMessages(confident)
	parseSMS(me, confident)
	#parseAllThreads()

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2] if len(sys.argv) >=3 else False)
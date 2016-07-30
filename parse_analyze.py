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

#=====================================================================================================
#                                  Parsing Messages and Contact Info
#=====================================================================================================
me = None
personDict = {} #name -> dateTime -> tuple of messages
fullTextDict = {} #dateTime -> tuple of everyone's messages
vCardDict = {} #phone number/email -> name
aliasDict = {} #alias1 -> alias2, alias2 -> alias1
"""
For parsing all Facebook messages into dictionaries
"""
def parseFBMessages():
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
	mergeAndSortPersonDict(newPersonDict)
	mergeAndSortFullTextDict(newFullTextDict)

"""
Parsing .xml file containing all sms ("Super Backup" for Android)
"""
def parseSMS(me):
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
		mergeAndSortPersonDict(newPersonDict)
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
		folder = folder if folder else "./chatparsed2/threads/"
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
All vcards must be in same file.
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
					index = number.find(":")
					number = number[index + 1:]
				number = number.rstrip()
				number = formatPhoneNumber(number) #trying this out
				vCardDict[number] = currName
	def parseVCF2():
		for line in vcfFile:
			if line.startswith('FN'):
				currName = line[3:len(line)-1]
			elif line.startswith('TEL;') or line.startswith('EMAIL'):
				index = line.find(':')
				number = line[index+1:]
				number = number.rstrip()
				number = formatPhoneNumber(number) #trying this out
				vCardDict[number] = currName
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

def mergeAndSortPersonDict(newDict):
	global personDict
	if personDict == {}:
		personDict = newDict
		for name in personDict:
			personDict[name] = OrderedDict(sorted(personDict[name].items(), key=lambda t: t[0]))
		return

	for name in newDict:
		if name not in personDict.keys():
			if name not in aliasDict.keys():
				matchAlias(personDict.keys(), name, newDict)
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
	what to do if there are no good matches? 
Step 2: If there are a few possible matches, match using elements of writing style
"""
def matchAliases(existingNames, otherNames, otherNamesDict):
	CUTOFFSCORE = 2 #play around with this
	for otherName in otherNames:
		candidates = possMatches(otherName, existingNames) #list of possible matches (determined by small edit distance)
		topCandidate, bestScore = candidates[0]
		correctMatch = False
		if bestScore < CUTOFFSCORE:
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
		else:
			aliasDict[otherName] = titlecase(otherName)
		#print(otherName, candidates)

def matchAlias(existingNames, otherName, otherNamesDict):
	matchAliases(existingNames, [otherName], otherNamesDict)
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
	#return sorted(editDistances.keys(), key = lambda x: editDistances[x])[:5] #?
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

def writingStyleMatchScore(otherName, otherNamesDict, possibleExistingMatch):
	#http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/
	existingNameText = " ".join(fullMessageList(possibleExistingMatch))
	otherNameText = " ".join(fullMessageList(otherName, otherNamesDict))
	tfidf_vectorizer = TfidfVectorizer(min_df=1)
	tfidf_matrix = tfidf_vectorizer.fit_transform(tuple([existingNameText, otherNameText]))
	similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0]
	#print(otherName, possibleExistingMatch, similarity)
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

def mostCommonNGrams(n, number):
	"""
	get common phrases of length n
	"""
	from nltk import ngrams
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
				#print(similarity)
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
Takes tf_idf vectors of top friends, reduces the dimensionality while seeking to preserve distances between vectors (MDS)
identifies the most distinguishing features (PCA), and projects the vectors onto 2D plane to show similarity between friends' word choice.
"""
def similarityPlot():
	import matplotlib.pyplot as plt
	N_DISTINGUISHING_FEATURES = 10
	tfidf_vectorizer = TfidfVectorizer(min_df=1)
	names = friendsAboveMinNumMessages(200) + [me]
	data = []
	for i in range(len(names)):
		data.append(getAllMessagesAsString(names[i]))
	tfidf_matrix = tfidf_vectorizer.fit_transform(data)
	nmds = manifold.MDS(metric = True, n_components = N_DISTINGUISHING_FEATURES) 
	npos = nmds.fit_transform(tfidf_matrix.toarray())
	clf = PCA(n_components=N_DISTINGUISHING_FEATURES)
	npos = clf.fit_transform(npos)
	plt.scatter(npos[:, 0], npos[:, 1], marker = 'o', c = 'b', cmap = plt.get_cmap('Spectral')) #change colors
	for name, x, y in zip(names, npos[:, 0], npos[:, 1]):
	    plt.annotate(
	        name, 
	        xy = (x, y), xytext = (-20, 20),
	        textcoords = 'offset points', ha = 'right', va = 'bottom',
	        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
	        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

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

def getAllMessagesAsString(personStr):
	string = ""
	for messages in personDict[personStr].values():
		for message in messages:
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
	#return True #remove when needed
	#http://stackoverflow.com/a/27084708
	if not s:
		return True
	try:
		s.encode('ascii')
	except UnicodeEncodeError:
		return False
	else:
		return True



def main(username):
	global me
	me = username
	parseFBMessages()
	parseSMS(me)
	#parseAllThreads()

if __name__ == "__main__":
    main(sys.argv[1])
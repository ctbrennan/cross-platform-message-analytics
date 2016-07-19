import sys
#sys.path.insert(0, '/home/connor/Documents/prj/parser/FB-Message-Parser/')
import fb_parser
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
from sklearn.feature_extraction.text import TfidfVectorizer #finding similarity between two texts, requires scipy
#=====================================================================================================
#                                  Parsing Messages and Contact Info
#=====================================================================================================
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
	#jsonFile = json.loads(open('parser/FB-Message-Parser/messages.json').read())
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
		"""
		newNames = []
		notFound = []
		for message in root:
			phoneNumber = formatPhoneNumber(message.attrib['address'])
			if message.attrib['name']:
				sender = titlecase(message.attrib['name'])
			elif phoneNumber in vCardDict.keys():
				sender = titlecase(vCardDict[phoneNumber])
				if sender not in newNames:
					newNames.append(sender)
			else:
				sender = phoneNumber
				if sender not in notFound:
					notFound.append(sender)
		
		if newPersonDict.keys():
			matchAliases(newPersonDict.keys(), newNames)
		"""
		for message in root:
			date = message.attrib['time']
			text = message.attrib['body']
			sender = titlecase(message.attrib['name']) if message.attrib['name'] else me
			"""
			if sender not in newPersonDict.keys():
				if sender in aliasDict.keys():
					sender = aliasDict[sender]
				else:
					newPersonDict[sender] = {}
			"""
			dateFormatted = datetime.strptime(date, '%b %d, %Y %I:%M:%S %p') #"Jul 10, 2016 8:28:10 PM"
			addToNewDict(newPersonDict, dateFormatted, text, sender)
			addToNewDict(newFullTextDict, dateFormatted, text)
		mergeAndSortPersonDict(newPersonDict)
		mergeAndSortFullTextDict(newFullTextDict)

	parseVCF()
	parseSuperBackup()
	return

"""
For parsing all imessage threads into dictionaries
"""
def parseAllThreads(me, folder):
	def parseThread(me, fileName):
		if vCardDict == {}:
			parseVCF()
		jsonFile = json.loads(open(fileName).read())
		number = jsonFile['messages'][0]['handle_id']
		if number in vCardDict.keys():
			person = vCardDict[number]
		else:
			return 1
		newPersonDict = {}
		newFullTextDict = {}
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
			"""
			if dateFormatted in personDict[sender].keys():
				currLst = list(personDict[sender][dateFormatted])
				currLst.append(text)
				personDict[sender][dateFormatted] = tuple(currLst)
			else:
				personDict[sender][dateFormatted] = tuple([text])
			if dateFormatted in fullTextDict.keys():
				currLst = list(fullTextDict[dateFormatted])
				currLst.append(text)
				fullTextDict[dateFormatted] = tuple(currLst)
			else:
				fullTextDict[dateFormatted] = tuple([text])
			"""
		return 0
	notSaved = 0
	for root, _, files in os.walk(folder):
	    for f in files:
	        fullpath = os.path.join(root, f)
	        notSaved += parseThread(me, fullpath)
	#print(notSaved)


"""
Parses file of all vcard data into dictionary mapping phone number/email to name.
All vcards must be in same file.
"""
def parseVCF():
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
	vcfFile = open('allme.vcf', 'r')#need to fix later
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
			currLst = list(newDict[dateFormatted])
			currLst.append(text)
			newDict[dateFormatted] = tuple(currLst)
		else:
			newDict[dateFormatted] = tuple([text])
	else: #newPersonDict
		if sender not in newDict.keys():
			newDict[sender] = {}
		if dateFormatted in newDict[sender].keys():
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
			fullTextDict[date] = newDict[person][date]
		else:
			fullTextDict[date] = combineMessageTuples(fullTextDict[date], newDict[date])
	fullTextDict = OrderedDict(sorted(fullTextDict.items(), key=lambda t: t[0]))




#=====================================================================================================
#                                       Combining Contacts
#=====================================================================================================
"""
add to alias dictionary a mapping from each name in otherNames to a name that's in existing names,
or adds a new name if no good match is found.
Step 1: compile list of possible matches by using minEditDistance (need to deal with middle names, non-English characters, initials for last names)
	what to do if there are no good matches? 
Step 2: Somone is less likely to be a match if the existing name already corresponds to many other names
Step 3: If there are a few possible matches, match using elements of writing style
"""
def matchAliases(existingNames, otherNames, otherNamesDict):
	for otherName in otherNames:
		candidates = possMatches(otherName, existingNames) #list of possible matches (determined by small edit distance)
		topCandidate, bestScore = candidates[0]
		CUTOFFSCORE = 3 #play around with this
		if bestScore < CUTOFFSCORE:
			if candidates[1][1] == bestScore: #multiple best matches
				writingStyleSimilarityDict = {} #candidate existingName -> similarity to otherName 
				toCompare = [candidates[0][0]]
				for candidate in candidates:
					if candidate[1] == bestScore:
						writingStyleSimilarityDict[candidate[0]] = writingStyleMatchScore(otherName, otherNamesDict, candidate[0])
				topCandidate = sorted(writingStyleSimilarityDict.items(), key = lambda x: writingStyleSimilarityDict[x])[0]
			aliasDict[otherName] = titlecase(topCandidate)
		else:
			aliasDict[otherName] = titlecase(otherName)
		print(otherName, candidates)

def matchAlias(existingNames, otherName, otherNamesDict):
	matchAliases(existingNames, [otherName], otherNamesDict)
	return aliasDict[otherName]

def possMatches(name, existingNames, number=5):
	if name in existingNames:
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
	if splitW1[0] == splitW2[0]: #first name match
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
	allPartScore = minEditDistance(w1, w2)
	if len(splitW1) == len(splitW2) or allPartScore < (abs(len(w1)-len(w2))*2): #is there a better multiplier?
		return allPartScore 
	else:
		if len(splitW1) > len(splitW2):
			extra = splitW1
			less = splitW2
		else:
			extra = splitW2
			less = splitW1
		extraPartTups = itertools.combinations(extra, len(less))
		minScore = allPartScore
		for extraPartTup in extraPartTups:
			score = 0
			for i in range(len(extraPartTup)):
				score += minEditDistance(extraPartTup[i], less[i])
			minScore = min(score, minScore)
		return minScore
	"""

def writingStyleMatchScore(otherName, otherNamesDict, possibleExistingMatch):
	#http://stackoverflow.com/a/8897648
	existingNameText = " ".join(fullMessageList(possibleExistingMatch))
	otherNameText = " ".join(fullMessageList(otherName, otherNamesDict)) 
	print(otherNameText)
	print(type(otherNameText))
	print(otherName)
	vect = TfidfVectorizer(min_df=1)
	score = vect.fit_transform(existingNameText, otherNameText)
	print(score)
	return score
	

	


#=====================================================================================================
#                                        Analytics/Fun
#=================================================== ==================================================

def wordCloud(personStr):
	text = getAllMessagesAsString(personStr)

	# Generate a word cloud image
	wordcloud = WordCloud().generate(text)

	# Display the generated image:
	# the matplotlib way:
	import matplotlib.pyplot as plt
	"""
	plt.imshow(wordcloud)
	plt.axis("off")
	"""
	# take relative word frequencies into account, lower max_font_size
	wordcloud = WordCloud(max_font_size=40, relative_scaling=.5).generate(text)
	plt.figure()
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()

def topFriends(me, number):
	if personDict == {}:
		parseFBMessages()
	#temp = OrderedDict(sorted(personDict.keys(), key=lambda t: t[0]))
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
			break

def topFriendsMonth(me, number, month, year) : 
	monthStart = datetime(int(year), int(month), 1)
	monthEnd = datetime(int(year), int(month), calendar.monthrange(int(year),int(month))[1])
	if personDict == {}:
		parseFBMessages()
	#temp = sorted(personDict.keys(), key= lambda x: -numMessagesMonth(x, month, year))
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
			break


def plotTopFriendsOverTime(me, number):
	import matplotlib.pyplot as plt
	if personDict == {}:
		parseFBMessages()
	earliestDateTime = min(fullTextDict.keys())
	earliestYear = earliestDateTime.year
	lastDateTime = max(fullTextDict.keys())
	lastYear = lastDateTime.year
	messageCount = {} #key = person, value = [month1count, month2count, ...]
	topFriendsList = topFriends(me, number)
	for friend in topFriendsList:
		messageCount[friend] = []
	for year in range(earliestYear, lastYear+1):
		for month in range(1, 13):
				monthStart = datetime(int(year), int(month), 1)
				monthEnd = datetime(int(year), int(month), calendar.monthrange(int(year),int(month))[1])
				for friend in topFriendsList:
					messageCount[friend].append(numMessagesMonth(friend, monthStart, monthEnd))
	for friend, countList in messageCount.items():
		plt.plot(countList, label = friend)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.show()

#make messages searchable

#=====================================================================================================
#                                           Helpers/Utilities
#=====================================================================================================
def getAllMessagesAsString(personStr):
	string = ""
	for messages in personDict[personStr].values():
		for message in messages:
			string += message + " "
	return string
def numMessagesMonth(person, monthStart, monthEnd):
	# monthStart = datetime(int(year), int(month), 1)
	# monthEnd = datetime(int(year), int(month), calendar.monthrange(int(year),int(month))[1])
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
	if fullTextDict == {}:
		parseFBMessages()
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

def fullWordList(name, sourceDict=None):
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
	if reformattedStr[0] == '1':
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
    

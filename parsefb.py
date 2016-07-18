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
	#global personDict, fullTextDict
	for person in jsonFile['threads']:
		for message in person['messages']:
			sender = message['sender']
			if sender not in personDict.keys():
				personDict[sender] = {}
			dateFormatted = datetime.strptime(message['date_time'], '%A, %B %d, %Y at %I:%M%p ')

			if dateFormatted in personDict[message['sender']].keys():
				currLst = list(personDict[message['sender']][dateFormatted])
				currLst.append(message['text'])
				personDict[message['sender']][dateFormatted] = tuple(currLst)
			else:
				personDict[message['sender']][dateFormatted] = tuple([message['text']])
			if dateFormatted in fullTextDict.keys():
				currLst = list(fullTextDict[dateFormatted])
				currLst.append(message['text'])
				fullTextDict[dateFormatted] = tuple(currLst)
			else:
				fullTextDict[dateFormatted] = tuple([message['text']])

	for sender in personDict.keys():
		personDict[sender] = OrderedDict(sorted(personDict[sender].items(), key=lambda t: t[0]))
	fullTextDict = OrderedDict(sorted(fullTextDict.items(), key=lambda t: t[0]))

"""
Parsing .xml file containing all sms ("Super Backup" for Android)
"""
def parseSMS(me):
	def parseSuperBackup():
		tree = ET.parse('allsms.xml') #fix this later
		root = tree.getroot()
		newNames = []
		notFound = []
		for message in root:
			phoneNumber = formatPhoneNumber(message.attrib['address'])
			if message.attrib['name']:
				sender = message.attrib['name'] 
			elif phoneNumber in vCardDict.keys():
				sender = vCardDict[phoneNumber]
				if sender not in newNames:
					newNames.append(sender)
			else:
				sender = phoneNumber
				if sender not in notFound:
					notFound.append(sender)
		if personDict.keys():
			matchAliases(personDict.keys(), newNames)
		for message in root:
			date = message.attrib['time']
			text = message.attrib['body']
			sender = message.attrib['name'] if message.attrib['name'] else me
			if sender not in personDict.keys():
				if sender in aliasDict.keys():
					sender = aliasDict[sender]
				else:
					personDict[sender] = {}
			dateFormatted = datetime.strptime(date, '%b %d, %Y %I:%M:%S %p') #"Jul 10, 2016 8:28:10 PM"
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
		for message in jsonFile['messages']:
			fromMe = message['is_from_me']
			date = message['date']
			text = message['text']
			sender = me if fromMe else person
			if sender not in personDict.keys():
				personDict[sender] = {}
			dateFormatted = datetime.strptime(date, '%Y-%m-%d %H:%M:%S') #"2016-01-13 23:36:32"
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
	if sender is None: #fullTextDict
		if dateFormatted in newDict.keys():
			currLst = list(newDict[dateFormatted])
			currLst.append(text)
			newDict[dateFormatted] = tuple(currLst)
		else:
			newDict[dateFormatted] = tuple([text])
	else:
		if dateFormatted in newDict[sender].keys():
			currLst = list(newDict[sender][dateFormatted])
			currLst.append(text)
			newDict[sender][dateFormatted] = tuple(currLst)
		else:
			newDict[sender][dateFormatted] = tuple([text])



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
def matchAliases(existingNames, otherNames):
	for otherName in otherNames:
		candidates = possMatches(otherName, existingNames) #list of possible matches (determined by small edit distance)
		topCandidate, bestScore = candidates[0]
		if candidates[1][1] == bestScore: #multiple best matches
			toCompare = [candidate[0][0]]
			for candidates in candidates:
				if candidate[1] == bestScore:
					writingStyleSimilarityDict[candidate[0]] = writingStyleMatchScore(otherName, candidate[0])
			topCandidate = sorted(writingStyleSimilarityDict.items(), key = lambda x: writingStyleSimilarityDict[x])[0]
		aliasDict[name] = topCandidate
		print(name, candidates)

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

def writingStyleMatchScore(otherName, possibleExistingMatch):
	"""
	existingNameText =  
	possMatchText = 
	"""
	





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

def fullWordList(existingName, sourceDict):
	fullWordList = []
	for messTup in sourceDict.values():
		i = 0
		if i>0:
			break
		i += 1
		for mess in messTup:
			words = mess.split(" ")
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

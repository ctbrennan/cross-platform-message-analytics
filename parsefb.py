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
"""
with open('messages.htm', "r") as f:
	chat = fb_parser.html_to_py(f)
	# Dump to json to prove works:
	fb_parser.py_to_json(chat)
"""
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
	#jsonFile = json.loads(open('parser/FB-Message-Parser/messages.json').read())
	jsonFile = json.loads(open('messages.json').read())
	global personDict, fullTextDict
	for person in jsonFile['threads']:
		for message in person['messages']:
			if message['sender'] not in personDict.keys():
				personDict[message['sender']] = {}
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
			#sender = message.attrib['name'] if message.attrib['name'] else me
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
		#print(newNames)
		#print(vCardDict.keys())
		#print(notFound)
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
		global personDict, fullTextDict
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
	vcfFile = open('allme.vcf', 'r')#need to fix later
	i = 0
	for line in vcfFile: #hacky, consider changing
		i += 1
		if line.startswith('VERSION'):
			if '3.' in line:
				parseVCF3()
			else:
				parseVCF2()


#=====================================================================================================
#                                       Combining Contacts
#=====================================================================================================
"""
add to alias dictionary a mapping from each name in otherNames to a name that's in existing names,
or adds a new name if no good match is found.
Step 1: compile list of possible matches by using minEditDistance
Step 2: Somone is less likely to be a match if the existing name already corresponds to many other names
Step 3: If there are a few possible matches, match using elements of writing style
"""
def matchAliases(existingNames, otherNames):
	for name in otherNames:
		candidates = possMatches(name, existingNames) #list of possible matches (determined by small edit distance)
		topCandidate = candidates[0]
		aliasDict[name] = topCandidate
		print(name, topCandidate)

def possMatches(name, existingNames, number=5):
	editDistances = {} #existing name -> min edit distance
	for existingName in existingNames:
		score = minEditDistance(name, existingName)
		editDistances[existingName] = score
	return sorted(editDistances.keys(), key = lambda x: editDistances[x])[:5] #?


def minEditDistance(w1, w2):
	def diff(a, b):
		if a == b:
			return 0
		return 1

	table = [[0 for _ in range(len(w2))] for _ in range(len(w1))]
	for i in range(0, len(w1)):
		table[i][0] = i
	for j in range(1, len(w2)):
		table[0][j] = j
	for i in range(1, len(w1)):
		for j in range(1, len(w2)):
			table[i][j] = min(table[i-1][j] + 1, table[i][j-1] + 1, table[i-1][j-1] + diff(w1[i], w2[j]))
	return table[len(w1) - 1][len(w2) - 1]



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

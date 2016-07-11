import sys
#sys.path.insert(0, '/home/connor/Documents/prj/parser/FB-Message-Parser/')
import fb_parser
from collections import OrderedDict
import json
from datetime import datetime
from wordcloud import WordCloud
import calendar
import os, os.path #parseAllThreads

"""
with open('messages.htm', "r") as f:
	chat = fb_parser.html_to_py(f)
	# Dump to json to prove works:
	fb_parser.py_to_json(chat)
"""
"""
import sqlite3
conn = sqlite3.connect('chat.db')
c = conn.cursor()
print(c.tables)
"""
import nltk


personDict = {}
fullTextDict = {}
vCardDict = {}
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
def parseAllThreads(me, folder):
	def parseThread(me, fileName):
		global personDict, fullTextDict
		if vCardDict == {}:
			parseVCF()
			print(vCardDict)
		jsonFile = json.loads(open(fileName).read())
		number = jsonFile['messages'][0]['handle_id']
		person = vCardDict[number]
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
	for root, _, files in os.walk(folder):
	    for f in files:
	        fullpath = os.path.join(root, f)
	        parseThread(me, fullpath)



def parseVCF():
	global vCardDict
	vcfFile = open('all.vcf', 'r')#need to fix later
	dictionary = {}
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
			"""
			if currName not in dictionary.keys():
				dictionary[currName] = tuple([number])
			else:
				currLst = list(dictionary[currName])
				currLst.append(number)
				dictionary[currName] = tuple(currLst)
			"""
			dictionary[number] = currName
	vCardDict = dictionary



def getAllMessagesAsString(personStr):
	string = ""
	for messages in personDict[personStr].values():
		for message in messages:
			string += message + " "
	return string

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
				#print(person)
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
				#print(person)
				i += 1
		else:
			return topFriends
			break

def numMessagesMonth(person, monthStart, monthEnd):
	# monthStart = datetime(int(year), int(month), 1)
	# monthEnd = datetime(int(year), int(month), calendar.monthrange(int(year),int(month))[1])
	count = 0
	for datetime in personDict[person]:
		if datetime >= monthStart and datetime <= monthEnd:
			count += 1
	return count



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
		#print(countList)
		plt.plot(countList, label = friend)
	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.show()


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

def minEditDistance(w1, w2):
	table = []
	for _ in range(len(w1)):
		table.append([])


"""
def randomInStyleOf(name):
	allMessages = getAllMessagesAsString(name)

#def parseSentenceAroundWord(message, word):
"""



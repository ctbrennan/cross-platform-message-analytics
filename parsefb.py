import sys
sys.path.insert(0, '/home/connor/Documents/prj/parser/FB-Message-Parser/')
import fb_parser
from collections import OrderedDict
import json
from datetime import datetime


"""
with open('messages.htm', "r") as f:
	chat = fb_parser.html_to_py(f)
	# Dump to json to prove works:
	fb_parser.py_to_json(chat)
"""
jsonFile = json.loads(open('parser/FB-Message-Parser/messages.json').read())
personDict = {}
for person in jsonFile['threads']:
	for message in person['messages']:
		if message['sender'] not in personDict.keys():
			personDict[message['sender']] = {}
		dateFormatted = datetime.strptime(message['date_time'], '%A, %B %d, %Y at %I:%M%p ')
		if dateFormatted in personDict[message['sender']].keys():
			tup = personDict[message['sender']][dateFormatted].append(message['text'])
		else:
			personDict[message['sender']][dateFormatted] = [message['text']]
#orderedPersonDict = {}
for sender in personDict.keys():
	personDict[sender] = OrderedDict(sorted(personDict[sender].items(), key=lambda t: t[0]))



def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[str(name[:-1])] = str(x)

    flatten(y)
    return out

#r = json.load(open('messages.json'), object_pairs_hook=OrderedDict)
#print json.dumps(r, indent=2)
"""
from pandas.io.json import json_normalize
json_normalize(r)
"""
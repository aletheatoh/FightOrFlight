import json

with open('new.json', 'r') as outfile:
    data = json.load(outfile)

for dp in data:
    print(dp)

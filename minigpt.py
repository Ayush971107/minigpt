import json

with open('gutenberg-poetry-v001.ndjson', encoding='utf-8') as fin, \
     open('poems.txt', 'w', encoding='utf-8') as fout:
    for line in fin:
        poem = json.loads(line).get('s', '')
        fout.write(poem + '\n')
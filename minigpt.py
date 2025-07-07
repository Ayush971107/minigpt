import json

with open('gutenberg-poetry-v001.ndjson', encoding='utf-8') as fin, \
     open('poems.txt', 'w', encoding='utf-8') as fout:
    for line in fin:
        poem = json.loads(line).get('s', '')
        fout.write(poem + '\n')

with open('poems.txt', 'r', encoding = 'utf-8') as f:
  text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

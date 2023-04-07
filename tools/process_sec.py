import json

with open('thrid.json','r') as f:
    js = json.load(f)
    
with open('fails.json','r') as f:
    fails = json.load(f)['all']

df = {'1':[], '2':[], "3":[]}

for fail in fails:
    img = fail['img']
    df[fail['img']].append(fail['nums'])
train = js['training']
print(len(train))

for jso in train:
    num = jso['label'].split('/')[1]
    x = jso['label'].split('/')[-1].split('_')[-1]
    if x in df[num]:
        train.remove(jso)
        
print(len(train))
js['training'] = train

with open('fourth.json', 'w') as f:
    json.dump(js, f)
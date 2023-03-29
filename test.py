import json

s = r'{"C": "\u3042", "A": {"i": 1, "j": 2}, "B": [{"X": 1, "Y": 10}, {"X": 2, "Y": 20}]}'

print(s)
# {"C": "\u3042", "A": {"i": 1, "j": 2}, "B": [{"X": 1, "Y": 10}, {"X": 2, "Y": 20}]}

d = json.loads(s)

print(type(d))

sd = json.dumps(d)

print(sd)
# {"A": {"i": 1, "j": 2}, "B": [{"X": 1, "Y": 10}, {"X": 2, "Y": 20}], "C": "\u3042"}

print(type(sd))
# <class 'str'>

with open('json_data/test1.json', 'w') as f:
    json.dump(d, f)

with open('json_data/test1.json') as f:
    df = json.load(f)
    print(type(df))
import json
def look_at_file():
    with open('my_runs/22/metrics.json') as f:
        data = json.load(f)
    print(data)
    pass
look_at_file()
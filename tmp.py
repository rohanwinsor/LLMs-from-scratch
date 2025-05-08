import json

for i in range(10):
    with open(f"{i}.json", "w") as f:
        json.dump(list(range(10)), f, indent=4)
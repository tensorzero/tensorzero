import json


def load_beerqa():
    with open("beerqa_train_subsample.json", "r") as f:
        data = json.load(f)
    return data["data"]

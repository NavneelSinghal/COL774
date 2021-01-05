import sys
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np


with open(sys.argv[1], "r") as f:
    gold = f.readlines()
    gold = [int(float(each.strip())) for each in gold]

status = False
try:
    with open(sys.argv[2], "r") as f:
        pred = f.readlines()
        try:
            pred = [int(float(each.strip())) for each in pred]
            status = True
        except Exception as e:
            print("Error in reading prediction file: ", e)
except:
    print("Prediction file not found")

with open(sys.argv[3], 'w') as fp:
    if status:
        try:
            acc = 100*accuracy_score(gold, pred)
            print(f"Accuracy (%): {acc}")
        except Exception as e:
            print("Error in calculating accuracy: ", e)
            acc = 0.0
    else:
        acc = 0.0
        print("Prediction file not found")
        print(f"Accuracy (%): {acc}")
    acc = "{:02f}".format(acc)
    fp.write(acc)

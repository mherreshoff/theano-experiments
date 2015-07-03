#!/usr/local/bin/python
import sys
import matplotlib.pyplot as plt 

f = open(sys.argv[1], "r")
ts = []
trains = []
tests = []

for line in f:
  t, train, test = [float(x) for x in line.strip().split("\t")]
  if train < 0.1:
    ts.append(t)
    trains.append(train)
    tests.append(test)

plt.xlabel("Iteration")
plt.ylabel("Error Rates")
plt.plot(ts, trains, color='r', label="train")
plt.plot(ts, tests, color='b', label="test")
plt.legend()
plt.show()

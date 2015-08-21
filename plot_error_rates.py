#!/usr/bin/python
import sys
import matplotlib.pyplot as plt 

def read_graph(dir_name):
  f = open(dir_name + "/graph.tsv", "r")
  ts = []
  trains = []
  tests = []

  for line in f:
    t, train, test = [float(x) for x in line.strip().split("\t")]
    if t >= 5:
      ts.append(t)
      trains.append(train)
      tests.append(test)
  return (dir_name, ts, trains, tests)

graphs = [read_graph(dir_name) for dir_name in sys.argv[1:]]
plt.xlabel("Iteration")
plt.ylabel("Error Rates")
for dir_name, ts, trains, tests in graphs:
  plt.plot(ts, trains, label=dir_name+":train")
  plt.plot(ts, tests, label=dir_name+":test")
plt.legend()
plt.show()

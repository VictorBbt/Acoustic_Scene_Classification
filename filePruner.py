import os, sys, random

#file that prunes a given proportion of files in a directory
dir = sys.argv[1]


prop = 0.33

ls = os.listdir(dir)
n = len(ls)
# n_sup = int(prop*n)

random.shuffle(ls)
# ls_sup = ls[:n_sup]

n_objective = 800

to_prune = n - n_objective
ls_sup = ls[:to_prune]

for file in ls_sup:
    path = os.path.join(dir,file)
    os.remove(path)

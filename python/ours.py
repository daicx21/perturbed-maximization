# Parse the arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help = "name of the dataset")
parser.add_argument("maxprob", help = "Q", type = float)
parser.add_argument("beta", help = "β", type = float)
args = parser.parse_args()

# Set up the dataset being tested and the algorithm
dataset   = args.dataset
maxprob   = args.maxprob
beta      = args.beta

# Import the relavant codes
import algorithms
import classes
import metrics
import numpy as np
from collections import defaultdict

# Read the dataset and calculate the maximum possible quality
instance = classes.InputInstance(dataset, init=True)

# Run the algorithm
assignment = algorithms.ours(instance, beta=beta, maxprob=maxprob)

pquality = 0
for i in range(instance.np):
	for j in range(instance.nr):
		pquality += (assignment[i][j] - assignment[i][j] * assignment[i][j] * beta) * instance.s[i][j]
print(f'Quality: {metrics.quality(instance, assignment):.2f}', '/', f'{instance.max_quality:.2f}')
print(f'PQuality: {pquality:.2f}')
for i in range(5):
	print(f'{metrics.name(i, style = 2)}: {metrics.calc(instance, assignment, i):.2f}')
print()

with open('results/output.out', 'w') as file:
	print(instance.nr, instance.np, file=file)
	for i in range(instance.nr):
		print(1, file=file)
	for i in range(instance.nr):
		for j in range(instance.np):
			if assignment[j][i] > (1e-6):
				print(i, j + instance.nr, round(assignment[j][i], 6), file=file)

with open('results/bidauthorship.out', 'w') as file:
	for i in range(instance.nr):
		for j in instance.bidauthorlist[i]:
			if j > i and instance.bidauthorship[j][i]:
				print(i, j, file=file)

with open('results/coauthorship.out', 'w') as file:
	for i in range(instance.nr):
		for j in instance.coauthorlist[i]:
			if j > i:
				print(i, j, file=file)

sum_coauthors = 0
with open('results/coauthorvio.out', 'w') as file:
	for i in range(instance.nr):
		for j in instance.coauthorlist[i]:
			if j <= i:
				continue
			for k in range(instance.np):
				sum_coauthors += assignment[k][i] * assignment[k][j]
				if assignment[k][i] > (1e-6) and assignment[k][j] > (1e-6):
					print(i, k, file=file)
					print(j, k, file=file)
print('sum_coauthors_prob:', sum_coauthors)

sum_cocoauthors = 0
for i in range(instance.nr):
	vis = defaultdict(bool)
	for j in instance.coauthorlist[i]:
		for k in instance.coauthorlist[j]:
			if k <= i or vis[k]:
				continue
			vis[k] = True
			for p in range(instance.np):
				sum_cocoauthors += assignment[p][i] * assignment[p][k]
print('sum_cocoauthors_prob:', sum_cocoauthors)

sum_2cycles = 0.0
for i in range(instance.nr):
	for j in instance.bidauthorlist[i]:
		if j > i and instance.bidauthorship[j][i]:
			for k in instance.paperlist[i]:
				for l in instance.paperlist[j]:
					if instance.bid[l][i] and instance.bid[k][j]:
						sum_2cycles += assignment[l][i] * assignment[k][j]
print('sum_2cycles_prob:', sum_2cycles)

import os
os.system('cpp/bvn < results/output.out > results/output_bvn.out')

assignment_final = [[False for _ in range(instance.nr)] for _ in range(instance.np)]
with open('results/output_bvn.out', 'r') as file:
	for line in file:
		result = line.strip().split()
		assignment_final[int(result[1]) - instance.nr][int(result[0])] = True

for i in range(instance.np):
	if assignment_final[i].count(True) != instance.ellp:
		print(i, 'no!!!')

sum_coauthors = 0
for i in range(instance.nr):
	for j in instance.coauthorlist[i]:
		if j <= i:
			continue
		for k in range(instance.np):
			if assignment_final[k][i] == True and assignment_final[k][j] == True:
				sum_coauthors += 1
print('sum_coauthors:', sum_coauthors)

sum_cocoauthors = 0
for i in range(instance.nr):
	vis = defaultdict(bool)
	for j in instance.coauthorlist[i]:
		for k in instance.coauthorlist[j]:
			if k <= i or vis[k]:
				continue
			vis[k] = True
			for p in range(instance.np):
				sum_cocoauthors += int(assignment_final[p][i]) * int(assignment_final[p][k])
print('sum_cocoauthors:', sum_cocoauthors)

sum_2cycles = 0
for i in range(instance.nr):
	for j in instance.bidauthorlist[i]:
		if j > i and instance.bidauthorship[j][i]:
			for k in instance.paperlist[i]:
				for l in instance.paperlist[j]:
					if instance.bid[l][i] and instance.bid[k][j]:
						sum_2cycles += int(assignment_final[l][i]) * int(assignment_final[k][j])
print('sum_2cycles:', sum_2cycles)
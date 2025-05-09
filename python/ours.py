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

list_nonempty_p = [[] for _ in range(instance.np)]
list_nonempty_r = [[] for _ in range(instance.nr)]

with open('results/output.out', 'w') as file:
	print(instance.nr, instance.np, file=file)
	for i in range(instance.nr):
		print(1, file=file)
	for j in range(instance.np):
		list_nonempty = []
		sum = 0
		for i in range(instance.nr):
			if assignment[j][i] > (1e-8):
				assignment[j][i] = round(assignment[j][i], 7)
				list_nonempty.append([j, i, assignment[j][i]])
				sum += assignment[j][i]
				list_nonempty_p[j].append(i)
				list_nonempty_r[i].append(j)
		if (abs(sum - round(sum)) > (1e-8)):
			if sum - round(sum) > (1e-8):
				sum = sum - round(sum)
				for j in range(len(list_nonempty)):
					now = min(sum, list_nonempty[j][2])
					assignment[list_nonempty[j][0]][list_nonempty[j][1]] -= now
					sum -= now
			else:
				sum = round(sum) - sum
				for j in range(len(list_nonempty)):
					now = min(sum, 1 - list_nonempty[j][2])
					assignment[list_nonempty[j][0]][list_nonempty[j][1]] += now
					sum -= now

	for i in range(instance.nr):
		for j in list_nonempty_r[i]:
			print(i, j + instance.nr, "{:.7f}".format(assignment[j][i]), file=file)

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
sum_cocoauthors = 0
with open('results/coauthorvio.out', 'w') as file:
	for p in range(instance.np):
		for i in list_nonempty_p[p]:
			vis = defaultdict(bool)
			for j in instance.coauthorlist[i]:
				if j > i:
					sum_coauthors += assignment[p][i] * assignment[p][j]
					if assignment[p][i] > (1e-8) and assignment[p][j] > (1e-8):
						print(i, p, 0, file=file)
						print(j, p, 0, file=file)
				for k in instance.coauthorlist[j]:
					if k <= i or instance.coauthorship[i][k] or vis[k]:
						continue
					vis[k] = True
					sum_cocoauthors += assignment[p][i] * assignment[p][k]
					if assignment[p][i] > (1e-8) and assignment[p][k] > (1e-8):
						print(i, p, 1, file=file)
						print(k, p, 1, file=file)
					
print('sum_coauthors_prob:', sum_coauthors)
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
sum_cocoauthors = 0
for p in range(instance.np):
	for i in list_nonempty_p[p]:
		vis = defaultdict(bool)
		for j in instance.coauthorlist[i]:
			if j > i:
				sum_coauthors += int(assignment_final[p][i]) * int(assignment_final[p][j])
			for k in instance.coauthorlist[j]:
				if k <= i or instance.coauthorship[i][k] or vis[k]:
					continue
				vis[k] = True
				sum_cocoauthors += int(assignment_final[p][i]) * int(assignment_final[p][k])
print('sum_coauthors:', sum_coauthors)
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
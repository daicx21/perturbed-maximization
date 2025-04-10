# Parse the arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help = "name of the dataset")
parser.add_argument("maxprob", help = "Q", type = float)
parser.add_argument("beta", help = "β", type = float)
parser.add_argument("alpha", help = "α", type = float)
args = parser.parse_args()

# Set up the dataset being tested and the algorithm
dataset   = args.dataset
maxprob   = args.maxprob
beta      = args.beta
alpha     = args.alpha

# Import the relavant codes
import algorithms
import classes
import metrics
import numpy as np

# Read the dataset and calculate the maximum possible quality
instance = classes.InputInstance(dataset, init=True)

# Run the algorithm
assignment = algorithms.PMQ(instance, beta=beta, maxprob=maxprob)

# Output statistics
pquality = 0
for i in range(instance.np):
	for j in range(instance.nr):
		pquality += (assignment[i][j] - assignment[i][j] * assignment[i][j] * beta) * instance.s[i][j]
print(f'Quality: {metrics.quality(instance, assignment):.2f}', '/', f'{instance.max_quality:.2f}')
print(f'PQuality: {pquality:.2f}')
for i in range(5):
	print(f'{metrics.name(i, style = 2)}: {metrics.calc(instance, assignment, i):.2f}')
print()

if dataset.lower() == 'aamas2021' or dataset.lower() == 'iclr2018':
	sum = 0
	for i in range(instance.nr):
		for j in range(i + 1, instance.nr):
			if instance.coauthorship[i][j] == False:
				continue
			for k in range(instance.np):
				sum += assignment[k][i] * assignment[k][j]
				if assignment[k][i] > (1e-6) and assignment[k][j] > (1e-6):
					print(i, j, k, assignment[k][i], assignment[k][j])
	print('sum_coauthor_prob:', sum, '!!!')
	
	with open('results/output.out', 'w') as file:
		print(instance.nr, instance.np, file=file)
		for i in range(instance.nr):
			print(1, file=file)
		for i in range(instance.nr):
			for j in range(instance.np):
				print(i, j + instance.nr, '{:.15f}'.format(assignment[j][i]), file=file)

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

	sum = 0
	for i in range(instance.nr):
		for j in range(i + 1, instance.nr):
			if instance.coauthorship[i][j] == False:
				continue
			for k in range(instance.np):
				if assignment_final[k][i] == True and assignment_final[k][j] == True:
					sum += 1
					print(i, j, k, assignment[k][i], assignment[k][j], 'final')
	print('sum_coauthor:', sum, '!!!')

	instance.deleted = [[False for _ in range(instance.nr)] for _ in range(instance.np)]

	cnt1 = cnt2 = 0

	for i in range(instance.np):
		r_list = sorted([j for j in range(instance.nr)], key=lambda x: instance.s[i][x])
		vis = [False for _ in range(instance.nr)]
		for j in reversed(r_list):
			vis[j] = True
			instance.deleted[i][j] = False
			for k in instance.coauthorlist[j]:
				if vis[k] and not instance.deleted[i][k]:
					instance.deleted[i][j] = True
					break
			if instance.deleted[i][j]:
				cnt1 += 1
			else:
				cnt2 += 1

	# print(cnt1, cnt2)

	# assignment = algorithms.PMQ_second(instance, beta=beta, maxprob=maxprob)

	# pquality = 0
	# for i in range(instance.np):
	# 	for j in range(instance.nr):
	# 		pquality += (assignment[i][j] - assignment[i][j] * assignment[i][j] * beta) * instance.s[i][j]
	# print(f'Quality: {metrics.quality(instance, assignment):.2f}', '/', f'{instance.max_quality:.2f}')
	# print(f'PQuality: {pquality:.2f}')
	# for i in range(5):
	# 	print(f'{metrics.name(i, style = 2)}: {metrics.calc(instance, assignment, i):.2f}')
	# print()
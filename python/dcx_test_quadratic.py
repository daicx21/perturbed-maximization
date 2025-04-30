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
print(666)
# Read the dataset and calculate the maximum possible quality
instance = classes.InputInstance(dataset, init=True)
print(777)
# Run the algorithm
assignment = algorithms.PMQ(instance, beta=beta, maxprob=maxprob)
print(888)

if dataset.lower() == 'testlarge':
	# Output statistics
	pquality = 0
	for i in range(instance.np):
		for j in instance.biddedlist[i]:
			pquality += (assignment[i][j] - assignment[i][j] * assignment[i][j] * beta) * instance.s[i][j]
	print(f'Quality: {metrics.quality(instance, assignment):.2f}', '/', f'{instance.max_quality:.2f}')
	print(f'PQuality: {pquality:.2f}')
	for i in range(5):
		print(f'{metrics.name(i, style = 2)}: {metrics.calc(instance, assignment, i):.2f}')
	print()
else:
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


if dataset.lower() == 'testlarge':
	sum_coauthors = 0
	for i in range(instance.nr):
		for j in instance.coauthorlist[i]:
			if j <= i:
				continue
			for k in instance.bidlist[i]:
				if instance.bid[k][j]:
					sum_coauthors += assignment[k][i] * assignment[k][j]
	print('sum_coauthors_prob:', sum_coauthors)
	
	sum_cocoauthors = 0
	for i in range(instance.nr):
		vis = defaultdict(bool)
		for j in instance.coauthorlist[i]:
			for k in instance.coauthorlist[j]:
				if k <= i or vis[k]:
					continue
				vis[k] = True
				for p in instance.bidlist[i]:
					if instance.bid[p][k]:
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

	with open('results/output.out', 'w') as file:
		print(instance.nr, instance.np, file=file)
		for i in range(instance.nr):
			print(1, file=file)
		for i in range(instance.nr):
			for j in instance.bidlist[i]:
				print(i, j + instance.nr, '{:.15f}'.format(assignment[j][i]), file=file)

	import os
	os.system('cpp/bvn < results/output.out > results/output_bvn.out')

	assignment_final = [{} for _ in range(instance.np)]
	for i in range(instance.np):
		for j in instance.biddedlist[i]:
			assignment_final[i][j] = False

	with open('results/output_bvn.out', 'r') as file:
		for line in file:
			result = line.strip().split()
			assignment_final[int(result[1]) - instance.nr][int(result[0])] = True

	for i in range(instance.np):
		counter = 0
		for j in assignment_final[i].keys():
			if assignment_final[i][j] == True:
				counter += 1
		if counter != instance.ellp:
			print(i, 'no!!!')

	sum_coauthors = 0
	for i in range(instance.nr):
		for j in instance.coauthorlist[i]:
			if j < i:
				continue
			for k in instance.bidlist[i]:
				if assignment_final[k].get(i) == True and assignment_final[k].get(j) == True:
					sum_coauthors += 1
	print('sum_coauthors:', sum_coauthors)

	sum_cocoauthors = 0
	for i in range(instance.nr):
		vis = [False for _ in range(instance.nr)]
		for j in instance.coauthorlist[i]:
			for k in instance.coauthorlist[j]:
				if k <= i or vis[k]:
					continue
				vis[k] = True
				for p in instance.bidlist[i]:
					if assignment_final[p].get(i) == True and assignment_final[p].get(k) == True:
						sum_cocoauthors += 1
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
	print()


	instance.deleted = [defaultdict(bool) for _ in range(instance.np)]
	for i in range(instance.np):
		r_list = sorted(instance.biddedlist[i], key=lambda x: assignment[i][x])
		vis = defaultdict(bool)
		for j in reversed(r_list):
			vis[j] = True
			for k in instance.coauthorlist[j]:
				if instance.bid[i][k] and vis[k] and not instance.deleted[i][k]:
					instance.deleted[i][j] = True
					break

	map_2cycles = defaultdict(list)
	vis_pr_list = defaultdict(bool)
	pr_list = []
	for i in range(instance.nr):
		for j in instance.bidauthorlist[i]:
			if j > i and instance.bidauthorship[j][i]:
				for k in instance.paperlist[i]:
					for l in instance.paperlist[j]:
						if instance.bid[l][i] and instance.bid[k][j]:
							map_2cycles[(l, i)].append((k, j))
							map_2cycles[(k, j)].append((l, i))
							if not vis_pr_list[(l, i)]:
								vis_pr_list[(l, i)] = True
								pr_list.append((l, i))
							if not vis_pr_list[(k, j)]:
								vis_pr_list[(k, j)] = True
								pr_list.append((k, j))

	print('len:', len(pr_list))

	pr_list = sorted(pr_list, key=lambda x: assignment[x[0]][x[1]])
	vis_pr_list = defaultdict(bool)
	for tup1 in reversed(pr_list):
		vis_pr_list[tup1] = True
		for tup2 in map_2cycles[tup1]:
			if vis_pr_list[tup2] and not instance.deleted[tup2[0]][tup2[1]]:
				instance.deleted[tup1[0]][tup1[1]] = True
				break

	cnt1 = cnt2 = 0
	for i in range(instance.np):
		for j in instance.biddedlist[i]:
			if instance.deleted[i][j]:
				cnt1 += 1
			else:
				cnt2 += 1
	print('num_deleted: ', cnt1, ' num_non_deleted: ', cnt2)
	print()

else:
	sum_coauthors = 0
	for i in range(instance.nr):
		for j in range(i + 1, instance.nr):
			if instance.coauthorship[i][j] == False:
				continue
			for k in range(instance.np):
				sum_coauthors += assignment[k][i] * assignment[k][j]
	print('sum_coauthors_prob:', sum_coauthors)
	
	sum_cocoauthors = 0
	for i in range(instance.nr):
		vis = [False for _ in range(instance.nr)]
		for j in range(instance.nr):
			if instance.coauthorship[i][j] == False:
				continue
			for k in range(instance.nr):
				if k <= i or vis[k] or instance.coauthorship[j][k] == False:
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

	sum_coauthors = 0
	for i in range(instance.nr):
		for j in range(i + 1, instance.nr):
			if instance.coauthorship[i][j] == False:
				continue
			for k in range(instance.np):
				if assignment_final[k][i] == True and assignment_final[k][j] == True:
					sum_coauthors += 1
	print('sum_coauthors:', sum_coauthors)

	sum_cocoauthors = 0
	for i in range(instance.nr):
		vis = [False for _ in range(instance.nr)]
		for j in range(instance.nr):
			if instance.coauthorship[i][j] == False:
				continue
			for k in range(instance.nr):
				if k <= i or vis[k] or instance.coauthorship[j][k] == False:
					continue
				vis[k] = True
				for p in range(instance.np):
					if assignment_final[p][i] == True and assignment_final[p][k] == True:
						sum_cocoauthors += 1
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
	print()

	instance.deleted = [[False for _ in range(instance.nr)] for _ in range(instance.np)]
	for i in range(instance.np):
		r_list = sorted([j for j in range(instance.nr)], key=lambda x: assignment[i][x])
		vis = [False for _ in range(instance.nr)]
		for j in reversed(r_list):
			vis[j] = True
			for k in instance.coauthorlist[j]:
				if vis[k] and not instance.deleted[i][k]:
					instance.deleted[i][j] = True
					break

	from collections import defaultdict
	map_2cycles = defaultdict(list)
	vis_pr_list = defaultdict(bool)
	pr_list = []
	for i in range(instance.nr):
		for j in instance.bidauthorlist[i]:
			if j > i and instance.bidauthorship[j][i]:
				for k in instance.paperlist[i]:
					for l in instance.paperlist[j]:
						if instance.bid[l][i] and instance.bid[k][j]:
							map_2cycles[(l, i)].append((k, j))
							map_2cycles[(k, j)].append((l, i))
							if not vis_pr_list[(l, i)]:
								vis_pr_list[(l, i)] = True
								pr_list.append((l, i))
							if not vis_pr_list[(k, j)]:
								vis_pr_list[(k, j)] = True
								pr_list.append((k, j))

	pr_list = sorted(pr_list, key=lambda x: assignment[x[0]][x[1]])
	vis_pr_list = defaultdict(bool)
	for tup1 in reversed(pr_list):
		vis_pr_list[tup1] = True
		for tup2 in map_2cycles[tup1]:
			if vis_pr_list[tup2] and not instance.deleted[tup2[0]][tup2[1]]:
				instance.deleted[tup1[0]][tup1[1]] = True
				break

	cnt1 = cnt2 = 0
	for i in range(instance.np):
		for j in range(instance.nr):
			if instance.deleted[i][j]:
				cnt1 += 1
			else:
				cnt2 += 1
	print('num_deleted: ', cnt1, ' num_non_deleted: ', cnt2)
	print()
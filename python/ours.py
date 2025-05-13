# Parse the arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="name of the dataset")
parser.add_argument("maxprob", help="Q", type=float)
parser.add_argument("beta", help="β", type=float)
parser.add_argument("reward_region", help="reward_region", type=float)
parser.add_argument("target_sen", help="target_seniority", type=int)
parser.add_argument("reward_sen", help="reward_seniority", type=float)
parser.add_argument("pen_coauthor", help="pen_coauthor", type=float)
parser.add_argument("pen_2cycle", help="pen_2cycle", type=float)
args = parser.parse_args()

# Set up the dataset being tested and the algorithm
dataset   = args.dataset
maxprob   = args.maxprob
beta      = args.beta

reward_region = max(args.reward_region, 1e-3)
target_sen    = args.target_sen
reward_sen    = max(args.reward_sen, 1e-3)
pen_coauthor  = max(args.pen_coauthor, 1e-3)
pen_2cycle    = max(args.pen_2cycle, 1e-3)

# Import the relavant codes
import algorithms
import classes
import metrics
import numpy as np
from collections import defaultdict
import os
import time

datasets = [dataset]

for dataset in datasets:
	for test_flag in range(4, 5):
		start_time = time.time()

		# Read the dataset and calculate the maximum possible quality
		instance = classes.InputInstance(dataset, init=True)

		if test_flag == 0:
			algo_name = 'Ours'
		elif test_flag == 1:
			algo_name = 'Perturbed_Maximization'
		elif test_flag == 2:
			algo_name = 'Randomized'
		elif test_flag == 3:
			algo_name = 'Default'
		else:
			algo_name = 'Ours_minus_piecewise_linear'

		dir_name = 'test_results_simulated/{}'.format(algo_name)
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)

		# Run the algorithm
		if algo_name == 'Ours':
			assignment = algorithms.ours(instance,
										beta=beta,
										maxprob=maxprob,
										reward_region=reward_region,
										target_sen=target_sen,
										reward_sen=reward_sen,
										pen_coauthor=pen_coauthor,
										pen_2cycle=pen_2cycle)
		elif algo_name == 'Perturbed_Maximization':
			assignment = algorithms.PMQ(instance, beta=beta, maxprob=maxprob)
		elif algo_name == 'Randomized':
			assignment = algorithms.PLRA(instance, maxprob=maxprob)
		elif algo_name == 'Default':
			assignment = algorithms.PLRA(instance, maxprob=1)
		else:
			assignment = algorithms.ours_minus_piecewise_linear(instance,
																beta=beta,
																maxprob=maxprob,
																reward_region=reward_region,
																target_sen=target_sen,
																reward_sen=reward_sen,
																pen_coauthor=pen_coauthor,
																pen_2cycle=pen_2cycle)

		list_nonempty_p = [[] for _ in range(instance.np)]
		list_nonempty_r = [[] for _ in range(instance.nr)]

		with open('results/output.out', 'w') as file:
			print(instance.nr, instance.np, file=file)
			for i in range(instance.nr):
				print(instance.region[i] + 1, int(instance.seniority[i] >= 2) + 1, file=file)
			for j in range(instance.np):
				list_nonempty = []
				sum = 0
				for i in instance.remained_r_for_p[j]:
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

		if algo_name.startswith('Ours'):
			os.system('cpp/bvn < results/output.out > results/output_bvn.out')
		else:
			os.system('cpp/bvn1 < results/output.out > results/output_bvn.out')

		assignment_final = [defaultdict(bool) for _ in range(instance.np)]
		assignment_list = [[] for _ in range(instance.np)]
		with open('results/output_bvn.out', 'r') as file:
			for line in file:
				result = line.strip().split()
				assignment_final[int(result[1]) - instance.nr][int(result[0])] = True
				assignment_list[int(result[1]) - instance.nr].append(int(result[0]))

		with open('{}/results.txt'.format(dir_name), 'w') as file:
			end_time = time.time()
			print('running time: {:.2f} seconds'.format(end_time - start_time), file=file)
			sum = 0
			for i in range(instance.np):
				for j in instance.remained_r_for_p[i]:
					sum += instance.s[i][j] * int(assignment_final[i][j])
			print('Max Quality: {:.4f}'.format(instance.max_quality), file=file)
			print('Our Quality: {:.4f}'.format(sum), file=file)
			print('Quality: {:.4f}'.format(sum / instance.max_quality), file=file)
			for i in range(5):
				print(f'{metrics.name(i, style = 2)}: {metrics.calc(instance, assignment, i):.2f}', file=file)
			
			sum_coauthors = 0
			sum_cocoauthors = 0
			for p in range(instance.np):
				for i in assignment_list[p]:
					vis = defaultdict(bool)
					for j in instance.coauthorlist[i]:
						if j > i:
							sum_coauthors += int(assignment_final[p][i]) * int(assignment_final[p][j])
						for k in instance.coauthorlist[j]:
							if k <= i or instance.coauthorship[i][k] or vis[k]:
								continue
							vis[k] = True
							sum_cocoauthors += int(assignment_final[p][i]) * int(assignment_final[p][k])
			print('sum_coauthors:', sum_coauthors, file=file)
			print('sum_cocoauthors:', sum_cocoauthors, file=file)

			sum_2cycles = 0
			for i in range(instance.nr):
				for j in instance.bidauthorlist[i]:
					if j > i and instance.bidauthorship[j][i]:
						for k in instance.paperlist[i]:
							for l in instance.paperlist[j]:
								if instance.bid[l][i] and instance.bid[k][j]:
									sum_2cycles += int(assignment_final[l][i]) * int(assignment_final[k][j])
			print('sum_2cycles:', sum_2cycles, file=file)

			cnt_1 = 0
			cnt_2 = 0
			cnt_3 = 0
			cnt_4 = 0
			for i in range(instance.np):
				bo = [False for _ in range(5)]
				for j in assignment_list[i]:
					bo[instance.region[j]] = True
				sum = 0
				for j in range(5):
					sum += int(bo[j])
				if sum == 1:
					cnt_1 += 1
				if sum == 2:
					cnt_2 += 1
				if sum == 3:
					cnt_3 += 1
				if sum == 4:
					cnt_4 += 1
			print('#papers_such_that_reviewers_come_from_1_region:', cnt_1, file=file)
			print('#papers_such_that_reviewers_come_from_2_regions:', cnt_2, file=file)
			print('#papers_such_that_reviewers_come_from_3_regions:', cnt_3, file=file)
			print('#papers_such_that_reviewers_come_from_4_regions:', cnt_4, file=file)

			cnt_0 = 0
			cnt_1 = 0
			cnt_2 = 0
			cnt_3 = 0
			for i in range(instance.np):
				mx = 0
				for j in assignment_list[i]:
					mx = max(mx, instance.seniority[j])
				if mx == 0:
					cnt_0 += 1
				if mx == 1:
					cnt_1 += 1
				if mx == 2:
					cnt_2 += 1
				if mx == 3:
					cnt_3 += 1
			print('#papers_with_max_reviewer_seniority_0:', cnt_0, file=file)
			print('#papers_with_max_reviewer_seniority_1:', cnt_1, file=file)
			print('#papers_with_max_reviewer_seniority_2:', cnt_2, file=file)
			print('#papers_with_max_reviewer_seniority_3:', cnt_3, file=file)

		import matplotlib.pyplot as plt

		data_region = []
		for i in range(instance.np):
			bo = [False for _ in range(5)]
			for j in assignment_list[i]:
				bo[instance.region[j]] = True
			sum = 0
			for j in range(5):
				sum += int(bo[j])
			data_region.append(sum)

		plt.hist(data_region, bins=range(1, 5+1), align='left', rwidth=0.8)

		plt.xticks(range(1, 5))
		plt.xlabel('Different reviewer regions for a paper')
		plt.ylabel('Frequency')
		plt.title('Geographic Diversity')

		plt.savefig('{}/geographic_diversity.png'.format(dir_name))
		plt.clf()

		data_sen = []
		for i in range(instance.np):
			mx = 0
			for j in assignment_list[i]:
				mx = max(mx, instance.seniority[j])
			data_sen.append(mx)

		plt.hist(data_sen, bins=range(0, 4+1), align='left', rwidth=0.8)

		plt.xticks(range(0, 4))
		plt.xlabel('Maximum reviwer seniority for a paper')
		plt.ylabel('Frequency')
		plt.title('Reviewer Seniority')

		plt.savefig('{}/seniority.png'.format(dir_name))
		plt.clf()
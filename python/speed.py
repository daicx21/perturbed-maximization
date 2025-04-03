# Parse the arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help = "name of the dataset")
parser.add_argument("algorithm", help = "name of the algorithm")
parser.add_argument("maxprob", help = "Q", type = float)
parser.add_argument("beta", help = "β", type = float)
parser.add_argument("alpha", help = "α", type = float)
parser.add_argument("offset", help = "Increment in Q", type = float)
args = parser.parse_args()

# Set up the dataset being tested and the algorithm
dataset   = args.dataset
algorithm = args.algorithm
maxprob   = args.maxprob
beta      = args.beta
alpha     = args.alpha
offset    = args.offset

# Import the relavant codes
import algorithms
import classes
import metrics
import numpy as np

# Read the dataset and calculate the maximum possible quality
instance = classes.InputInstance(dataset, init=True)

# Run the algorithm
if (algorithm == 'PLRA'):
	assignment = algorithms.PLRA(instance, maxprob = maxprob)
elif (algorithm == 'PM-Q'):
	assignment = algorithms.PMQ(instance, beta = beta, maxprob = maxprob + offset)
elif (algorithm == 'PM-E'):
	assignment = algorithms.PME(instance, alpha = alpha, maxprob = maxprob + offset)
else:
	print("Algorithm name is incorrect.")

# Output statistics
pquality = 0
for i in range(instance.np):
	for j in range(instance.nr):
		if (algorithm == 'PM-Q'):
			pquality += (assignment[i][j] - assignment[i][j] * assignment[i][j] * beta) * instance.s[i][j]
		if (algorithm == 'PM-E'):
			pquality += (1 - np.exp(-alpha * assignment[i][j])) * instance.s[i][j]
print(f'Quality: {metrics.quality(instance, assignment):.2f}', '/', f'{instance.max_quality:.2f}')
print(f'PQuality: {pquality:.2f}')
for i in range(5):
	print(f'{metrics.name(i, style = 2)}: {metrics.calc(instance, assignment, i):.2f}')
 
if dataset.lower() == 'aamas2021' or dataset.lower() == 'iclr2018':
    sum = 0
    for i in range(instance.nr):
        for j in range(i + 1, instance.nr):
            if instance.coauthorship[i][j] == False:
                continue
            for k in range(instance.np):
                sum += assignment[k][i] * assignment[k][j]
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
    print('sum_coauthor:', sum, '!!!')
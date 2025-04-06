import csv
import numpy as np

class InputInstance:
	# Input instance of randomized assignment algorithms
	# Members:
	# 	np:   number of papers
	#   nr:   number of reviewers
	# 	ellp: number of reviewers required per paper
	#   ellr: maximum number of papers per reviewer
	#   s:    similarity matrix, index: [paper][reviewer]
	#   max_quality: maximum possible quality
	def __init__(self, dataset, init = True):
		self.dataset = dataset
		if dataset.lower() == 'aamas2021':
			data = np.load('datasets/aamas2021/aamas_text.npy')
			(self.np, self.nr) = data.shape
			self.s = []
			for row in data:
				self.s.append(row.tolist())
			self.ellp = 4
			self.ellr = 6

			data = np.load('datasets/aamas2021/aamas_authorship.npy')
			self.authorship = []
			self.authorlist = [[] for _ in range(self.np)]
			self.paperlist = [[] for _ in range(self.nr)]
			i = 0
			for row in data:
				r_list = list(map(bool, row.tolist()))
				self.authorship.append(r_list)
				for j in range(self.nr):
					if r_list[j] == True:
						self.authorlist[i].append(j)
						self.paperlist[j].append(i)
				i += 1

			self.bid = [[False for _ in range(self.nr)] for _ in range(self.np)]
			self.bidlist = [[] for _ in range(self.nr)]
			self.bidauthorship = [[False for _ in range(self.nr)] for _ in range(self.nr)]
			self.bidauthorlist = [[] for _ in range(self.nr)]
			with open('datasets/aamas2021/aamas_2021.csv', 'r') as file:
				reader = csv.reader(file)
				for row in reader:
					if row[0].startswith('pc') and (row[2] == 'yes' or row[2] == 'maybe'):
						self.bid[int(row[1]) - 1][int(row[0][3:]) - 1] = True
						self.bidlist[int(row[0][3:]) - 1].append(int(row[1]) - 1)

			for i in range(self.nr):
				for j in self.bidlist[i]:
					for k in self.authorlist[j]:
						if not self.bidauthorship[i][k]:
							self.bidauthorship[i][k] = True
							self.bidauthorlist[i].append(k)

			counter = 0
			for i in range(self.nr):
				for j in self.bidauthorlist[i]:
					if j > i and self.bidauthorship[j][i]:
						counter += 1
			
			print('num_2_cycles:', counter)

			self.coauthorship = [[False for _ in range(self.nr)] for _ in range(self.nr)]
			self.coauthorlist = [[] for _ in range(self.nr)]
			np.random.seed(123)
			cnt = 0
			for i in range(self.nr):
				for j in range(i + 1, self.nr):
					now = 0.0
					for k in range(self.np):
						now += self.s[k][i] * self.s[k][j]
					if np.random.binomial(1, 10 / self.nr * now) == 1:
						self.coauthorship[i][j] = self.coauthorship[j][i] = True
						self.coauthorlist[i].append(j)
						self.coauthorlist[j].append(i)
						cnt += 1
			print('avg_coauthors:', cnt * 2 / self.nr, '!!!')
   
		else:
			file = open('datasets/' + dataset.lower() + '.in', 'r')
			lines = file.readlines()
			file.close()
			# Read the number of papers and reviewers
			[self.np, self.nr] = list(map(int, lines[0].split(' ')))
			lines = lines[1 : ]
			# Read the similarity matrix
			self.s = []
			for line in lines:
				splitline = line.split(' ')
				while splitline[-1] == '\n':
					splitline.pop()
				self.s.append(list(map(float, splitline)))
			# Load the papers' requirement and reviewers' load limit
			self.ellp = 3
			self.ellr = 6 + 6 * (dataset[: 5] == 'AAMAS') + (dataset == 'Preflib2')
			if (dataset == 'counter'):
				self.ellp = 1
				self.ellr = 1

			if dataset.lower() == 'iclr2018':
				self.coauthorship = [[False for _ in range(self.nr)] for _ in range(self.nr)]
				self.coauthorlist = [[] for _ in range(self.nr)]

				# import torch
				# A = torch.tensor(self.s)
				# B = torch.mm(A.T, A)
				cnt = 0
				np.random.seed(123)
				for i in range(self.nr):
					for j in range(i + 1, self.nr):
						if np.random.binomial(1, 2 / self.nr) == 1:
						# if np.random.binomial(1, 10 / self.nr * B[i][j]) == 1:
							self.coauthorship[i][j] = self.coauthorship[j][i] = True
							self.coauthorlist[i].append(j)
							self.coauthorlist[j].append(i)
							cnt += 1
				print('avg_coauthors:', cnt * 2 / self.nr, '!!!')

		# Initialize maximum quality
		if (init):
			import algorithms
			import metrics
			self.max_quality = metrics.quality(self, algorithms.PLRA(self, maxprob = 1))
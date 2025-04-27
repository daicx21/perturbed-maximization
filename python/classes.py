import csv
import numpy as np
from collections import defaultdict

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
		if dataset.lower() == 'testlarge':
			np.random.seed(100)
			sz = 100
			h1 = 150
			h2 = 120
			self.np = sz * h1
			self.nr = sz * h2
			self.ellp = 4
			self.ellr = 6
			self.s = [defaultdict(float) for _ in range(self.np)]
			self.authorship = [defaultdict(bool) for _ in range(self.np)]
			self.authorlist = [[] for _ in range(self.np)]
			self.paperlist = [[] for _ in range(self.nr)]
			self.bid = [defaultdict(bool) for _ in range(self.np)]
			self.bidlist = [[] for _ in range(self.nr)]
			self.biddedlist = [[] for _ in range(self.np)]
			self.bidauthorship = [defaultdict(bool) for _ in range(self.nr)]
			self.bidauthorlist = [[] for _ in range(self.nr)]
			for i1 in range(sz):
				for i2 in range(h1):
					i = i1 * h1 + i2
					bo = defaultdict(bool)

					for j in range(90):
						id = np.random.randint(0, h2) + i1 * h2
						while bo[id] == True:
							id = np.random.randint(0, h2) + i1 * h2
						bo[id] = True
						self.s[i][id] = np.random.uniform(0.8, 1.0)
						self.bid[i][id] = True
						self.bidlist[id].append(i)
						self.biddedlist[i].append(id)
					
					for j in range(np.random.randint(0, 5)):
						id = np.random.randint(0, h2) + i1 * h2
						while bo[id] == True:
							id = np.random.randint(0, h2) + i1 * h2
						bo[id] = True
						self.authorship[i][id] = True
						self.authorlist[i].append(id)
						self.paperlist[id].append(i)

					for j in range(10):
						id = np.random.randint(0, self.nr)
						while bo[id] == True:
							id = np.random.randint(0, self.nr)
						bo[id] = True
						self.s[i][id] = np.random.uniform(0.5, 1.0)
						self.bid[i][id] = True
						self.bidlist[id].append(i)
						self.biddedlist[i].append(id)
			
			for i in range(self.nr):
				for j in self.bidlist[i]:
					for k in self.authorlist[j]:
						if not self.bidauthorship[i][k]:
							self.bidauthorship[i][k] = True
							self.bidauthorlist[i].append(k)
			
			counter = 0
			with open('results/bidauthorship.out', 'w') as file:
				for i in range(self.nr):
					for j in self.bidauthorlist[i]:
						if j > i and self.bidauthorship[j][i]:
							counter += 1
							print(i, j, file=file)
			
			print('num_2_cycles:', counter)

			self.coauthorship = [defaultdict(bool) for _ in range(self.nr)]
			self.coauthorlist = [[] for _ in range(self.nr)]
			cnt = 0
			for i1 in range(sz):
				for i in range(i1 * h2, (i1 + 1) * h2):
					for j in range(i + 1, (i1 + 1) * h2):
						now = 0.0
						for k in self.bidlist[i]:
							now += self.s[k][i] * self.s[k][j]
						if np.random.binomial(1, 5 / self.nr * now) == 1:
							self.coauthorship[i][j] = True
							self.coauthorship[j][i] = True
							self.coauthorlist[i].append(j)
							self.coauthorlist[j].append(i)
							cnt += 1
			print('avg_coauthors:', cnt * 2 / self.nr, '!!!')


		elif dataset.lower() == 'aamas2021':
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
			with open('results/bidauthorship.out', 'w') as file:
				for i in range(self.nr):
					for j in self.bidauthorlist[i]:
						if j > i and self.bidauthorship[j][i]:
							counter += 1
							print(i, j, file=file)
			
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
				np.random.seed(123)

				self.authorship = [[False for _ in range(self.nr)] for _ in range(self.np)]
				self.authorlist = [[] for _ in range(self.np)]
				self.paperlist = [[] for _ in range(self.nr)]

				self.bid = [[False for _ in range(self.nr)] for _ in range(self.np)]
				self.bidlist = [[] for _ in range(self.nr)]
				self.bidauthorship = [[False for _ in range(self.nr)] for _ in range(self.nr)]
				self.bidauthorlist = [[] for _ in range(self.nr)]

				avg_bid = 0
				avg_author = 0
				for i in range(self.nr):
					for j in range(self.np):
						if self.s[j][i] < 0.05:
							continue
						if np.random.rand() < 0.9:
							self.bid[j][i] = True
							self.bidlist[i].append(j)
							avg_bid += 1
						elif np.random.rand() < 0.5:
							self.authorship[j][i] = True
							self.authorlist[j].append(i)
							self.paperlist[i].append(j)
							avg_author += 1
				
				print("avg_bid:", avg_bid / self.nr, "avg_author:", avg_author / self.nr)

				for i in range(self.nr):
					for j in self.bidlist[i]:
						for k in self.authorlist[j]:
							if not self.bidauthorship[i][k]:
								self.bidauthorship[i][k] = True
								self.bidauthorlist[i].append(k)

				counter = 0
				with open('results/bidauthorship.out', 'w') as file:
					for i in range(self.nr):
						for j in self.bidauthorlist[i]:
							if j > i and self.bidauthorship[j][i]:
								counter += 1
								print(i, j, file=file)
				
				print('num_2_cycles:', counter)

				self.coauthorship = [[False for _ in range(self.nr)] for _ in range(self.nr)]
				self.coauthorlist = [[] for _ in range(self.nr)]

				import torch
				A = torch.tensor(self.s)
				B = torch.mm(A.T, A)
				cnt = 0
				for i in range(self.nr):
					for j in range(i + 1, self.nr):
						if np.random.binomial(1, 10 / self.nr * B[i][j]) == 1:
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
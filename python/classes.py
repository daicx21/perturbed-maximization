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
			self.np = 16000
			self.nr = 14000
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
			for i in range(self.np):
				bo = defaultdict(bool)
				for j in range(700):
					id = np.random.randint(0, self.nr)
					while bo[id] == True:
						id = np.random.randint(0, self.nr)
					bo[id] = True
					hh = np.random.normal(0.6, 0.15)
					while hh < 0.1 or hh > 1:
						hh = np.random.normal(0.6, 0.15)
					hhh = np.random.randint(0, 100)
					if hhh < 2:
						hh += 1
					elif hhh < 4:
						hh += 0.5
					hh /= 2
					self.s[i][id] = hh
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
			for i in range(self.nr):
				for j in self.bidauthorlist[i]:
					if j > i and self.bidauthorship[j][i]:
						counter += 1
			print('num_2_cycles:', counter)

			self.coauthorship = [defaultdict(bool) for _ in range(self.nr)]
			self.coauthorlist = [[] for _ in range(self.nr)]
			cnt = 0
			for i in range(self.np):
				bo = defaultdict(bool)
				for j in range(10):
					h1 = self.biddedlist[i][np.random.randint(0, len(self.biddedlist[i]))]
					h2 = self.biddedlist[i][np.random.randint(0, len(self.biddedlist[i]))]
					while h1 == h2 or self.coauthorship[h1][h2] == True:
						h1 = self.biddedlist[i][np.random.randint(0, len(self.biddedlist[i]))]
						h2 = self.biddedlist[i][np.random.randint(0, len(self.biddedlist[i]))]
					self.coauthorship[h1][h2] = True
					self.coauthorship[h2][h1] = True
					self.coauthorlist[h1].append(h2)
					self.coauthorlist[h2].append(h1)
					cnt += 1
			print('avg_coauthors:', cnt * 2 / self.nr)

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
			print('avg_coauthors:', cnt * 2 / self.nr)

		elif dataset.lower() == 'wu':
			import torch
			tensor_data = torch.load("datasets/wu/wu_tensor_data.pl", weights_only=False)
			(self.np, self.nr) = tensor_data['tpms'].T.shape
			self.s = []
			for row in tensor_data['tpms'].T:
				self.s.append(row.tolist())
			self.ellp = 4
			self.ellr = 6

			author_data = np.load('datasets/wu/wu_authorship.npy')
			self.authorship = []
			self.authorlist = [[] for _ in range(self.np)]
			self.paperlist = [[] for _ in range(self.nr)]
			for i, row in enumerate(author_data):
				mask = row.astype(bool)
				self.authorship.append(mask.tolist())
				cols = np.nonzero(mask)[0]
				for j in cols:
					self.s[i][j] = 0.0
					self.authorlist[i].append(j)
					self.paperlist[j].append(i)

			self.bid = [[False for _ in range(self.nr)] for _ in range(self.np)]
			self.bidlist = []
			self.bidauthorship = [[False for _ in range(self.nr)] for _ in range(self.nr)]
			self.bidauthorlist = [[] for _ in range(self.nr)]

			counter = 0
			for i, row in enumerate(tensor_data['label']):
				mask = row.astype(bool)
				cols = np.nonzero(mask)[0]
				self.bidlist.append(cols)
				for j in cols:
					if row[j] == 1:
						continue
					self.bid[j][i] = True
					if row[j] == 2:
						self.s[j][i] += 0.5
					else:
						self.s[j][i] += 1
					counter += 1
			print('num_bids:', counter)

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
			cnt = 0
			with open('datasets/wu/wu_coauthorship.in', 'r') as file:
				for line in file:
					i, j = map(int, line.strip().split())
					self.coauthorship[i][j] = self.coauthorship[j][i] = True
					self.coauthorlist[i].append(j)
					self.coauthorlist[j].append(i)
					cnt += 1
			print('avg_coauthors:', cnt * 2 / self.nr)

			for i in range(self.nr):
				for j in self.coauthorlist[i]:
					for k in self.paperlist[j]:
						self.s[k][i] = 0

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
				for i in range(self.nr):
					for j in self.bidauthorlist[i]:
						if j > i and self.bidauthorship[j][i]:
							counter += 1
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
				print('avg_coauthors:', cnt * 2 / self.nr)

		# Initialize maximum quality
		if (init):
			import algorithms
			import metrics
			self.max_quality = metrics.quality(self, algorithms.PLRA(self, maxprob = 1))
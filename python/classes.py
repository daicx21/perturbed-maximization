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
			self.np = 2446 * 6
			self.nr = 2483 * 5
			self.ellp = 4
			self.ellr = 6
			self.s = [defaultdict(float) for _ in range(self.np)]
			self.remained = [defaultdict(bool) for _ in range(self.np)]
			self.remained_r_for_p = [[] for _ in range(self.np)]
			self.remained_p_for_r = [[] for _ in range(self.nr)]
			self.authorship = [defaultdict(bool) for _ in range(self.np)]
			self.authorlist = [[] for _ in range(self.np)]
			self.paperlist = [[] for _ in range(self.nr)]
			self.bid = [defaultdict(bool) for _ in range(self.np)]
			self.bidlist = [[] for _ in range(self.nr)]
			self.biddedlist = [[] for _ in range(self.np)]
			self.bidauthorship = [defaultdict(bool) for _ in range(self.nr)]
			self.bidauthorlist = [[] for _ in range(self.nr)]
			self.coauthorship = [defaultdict(bool) for _ in range(self.nr)]
			self.coauthorlist = [[] for _ in range(self.nr)]

			np.random.seed(123)

			import torch
			tensor_data = torch.load("datasets/wu/wu_tensor_data.pl", weights_only=False)
			(wu_np, wu_nr) = tensor_data['tpms'].T.shape
			wu_s = []
			for row in tensor_data['tpms'].T:
				wu_s.append(row.tolist())

			for i1 in range(wu_np):
				list_i1 = wu_s[i1].copy()
				for j in range(wu_nr):
					list_i1[j] = (list_i1[j], j)
				list_i1 = sorted(list_i1, reverse=True)
				list_i1 = list_i1[:200]
				for i2 in range(6):
					i = i1 * 6 + i2
					for k in range(500):
						j1 = np.random.randint(0, 200)
						j2 = np.random.randint(0, 5)
						j = list_i1[j1][1] * 5 + j2
						rand_pow = np.random.uniform(0.8, 1.2)
						rand_val = list_i1[j1][0] ** rand_pow
						if not self.remained[i][j]:
							self.remained[i][j] = True
							self.s[i][j] = rand_val
							self.remained_r_for_p[i].append(j)
							self.remained_p_for_r[j].append(i)

			author_data = np.load('datasets/wu/wu_authorship.npy')
			for i1, row in enumerate(author_data):
				mask = row.astype(bool)
				cols = np.nonzero(mask)[0]
				for i2 in range(6):
					i = i1 * 6 + i2
					for j1 in cols:
						j2 = np.random.randint(0, 5)
						j = j1 * 5 + j2
						self.authorship[i][j] = True
						self.authorlist[i].append(j)
						self.paperlist[j].append(i)

			counter = 0
			for i1, row in enumerate(tensor_data['label']):
				mask = row.astype(bool)
				cols = np.nonzero(mask)[0]
				for i2 in range(5):
					i = i1 * 5 + i2
					for j1 in cols:
						if row[j1] == 1:
							continue
						j2 = np.random.randint(0, 6)
						j = j1 * 6 + j2
						if not self.remained[j][i]:
							self.remained[j][i] = True
							self.remained_r_for_p[j].append(i)
							self.remained_p_for_r[i].append(j)
						self.bid[j][i] = True
						self.bidlist[i].append(j)
						self.biddedlist[j].append(i)
						if row[j1] == 2:
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
			
			wu_coauthorlist = [[] for _ in range(wu_nr)]
			with open('datasets/wu/wu_coauthorship.in', 'r') as file:
				for line in file:
					i, j = map(int, line.strip().split())
					wu_coauthorlist[i].append(j)
			
			cnt = 0
			for i1 in range(wu_nr):
				for i2 in range(5):
					i = i1 * 5 + i2
					for j1 in wu_coauthorlist[i1]:
						j2 = np.random.randint(0, 5)
						j = j1 * 5 + j2
						if not self.coauthorship[i][j]:
							self.coauthorship[i][j] = self.coauthorship[j][i] = True
							self.coauthorlist[i].append(j)
							self.coauthorlist[j].append(i)
					if np.random.rand() < 0.3:
						j2 = np.random.randint(0, 5)
						j = i1 * 5 + j2
						if i != j and not self.coauthorship[i][j]:
							self.coauthorship[i][j] = self.coauthorship[j][i] = True
							self.coauthorlist[i].append(j)
							self.coauthorlist[j].append(i)

			cnt = 0
			for i in range(self.nr):
				cnt += len(self.coauthorlist[i])
			print('avg_coauthors:', cnt / self.nr)

			for i in range(self.nr):
				for j in self.paperlist[i]:
					self.s[j][i] = 0
				for j in self.coauthorlist[i]:
					for k in self.paperlist[j]:
						self.s[k][i] = 0

			self.region = []
			self.seniority = []
			with open('datasets/large/reviewers.in', 'r') as file:
				for line in file:
					reviewer_features = line.strip().split()
					self.region.append(int(reviewer_features[0]))
					self.seniority.append(int(reviewer_features[1]))

		elif dataset == 'S2ORC':
			import torch
			tensor_data = torch.load("datasets/wu/wu_tensor_data.pl", weights_only=False)
			(self.np, self.nr) = tensor_data['tpms'].T.shape
			self.s = []
			for row in tensor_data['tpms'].T:
				self.s.append(row.tolist())
			self.ellp = 4
			self.ellr = 5

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
			for i in range(self.np):
				for j in range(len(self.authorlist[i])):
					for k in range(j + 1, len(self.authorlist[i])):
						if not self.coauthorship[self.authorlist[i][j]][self.authorlist[i][k]]:
							self.coauthorship[self.authorlist[i][j]][self.authorlist[i][k]] = True
							self.coauthorship[self.authorlist[i][k]][self.authorlist[i][j]] = True
							self.coauthorlist[self.authorlist[i][j]].append(self.authorlist[i][k])
							self.coauthorlist[self.authorlist[i][k]].append(self.authorlist[i][j])
			for i in range(self.nr):
				for j in self.coauthorlist[i]:
					for k in self.paperlist[j]:
						self.s[k][i] = 0
			with open('datasets/wu/wu_coauthorship.in', 'r') as file:
				for line in file:
					i, j = map(int, line.strip().split())
					if not self.coauthorship[i][j]:
						self.coauthorship[i][j] = self.coauthorship[j][i] = True
						self.coauthorlist[i].append(j)
						self.coauthorlist[j].append(i)
			cnt = 0
			for i in range(self.nr):
				cnt += len(self.coauthorlist[i])
			print('avg_coauthors:', cnt / self.nr)

			self.region = []
			self.seniority = []
			with open('datasets/wu/reviewers.csv', newline='', encoding='utf-8') as csvfile:
				reader = list(csv.reader(csvfile))[1:]
				for row in reader:
					self.seniority.append(int(row[2]))
					self.region.append(int(row[5][6:]))

		elif dataset == 'AAMAS2021':
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
			with open('datasets/aamas2021/aamas_coauthorship.in', 'r') as file:
				for line in file:
					coauthors = line.strip().split()
					i = int(coauthors[0])
					j = int(coauthors[1])
					self.coauthorship[i][j] = self.coauthorship[j][i] = True
					self.coauthorlist[i].append(j)
					self.coauthorlist[j].append(i)
					cnt += 1
			print('avg_coauthors:', cnt * 2 / self.nr)

			for i in range(self.np):
				for j in range(self.nr):
					self.s[i][j] = min(self.s[i][j] * 20, 2)

			for i in range(self.nr):
				for j in self.paperlist[i]:
					self.s[j][i] = 0
				for j in self.coauthorlist[i]:
					for k in self.paperlist[j]:
						self.s[k][i] = 0

			self.seniority = []
			self.region = []
			with open('datasets/aamas2021/aamas_reviewers.in', 'r') as file:
				for line in file:
					reviewer_features = line.strip().split()
					self.seniority.append(int(reviewer_features[0]))
					self.region.append(int(reviewer_features[1]))

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
			self.ellp = 4
			self.ellr = 6 + 6 * (dataset[: 5] == 'AAMAS') + (dataset == 'Preflib2')
			if (dataset == 'counter'):
				self.ellp = 1
				self.ellr = 1

			if dataset == 'ICLR2018':
				self.authorship = [[False for _ in range(self.nr)] for _ in range(self.np)]
				self.authorlist = [[] for _ in range(self.np)]
				self.paperlist = [[] for _ in range(self.nr)]

				self.bid = [[False for _ in range(self.nr)] for _ in range(self.np)]
				self.bidlist = [[] for _ in range(self.nr)]
				self.bidauthorship = [[False for _ in range(self.nr)] for _ in range(self.nr)]
				self.bidauthorlist = [[] for _ in range(self.nr)]

				avg_bid = 0
				avg_author = 0
				with open('datasets/iclr2018/authorship.in', 'r') as file:
					for line in file:
						auth_pair = line.strip().split()
						i = int(auth_pair[0])
						j = int(auth_pair[1])
						self.authorship[j][i] = True
						self.authorlist[j].append(i)
						self.paperlist[i].append(j)
						avg_author += 1
				with open('datasets/iclr2018/bid.in', 'r') as file:
					for line in file:
						bid_pair = line.strip().split()
						i = int(bid_pair[0])
						j = int(bid_pair[1])
						self.bid[j][i] = True
						self.bidlist[i].append(j)
						avg_bid += 1
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
				cnt = 0
				with open('datasets/iclr2018/coauthorship.in', 'r') as file:
					for line in file:
						coauthor_pair = line.strip().split()
						i = int(coauthor_pair[0])
						j = int(coauthor_pair[1])
						self.coauthorship[i][j] = self.coauthorship[j][i] = True
						self.coauthorlist[i].append(j)
						self.coauthorlist[j].append(i)
						cnt += 1
				print('avg_coauthors:', cnt * 2 / self.nr)

				for i in range(self.nr):
					for j in self.paperlist[i]:
						self.s[j][i] = 0
					for j in self.coauthorlist[i]:
						for k in self.paperlist[j]:
							self.s[k][i] = 0
				
				for i in range(self.np):
					for j in range(self.nr):
						self.s[i][j] = min(self.s[i][j] * 20, 2)

				self.seniority = []
				self.region = []
				with open('datasets/iclr2018/reviewers.in', 'r') as file:
					for line in file:
						reviewer_features = line.strip().split()
						self.seniority.append(int(reviewer_features[0]))
						self.region.append(int(reviewer_features[1]))
		
		if dataset.lower() != 'testlarge':
			self.remained = [[False for _ in range(self.nr)] for _ in range(self.np)]
			self.remained_r_for_p = [[] for _ in range(self.np)]
			self.remained_p_for_r = [[] for _ in range(self.nr)]

			for i in range(self.np):
				for j in range(self.nr):
					if self.s[i][j] > 0:
						self.remained[i][j] = True
						self.remained_r_for_p[i].append(j)
						self.remained_p_for_r[j].append(i)

		# Initialize maximum quality
		if (init):
			import algorithms
			import metrics
			self.max_quality = metrics.quality(self, algorithms.PLRA(self, maxprob = 1))
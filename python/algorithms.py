# This file contains the Gurobi implementation of PLRA and two variants of PM.
# 	For PM-Quadratic, the perturbation function used is f(x) = x - beta * x ^ 2
# 	For PM-Exponential, the perturbation function used is f(x) = 1 - exp(-alpha * x)

def PLRA(instance, maxprob = 1.0):
	# PLRA Gurobi implementation
	# Inputs arguments:
	# 	instance: the input instance
	# 	maxprob:  maximum allowed assignment probability
	# Output: an assignment matrix in nested list
	# Initialize Gurobi solver
	import gurobipy as gp
	solver = gp.Model()
	solver.setParam('OutputFlag', 1)

	if instance.dataset.lower() == 'testlarge':
		# Initialize assignment matrix and objective function
		objective  = 0.0
		assignment = [{} for i in range(instance.np)]
		for i in range(instance.np):
			for j in instance.biddedlist[i]:
				x = solver.addVar(lb = 0, ub = maxprob, name = f"{i} {j}")
				assignment[i][j] = x
				objective += x * instance.s[i][j]
		solver.setObjective(objective, gp.GRB.MAXIMIZE)
		# Add ellp & ellr as constraints
		for i in range(instance.np):
			assigned = 0.0
			for j in instance.biddedlist[i]:
				assigned += assignment[i][j]
			solver.addConstr(assigned == instance.ellp)
		for j in range(instance.nr):
			load = 0.0
			for i in instance.bidlist[j]:
				load += assignment[i][j]
			solver.addConstr(load <= instance.ellr)

		# Run the Gurobi solver
		solver.params.Method = 1
		solver.optimize()

		# Return the resulting matching
		for i in range(instance.np):
			for j in assignment[i].keys():
				assignment[i][j] = assignment[i][j].X
		return assignment

	# Initialize assignment matrix and objective function
	objective  = 0.0
	assignment = [[0.0 for j in range(instance.nr)] for i in range(instance.np)]
	for i in range(instance.np):
		for j in range(instance.nr):
			x = solver.addVar(lb = 0, ub = maxprob, name = f"{i} {j}")
			assignment[i][j] = x
			objective += x * instance.s[i][j]
	solver.setObjective(objective, gp.GRB.MAXIMIZE)
	# Add ellp & ellr as constraints
	for i in range(instance.np):
		assigned = 0.0
		for j in range(instance.nr):
			assigned += assignment[i][j]
		solver.addConstr(assigned == instance.ellp)
	for j in range(instance.nr):
		load = 0.0
		for i in range(instance.np):
			load += assignment[i][j]
		solver.addConstr(load <= instance.ellr)

	# Run the Gurobi solver
	solver.params.Method = 1
	solver.optimize()

	# Return the resulting matching
	return [[assignment[i][j].X for j in range(instance.nr)] for i in range(instance.np)]

def PMQ(instance, beta = 0.5, maxprob = 1.0):
	# PM-Quadratic Gurobi implementation
	# 	The perturbation function used is f(x) = x - beta * x ^ 2
	# Inputs arguments:
	# 	instance: the input instance
	#   beta:     the parameter used in the perturbation function
	# 	maxprob:  maximum allowed assignment probability
	# Output: an assignment matrix in nested list

	# Initialize Gurobi solver
	import gurobipy as gp
	solver = gp.Model()
	solver.setParam('OutputFlag', 0)
	# Initialize assignment matrix and objective function
	objective  = 0.0
	assignment = [[0.0 for j in range(instance.nr)] for i in range(instance.np)]
	for i in range(instance.np):
		for j in range(instance.nr):
			x = solver.addVar(lb = 0, ub = maxprob, name = f"{i} {j}")
			assignment[i][j] = x
			objective += (x - beta * x * x) * instance.s[i][j]
	# Add ellp & ellr as constraints
	for i in range(instance.np):
		assigned = 0.0
		for j in range(instance.nr):
			assigned += assignment[i][j]
		solver.addConstr(assigned == instance.ellp)
	for j in range(instance.nr):
		load = 0.0
		for i in range(instance.np):
			load += assignment[i][j]
		solver.addConstr(load <= instance.ellr)
	solver.setObjective(objective, gp.GRB.MAXIMIZE)
	# Run the Gurobi solver
	solver.optimize()
	# Return the resulting matching
	return [[assignment[i][j].X for j in range(instance.nr)] for i in range(instance.np)]

def ours(instance, beta = 0.5, maxprob = 1.0):
	# Initialize Gurobi solver
	import gurobipy as gp
	solver = gp.Model()
	solver.setParam('OutputFlag', 1)

	assignment = [[0 for j in range(instance.nr)] for i in range(instance.np)]
	for i in range(instance.np):
		for j in range(instance.nr):
			if instance.s[i][j] > 0:
				assignment[i][j] = solver.addVar(lb = 0, ub = maxprob)

	for j in range(instance.nr):
		list_neighbours = instance.coauthorlist[j].copy()
		list_neighbours.append(j)
		for i in range(instance.np):
			hh = 0
			for k in list_neighbours:
				if instance.bid[i][k]:
					hh += assignment[i][k]
			hhh = solver.addVar()
			solver.addConstr(hhh == hh)
			solver.setPWLObj(hhh, [0, 1, 2], [0, 0, 0.2])

	deleted = [[False for _ in range(instance.nr)] for _ in range(instance.np)]
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

	pr_list = sorted(pr_list, key=lambda x: instance.s[x[0]][x[1]])
	vis_pr_list = defaultdict(bool)
	for tup1 in reversed(pr_list):
		vis_pr_list[tup1] = True
		for tup2 in map_2cycles[tup1]:
			if vis_pr_list[tup2] and not deleted[tup2[0]][tup2[1]]:
				deleted[tup1[0]][tup1[1]] = True
				break

	vis_pr_list = defaultdict(bool)
	for i in range(instance.nr):
		for j in instance.bidauthorlist[i]:
			if j > i and instance.bidauthorship[j][i]:
				for k in instance.paperlist[i]:
					for l in instance.paperlist[j]:
						if instance.bid[l][i] and instance.bid[k][j]:
							sum_2cycle = solver.addVar(lb = 0, ub = 1)
							solver.addConstr(sum_2cycle >= assignment[l][i] + assignment[k][j])
							xpts = []
							ypts = []
							now = 0
							while now <= 1:
								xpts.append(now)
								ypts.append(now * now * 0.1)
								now += 0.1
							solver.setPWLObj(sum_2cycle, xpts, ypts)

	for i in range(instance.np):
		for j in range(instance.nr):
			if instance.s[i][j] == 0:
				continue
			xpts = []
			ypts = []
			now = 0
			val = max(1e-3, instance.s[i][j] * beta)
			while now <= maxprob + (1e-6):
				xpts.append(now)
				ypts.append(- now * instance.s[i][j] + now * now * val)
				now += 0.1
			solver.setPWLObj(assignment[i][j], xpts, ypts)

	for i in range(instance.np):
		assigned = 0
		for j in range(instance.nr):
			assigned += assignment[i][j]
		solver.addConstr(assigned == instance.ellp)
	for j in range(instance.nr):
		load = 0
		for i in range(instance.np):
			load += assignment[i][j]
		solver.addConstr(load <= instance.ellr)

	solver.params.Method = 1
	# Run the Gurobi solver
	solver.optimize()
 
	# Return the resulting matching
	for i in range(instance.np):
		for j in range(instance.nr):
			if instance.s[i][j] > 0:
				assignment[i][j] = assignment[i][j].X
	return assignment

def PMPL(instance, beta = 0.5, maxprob = 1.0):
	# PM-Quadratic Gurobi implementation
	# 	The perturbation function used is f(x) = x - beta * x ^ 2
	# Inputs arguments:
	# 	instance: the input instance
	#   beta:     the parameter used in the perturbation function
	# 	maxprob:  maximum allowed assignment probability
	# Output: an assignment matrix in nested list

	# Initialize Gurobi solver
	import gurobipy as gp
	solver = gp.Model()
	solver.setParam('OutputFlag', 1)

	if instance.dataset.lower() == 'testlarge':
		objective  = 0.0
		assignment = [{} for i in range(instance.np)]

		for i in range(instance.np):
			for j in instance.biddedlist[i]:
				x = solver.addVar(lb = 0, ub = maxprob, name = f"{i} {j}")
				assignment[i][j] = x
				xpts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
				ypts = []
				for k in range(10):
					now = xpts[k]
					ypts.append(- (now - beta * now * now) * instance.s[i][j])
				solver.setPWLObj(x, xpts, ypts)
		# 		objective += - (x - beta * x * x) * instance.s[i][j]
		# solver.setObjective(objective)

		# Add ellp & ellr as constraints
		for i in range(instance.np):
			assigned = 0.0
			for j in instance.biddedlist[i]:
				assigned += assignment[i][j]
			solver.addConstr(assigned == instance.ellp)
		for j in range(instance.nr):
			load = 0.0
			for i in instance.bidlist[j]:
				load += assignment[i][j]
			solver.addConstr(load <= instance.ellr)

		# Run the Gurobi solver
		solver.params.Method = 1
		solver.optimize()
	
		# Return the resulting matching
		for i in range(instance.np):
			for j in assignment[i].keys():
				assignment[i][j] = assignment[i][j].X
		return assignment

	# Initialize assignment matrix and objective function
	assignment = [[0.0 for j in range(instance.nr)] for i in range(instance.np)]
	for i in range(instance.np):
		for j in range(instance.nr):
			x = solver.addVar(lb = 0, ub = maxprob, name = f"{i} {j}")
			assignment[i][j] = x

	deg = [len(instance.coauthorlist[i]) + 1 for i in range(instance.nr)]
	res = [[instance.s[i][j] for j in range(instance.nr)] for i in range(instance.np)]

	for i in range(instance.np):
		for j in range(instance.nr):
			now = instance.s[i][j] / deg[j]
			for k in instance.coauthorlist[j]:
				now = min(now, instance.s[i][k] / deg[k])
			now = max(now, 1e-3)
			res[i][j] -= now
			hhh = assignment[i][j]
			for k in instance.coauthorlist[j]:
				res[i][k] -= now
				hhh += assignment[i][k]
			hh = solver.addVar(lb = 0, ub = maxprob)
			solver.addConstr(hh == hhh)
			xpts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
			ypts = []
			for k in range(10):
				ypts.append(beta * xpts[k] * xpts[k] * now)
			solver.setPWLObj(hh, xpts, ypts)

	for i in range(instance.np):
		for j in range(instance.nr):
			xpts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
			ypts = []
			for k in range(10):
				ypts.append(- xpts[k] * instance.s[i][j] + beta * xpts[k] * xpts[k] * max(res[i][j], 1e-3))
			solver.setPWLObj(assignment[i][j], xpts, ypts)

	# Add ellp & ellr as constraints
	for i in range(instance.np):
		assigned = 0.0
		for j in range(instance.nr):
			assigned += assignment[i][j]
		solver.addConstr(assigned == instance.ellp)
	for j in range(instance.nr):
		load = 0.0
		for i in range(instance.np):
			load += assignment[i][j]
		solver.addConstr(load <= instance.ellr)

	# Dual Simplex
	solver.params.Method = 1
	# Run the Gurobi solver
	solver.optimize()
 
	# Return the resulting matching
	return [[assignment[i][j].X for j in range(instance.nr)] for i in range(instance.np)]

def PMPL_second(instance, beta = 0.5, maxprob = 1.0):
	# Initialize Gurobi solver
	import gurobipy as gp
	solver = gp.Model()
	solver.setParam('OutputFlag', 1)

	if instance.dataset.lower() == 'testlarge':
		objective  = 0.0
		assignment = [{} for i in range(instance.np)]
		for i in range(instance.np):
			for j in instance.biddedlist[i]:
				if instance.deleted[i][j] == False:
					x = solver.addVar(lb = 0, ub = maxprob, name = f"{i} {j}")
				else:
					x = solver.addVar(lb = 0, ub = 0, name = f"{i} {j}")
				assignment[i][j] = x
				xpts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
				ypts = []
				for k in range(10):
					now = xpts[k]
					ypts.append(- (now - beta * now * now) * instance.s[i][j])
				solver.setPWLObj(x, xpts, ypts)

		for j in range(instance.nr):
			list_neighbours = instance.coauthorlist[j].copy()
			list_neighbours.append(j)
			for i in instance.bidlist[j]:
				hh = 0
				for k in list_neighbours:
					if instance.bid[i][k]:
						hh += assignment[i][k]
				solver.addConstr(hh <= maxprob)
		
		for i in range(instance.nr):
			for j in instance.bidauthorlist[i]:
				if j > i and instance.bidauthorship[j][i]:
					for k in instance.paperlist[i]:
						for l in instance.paperlist[j]:
							if instance.bid[l][i] and instance.bid[k][j]:
								hh = assignment[l][i] + assignment[k][j]
								solver.addConstr(hh <= maxprob)

		# Add ellp & ellr as constraints
		for i in range(instance.np):
			assigned = 0.0
			for j in instance.biddedlist[i]:
				assigned += assignment[i][j]
			solver.addConstr(assigned == instance.ellp)
		for j in range(instance.nr):
			load = 0.0
			for i in instance.bidlist[j]:
				load += assignment[i][j]
			solver.addConstr(load <= instance.ellr)

		solver.params.Method = 1
		# Run the Gurobi solver
		import time
		print(time.time())
		solver.optimize()
		print(time.time())
	
		# Return the resulting matching
		for i in range(instance.np):
			for j in assignment[i].keys():
				assignment[i][j] = assignment[i][j].X
		return assignment

	# Initialize assignment matrix and objective function
	objective  = 0.0
	assignment = [[0.0 for j in range(instance.nr)] for i in range(instance.np)]
	for i in range(instance.np):
		for j in range(instance.nr):
			if instance.deleted[i][j] == False:
				x = solver.addVar(lb = 0, ub = maxprob, name = f"{i} {j}")
				xpts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
				ypts = []
				for k in range(10):
					now = xpts[k]
					ypts.append(- (now - beta * now * now) * instance.s[i][j])
				solver.setPWLObj(x, xpts, ypts)
			else:
				x = solver.addVar(lb = 0, ub = 0, name = f"{i} {j}")
			assignment[i][j] = x

	# import numpy as np
	# for j in range(instance.nr):
	# 	list_neighbours = [j]
	# 	for k in range(j + 1, instance.nr):
	# 		if instance.coauthorship[j][k] == True:
	# 			list_neighbours.append(k)
	# 	for i in range(instance.np):
	# 		hh = 0
	# 		for k in list_neighbours:
	# 			hh += assignment[i][k]
	# 		solver.addConstr(hh <= maxprob)

	# Add ellp & ellr as constraints
	for i in range(instance.np):
		assigned = 0.0
		for j in range(instance.nr):
			assigned += assignment[i][j]
		solver.addConstr(assigned == instance.ellp)
	for j in range(instance.nr):
		load = 0.0
		for i in range(instance.np):
			load += assignment[i][j]
		solver.addConstr(load <= instance.ellr)

	solver.params.Method = 1
	# Run the Gurobi solver
	solver.optimize()
 
	# Return the resulting matching
	return [[assignment[i][j].X for j in range(instance.nr)] for i in range(instance.np)]
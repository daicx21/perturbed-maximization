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
	solver.setParam('OutputFlag', 0)

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
	import time
	print(time.time())
	solver.optimize()
	print(time.time())

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
	solver.setParam('OutputFlag', 1)

	if instance.dataset.lower() == 'testlarge':
		objective  = 0.0
		assignment = [{} for i in range(instance.np)]

		for i in range(instance.np):
			for j in instance.biddedlist[i]:
				x = solver.addVar(lb = 0, ub = maxprob, name = f"{i} {j}")
				assignment[i][j] = x
				objective += (x - beta * x * x) * instance.s[i][j]

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

		solver.setObjective(objective, gp.GRB.MAXIMIZE)
		# Run the Gurobi solver
		import time
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
			objective += (x - beta * x * x) * instance.s[i][j]

	for j in range(instance.nr):
		list_neighbours = [j]
		for k in range(j + 1, instance.nr):
			if instance.coauthorship[j][k] == True:
				list_neighbours.append(k)
		for i in range(instance.np):
			hh = 0
			for k in list_neighbours:
				hh += assignment[i][k]
			solver.addConstr(hh <= maxprob)
	
	map_bo = {}
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
	import time
	solver.optimize()
 
	# Return the resulting matching
	return [[assignment[i][j].X for j in range(instance.nr)] for i in range(instance.np)]

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

		# for j in range(instance.nr):
		# 	list_neighbours = instance.coauthorlist[j].copy()
		# 	list_neighbours.append(j)
		# 	for i in instance.bidlist[j]:
		# 		hh = 0
		# 		for k in list_neighbours:
		# 			if instance.bid[i][k]:
		# 				hh += assignment[i][k]
		# 		solver.addConstr(hh <= maxprob)
		
		# for i in range(instance.nr):
		# 	for j in instance.bidauthorlist[i]:
		# 		if j > i and instance.bidauthorship[j][i]:
		# 			for k in instance.paperlist[i]:
		# 				for l in instance.paperlist[j]:
		# 					if instance.bid[l][i] and instance.bid[k][j]:
		# 						hh = assignment[l][i] + assignment[k][j]
		# 						solver.addConstr(hh <= maxprob)

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
			xpts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
			ypts = []
			for k in range(10):
				now = xpts[k]
				ypts.append(- (now - beta * now * now) * instance.s[i][j])
			solver.setPWLObj(x, xpts, ypts)

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
	
	# map_bo = {}
	# for i in range(instance.nr):
	# 	for j in instance.bidauthorlist[i]:
	# 		if j > i and instance.bidauthorship[j][i]:
	# 			for k in instance.paperlist[i]:
	# 				for l in instance.paperlist[j]:
	# 					if instance.bid[l][i] and instance.bid[k][j]:
	# 						# if map_bo.get((l, i)) != True:
	# 						# 	map_bo[(l, i)] = True
	# 						# 	hh = solver.addVar(lb = 0, ub = maxprob)
	# 						# 	solver.addConstr(hh == assignment[l][i])
	# 						# 	solver.setPWLObj(hh, [0, maxprob], [0, maxprob * beta * instance.s[l][i]])
	# 						# if map_bo.get((k, j)) != True:
	# 						# 	map_bo[(k, j)] = True
	# 						# 	hh = solver.addVar(lb = 0, ub = maxprob)
	# 						# 	solver.addConstr(hh == assignment[k][j])
	# 						# 	solver.setPWLObj(hh, [0, maxprob], [0, maxprob * beta * instance.s[k][j]])
	# 						hh = assignment[l][i] + assignment[k][j]
	# 						solver.addConstr(hh <= maxprob)

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
	import time
	print(time.time())
	solver.optimize()
	print(time.time())
 
	# Return the resulting matching
	return [[assignment[i][j].X for j in range(instance.nr)] for i in range(instance.np)]

def PMPL_second(instance, beta = 0.5, maxprob = 1.0):
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

		import numpy as np
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
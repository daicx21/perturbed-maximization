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

	# solver.Params.Method = 2  # Barrier
	# solver.Params.BarOrder = 1  # GPU

	# Run the Gurobi solver
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
	solver.setParam('OutputFlag', 1)

	# Initialize assignment matrix and objective function
	objective  = 0.0
	assignment = [[0.0 for j in range(instance.nr)] for i in range(instance.np)]
	for i in range(instance.np):
		for j in range(instance.nr):
			x = solver.addVar(lb = 0, ub = maxprob, name = f"{i} {j}")
			assignment[i][j] = x
			objective += (-x + beta * x * x) * instance.s[i][j]

	if instance.dataset.lower() == 'aamas2021' or instance.dataset.lower() == 'iclr2018':
		coauthorship_degree = [0 for j in range(instance.nr)]
		for j in range(instance.nr):
			for k in range(j + 1, instance.nr):
				if instance.coauthorship[j][k] == True:
					coauthorship_degree[j] += 1
					coauthorship_degree[k] += 1

		import numpy as np
		for j in range(instance.nr):
			for k in range(j + 1, instance.nr):
				if instance.coauthorship[j][k] == False:
					continue
				for i in range(instance.np):
					objective += 2.0 * beta * (np.sqrt((instance.s[i][j] / coauthorship_degree[j]) * (instance.s[i][k] / coauthorship_degree[k])) - (1e-6)) * assignment[i][j] * assignment[i][k]

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

	solver.setObjective(objective, gp.GRB.MINIMIZE)
	# Run the Gurobi solver
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
	solver.setParam('OutputFlag', 0)

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
			# objective += (x - beta * x * x) * instance.s[i][j]

	if instance.dataset.lower() == 'aamas2021' or instance.dataset.lower() == 'iclr2018':
		coauthorship_degree = [0 for j in range(instance.nr)]
		for j in range(instance.nr):
			for k in range(j + 1, instance.nr):
				if instance.coauthorship[i][j] == True:
					coauthorship_degree[j] += 1
					coauthorship_degree[k] += 1

		import numpy as np
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
	
	if instance.dataset.lower() == 'aamas2021':
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

	solver.params.Method = 1
	# Run the Gurobi solver
	solver.optimize()
 
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

	if instance.dataset.lower() == 'aamas2021' or instance.dataset.lower() == 'iclr2018':
		coauthorship_degree = [0 for j in range(instance.nr)]
		for j in range(instance.nr):
			for k in range(j + 1, instance.nr):
				if instance.coauthorship[i][j] == True:
					coauthorship_degree[j] += 1
					coauthorship_degree[k] += 1

		import numpy as np
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

def PME(instance, alpha = 1.0, maxprob = 1.0):
	# PM-Exponential Gurobi implementation
	# 	The perturbation function used is f(x) = 1 - exp(-alpha * x)
	#   The objective function is not quadratic, but Gurobi only supports quadratic 
	# 	objectives. Therefore, the successive approximation method is used.
	# Inputs arguments:
	# 	instance: the input instance
	#   alpha:    the parameter used in the perturbation function
	# 	maxprob:  maximum allowed assignment probability
	# Output: an assignment matrix in nested list
	def FinishedPME(instance, current, nextret, eps):
		# Decides whether or not to proceed the successive approximation method
		# Inputs arguments:
		# 	instance: the input instance
		#   current:  the current assignment
		# 	nextret:  the next assignment
		#   eps:      the precision parameter
		# Output: [L infinity distance of current and nextret <= eps]
		for i in range(instance.np):
			for j in range(instance.nr):
				if (abs(current[i][j] - nextret[i][j]) > eps):
					return False
		return True

	def IterationPME(instance, current, alpha, maxprob):
		# One iteration of the successive approximation method
		# Inputs arguments:
		# 	instance: the input instance
		#   current:  the current assignment
		#   alpha:    the parameter used in the perturbation function
		# 	maxprob:  maximum allowed assignment probability
		# Output: the next assignment
		# Initialize Gurobi solver
		import gurobipy as gp
		from numpy import exp
		solver = gp.Model()
		solver.setParam('OutputFlag', 0)
		# Initialize assignment matrix and objective function
		objective  = 0.0
		assignment = [[0.0 for j in range(instance.nr)] for i in range(instance.np)]
		for i in range(instance.np):
			for j in range(instance.nr):
				x = solver.addVar(lb = 0, ub = maxprob, name = f"{i} {j}")
				assignment[i][j] = x
				pos = current[i][j]
				coef = [exp(-alpha * pos), -alpha / 2 * exp(-alpha * pos)]
				# Taylor expansion at pos to the second order
				objective += (coef[0] * (x - pos) + coef[1] * (x - pos) * (x - pos)) * instance.s[i][j]
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
		solver.optimize()
		# Return the resulting matching
		return [[assignment[i][j].X for j in range(instance.nr)] for i in range(instance.np)]

	# The successive approximation method
	eps = 1e-2
	current = [[0.0 for j in range(instance.nr)] for i in range(instance.np)]
	nextret = IterationPME(instance, current, alpha, maxprob)
	while not FinishedPME(instance, current, nextret, eps):
		current = nextret
		nextret = IterationPME(instance, current, alpha, maxprob)
	return nextret
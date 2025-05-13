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
	# Initialize assignment matrix and objective function
	objective  = 0.0
	from collections import defaultdict
	assignment = [defaultdict(float) for i in range(instance.np)]
	for i in range(instance.np):
		for j in instance.remained_r_for_p[i]:
			x = solver.addVar(lb = 0, ub = maxprob, name = f"{i} {j}")
			assignment[i][j] = x
			objective += x * instance.s[i][j]
	solver.setObjective(objective, gp.GRB.MAXIMIZE)
	# Add ellp & ellr as constraints
	for i in range(instance.np):
		assigned = 0.0
		for j in instance.remained_r_for_p[i]:
			assigned += assignment[i][j]
		solver.addConstr(assigned == instance.ellp)
	for j in range(instance.nr):
		load = 0.0
		for i in instance.remained_p_for_r[j]:
			load += assignment[i][j]
		solver.addConstr(load <= instance.ellr)
	# Use dual simplex
	solver.params.Method = 1
	# Run the Gurobi solver
	solver.optimize()
	# Return the resulting matching
	for i in range(instance.np):
		for j in instance.remained_r_for_p[i]:
			assignment[i][j] = assignment[i][j].X
	return assignment

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
	from collections import defaultdict
	assignment = [defaultdict(float) for i in range(instance.np)]
	for i in range(instance.np):
		for j in instance.remained_r_for_p[i]:
			x = solver.addVar(lb = 0, ub = maxprob, name = f"{i} {j}")
			assignment[i][j] = x
			objective += (-x + beta * x * x) * max(instance.s[i][j], 1e-3)
	# Add ellp & ellr as constraints
	for i in range(instance.np):
		assigned = 0.0
		for j in instance.remained_r_for_p[i]:
			assigned += assignment[i][j]
		solver.addConstr(assigned == instance.ellp)
	for j in range(instance.nr):
		load = 0.0
		for i in instance.remained_p_for_r[j]:
			load += assignment[i][j]
		solver.addConstr(load <= instance.ellr)
	solver.setObjective(objective)
	solver.params.Method = 2
	# Run the Gurobi solver
	solver.optimize()
	# Return the resulting matching
	for i in range(instance.np):
		for j in instance.remained_r_for_p[i]:
			assignment[i][j] = assignment[i][j].X
	return assignment

def ours(instance, beta = 0.5, maxprob = 1.0, reward_region = 0.1, target_sen = 2, reward_sen = 0.1, pen_coauthor = 0.2, pen_2cycle = 0.1):
	# Initialize Gurobi solver
	import gurobipy as gp
	solver = gp.Model()
	solver.setParam('OutputFlag', 1)

	# Set a large scalar for all objectives to prevent the solver get stucked because of numerical precision issues
	obj_scalar = 1000

	# Create variables
	from collections import defaultdict
	assignment = [defaultdict(float) for i in range(instance.np)]
	for i in range(instance.np):
		for j in instance.remained_r_for_p[i]:
			assignment[i][j] = solver.addVar(lb = 0, ub = maxprob)

	# Set coauthor penalty
	objective = 0
	for j in range(instance.nr):
		list_neighbours = instance.coauthorlist[j].copy()
		list_neighbours.append(j)
		for i in instance.remained_p_for_r[j]:
			coauthor_var = solver.addVar(lb = 1)
			coauthor_sum = 0
			for k in list_neighbours:
				coauthor_sum += assignment[i][k]
			solver.addConstr(coauthor_var >= coauthor_sum)
			objective += obj_scalar * pen_coauthor * coauthor_var

	# Set region reward
	for i in range(instance.np):
		reg_sum = [0 for _ in range(5)]
		for j in instance.remained_r_for_p[i]:
			reg_sum[instance.region[j]] += assignment[i][j]
		for reg in range(5):
			reg_var = solver.addVar(lb = 0, ub = 1)
			solver.addConstr(reg_var <= reg_sum[reg])
			objective -= obj_scalar * reward_region * reg_var

	# Set seniority reward
	for i in range(instance.np):
		sen_var = solver.addVar(lb = 0, ub = target_sen)
		sen_sum = 0
		for j in instance.remained_r_for_p[i]:
			if instance.seniority[j] >= 2:
				sen_sum += assignment[i][j]
		solver.addConstr(sen_var <= sen_sum)
		objective -= obj_scalar * reward_sen * sen_var
	
	# Set linear objective
	# Gurobi needs to first set linear objectives, then set piecewise-linear objectives
	solver.setObjective(objective)

	# Set 2cycle penalty
	from collections import defaultdict
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
								ypts.append(obj_scalar * pen_2cycle * now * now)
								now += 0.1
							solver.setPWLObj(sum_2cycle, xpts, ypts)

	# Add piecewise linear objectives
	for i in range(instance.np):
		for j in instance.remained_r_for_p[i]:
			xpts = []
			ypts = []
			now = 0
			val = max(1e-3, instance.s[i][j])
			while now <= maxprob + (1e-6):
				xpts.append(now)
				ypts.append(obj_scalar * (- now + beta * now * now) * val)
				now += 0.1
			solver.setPWLObj(assignment[i][j], xpts, ypts)

	# Add assigned and load constraints
	for i in range(instance.np):
		assigned = 0
		for j in instance.remained_r_for_p[i]:
			assigned += assignment[i][j]
		solver.addConstr(assigned == instance.ellp)
	for j in range(instance.nr):
		load = 0
		for i in instance.remained_p_for_r[j]:
			load += assignment[i][j]
		solver.addConstr(load <= instance.ellr)

	# Use dual simplex
	solver.params.Method = 1
	# Run the Gurobi solver
	solver.optimize()
 
	# Return the resulting matching
	for i in range(instance.np):
		for j in instance.remained_r_for_p[i]:
			assignment[i][j] = assignment[i][j].X
	return assignment

def ours_minus_piecewise_linear(instance, beta = 0.5, maxprob = 1.0, reward_region = 0.1, target_sen = 2, reward_sen = 0.1, pen_coauthor = 0.2, pen_2cycle = 0.1):
	# Initialize Gurobi solver
	import gurobipy as gp
	solver = gp.Model()
	solver.setParam('OutputFlag', 1)

	# Set a large scalar for all objectives to prevent the solver get stucked because of numerical precision issues
	obj_scalar = 1000

	# Create variables
	from collections import defaultdict
	assignment = [defaultdict(float) for i in range(instance.np)]
	for i in range(instance.np):
		for j in instance.remained_r_for_p[i]:
			assignment[i][j] = solver.addVar(lb = 0, ub = maxprob)

	# Set coauthor penalty
	objective = 0
	for j in range(instance.nr):
		list_neighbours = instance.coauthorlist[j].copy()
		list_neighbours.append(j)
		for i in instance.remained_p_for_r[j]:
			coauthor_var = solver.addVar(lb = 1)
			coauthor_sum = 0
			for k in list_neighbours:
				coauthor_sum += assignment[i][k]
			solver.addConstr(coauthor_var >= coauthor_sum)
			objective += obj_scalar * pen_coauthor * coauthor_var

	# Set region reward
	for i in range(instance.np):
		reg_sum = [0 for _ in range(5)]
		for j in instance.remained_r_for_p[i]:
			reg_sum[instance.region[j]] += assignment[i][j]
		for reg in range(5):
			reg_var = solver.addVar(lb = 0, ub = 1)
			solver.addConstr(reg_var <= reg_sum[reg])
			objective -= obj_scalar * reward_region * reg_var

	# Set seniority reward
	for i in range(instance.np):
		sen_var = solver.addVar(lb = 0, ub = target_sen)
		sen_sum = 0
		for j in instance.remained_r_for_p[i]:
			if instance.seniority[j] >= 2:
				sen_sum += assignment[i][j]
		solver.addConstr(sen_var <= sen_sum)
		objective -= obj_scalar * reward_sen * sen_var

	# Set 2cycle penalty
	from collections import defaultdict
	vis_pr_list = defaultdict(bool)
	for i in range(instance.nr):
		for j in instance.bidauthorlist[i]:
			if j > i and instance.bidauthorship[j][i]:
				for k in instance.paperlist[i]:
					for l in instance.paperlist[j]:
						if instance.bid[l][i] and instance.bid[k][j]:
							sum_2cycle = solver.addVar(lb = 0, ub = 1)
							solver.addConstr(sum_2cycle >= assignment[l][i] + assignment[k][j])
							objective += obj_scalar * pen_2cycle * sum_2cycle * sum_2cycle 

	# Add piecewise linear objectives
	for i in range(instance.np):
		for j in instance.remained_r_for_p[i]:
			objective += obj_scalar * (-assignment[i][j] + beta * assignment[i][j] * assignment[i][j]) * max(1e-3, instance.s[i][j])
	
	solver.setObjective(objective)

	# Add assigned and load constraints
	for i in range(instance.np):
		assigned = 0
		for j in instance.remained_r_for_p[i]:
			assigned += assignment[i][j]
		solver.addConstr(assigned == instance.ellp)
	for j in range(instance.nr):
		load = 0
		for i in instance.remained_p_for_r[j]:
			load += assignment[i][j]
		solver.addConstr(load <= instance.ellr)

	# Run the Gurobi solver
	solver.optimize()
 
	# Return the resulting matching
	for i in range(instance.np):
		for j in instance.remained_r_for_p[i]:
			assignment[i][j] = assignment[i][j].X
	return assignment
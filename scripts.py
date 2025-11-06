import pulp

solver = pulp.CPLEX_PY(msg=True)

prob = pulp.LpProblem("test", pulp.LpMinimize)
x = pulp.LpVariable("x", lowBound=0)
prob += x
prob.solve(solver)
print("Status:", pulp.LpStatus[prob.status])
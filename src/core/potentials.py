import numpy as np
from transport.src.core.models import TransportPlan, TransportProblem
from transport.src.core.northwest_corner import northwest_corner


def potential_method(problem: TransportProblem, initial_plan=None):
    if initial_plan is None:
        initial_plan = northwest_corner(problem)
    plan = initial_plan.allocation.copy()
    costs = problem.costs.copy()
    m, n = costs.shape
    iterations = [initial_plan.iterations[-1]]

    while True:
        basis = np.argwhere(plan > 0)
        u = np.zeros(m)
        v = np.zeros(n)
        deltas = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if plan[i, j] == 0:
                    deltas[i, j] = costs[i, j] - u[i] - v[j]

        min_delta = np.min(deltas)
        if min_delta >= -1e-9:
            total_cost = np.sum(plan * costs)
            return TransportPlan(plan, total_cost, iterations)
        i0, j0 = np.unravel_index(np.argmin(deltas), deltas.shape)
        cycle = find_cycle(plan, (i0, j0))
        theta = min(plan[i, j] for (i, j) in cycle[1::2])
        # Перераспределить
        for idx, (i, j) in enumerate(cycle):
            if idx % 2 == 0:
                plan[i, j] += theta
            else:
                plan[i, j] -= theta
        # Убрать численные погрешности
        plan[np.abs(plan) < 1e-9] = 0
        iterations.append({'allocation': plan.copy(), 'deltas': deltas.copy(), 'cycle': cycle, 'theta': theta})

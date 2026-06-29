import numpy as np
from transport.src.core.models import TransportProblem, TransportPlan


def northwest_corner(problem: TransportProblem) -> TransportPlan:
    costs = problem.costs.copy()
    supply = problem.supplies.copy()
    demand = problem.demands.copy()
    m, n = costs.shape
    allocation = np.zeros((m, n), dtype=float)
    iterations = []
    i, j = 0, 0

    while i < m and j < n:
        x = min(supply[i], demand[j])
        allocation[i, j] = x
        supply[i] -= x
        demand[j] -= x
        iterations.append({
            'step': f'({i + 1},{j + 1})',
            'allocation': allocation.copy(),
            'supply': supply.copy(),
            'demand': demand.copy()
        })
        if supply[i] == 0:
            i += 1
        elif demand[j] == 0:
            j += 1
        else:
            i += 1
            j += 1

    total_cost = np.sum(allocation * costs)
    return TransportPlan(allocation, total_cost, iterations)

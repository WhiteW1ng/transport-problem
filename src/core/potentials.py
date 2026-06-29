import numpy as np
from collections import deque
from transport.src.core.models import TransportPlan, TransportProblem
from transport.src.core.northwest_corner import northwest_corner

def potential_method(problem: TransportProblem, initial_plan=None):
    if initial_plan is None:
        initial_plan = northwest_corner(problem)

    costs = problem.costs.copy()
    supply = problem.supplies.copy()
    demand = problem.demands.copy()
    m, n = costs.shape
    allocation = initial_plan.allocation.copy()
    iterations = [initial_plan.iterations[-1].copy()]  # сохраним начальный план как итерацию

    while True:
        basis = list(zip(*np.where(allocation > 1e-9)))
        u = np.full(m, np.nan)
        v = np.full(n, np.nan)
        u[0] = 0

        from collections import deque
        q = deque()
        q.append(0)
        adj = [[] for _ in range(m + n)]
        for (i, j) in basis:
            adj[i].append((j, 'v', costs[i, j]))
            adj[m + j].append((i, 'u', costs[i, j]))

        visited = [False] * (m + n)
        visited[0] = True
        q = deque([0])
        while q:
            node = q.popleft()
            for neighbor, typ, cst in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    if typ == 'v':
                        i = node
                        j = neighbor - m
                        v[j] = cst - u[i]
                    else:
                        j = node - m
                        i = neighbor
                        u[i] = cst - v[j]
                    q.append(neighbor)

        u = np.nan_to_num(u, nan=0.0)
        v = np.nan_to_num(v, nan=0.0)

        deltas = np.full((m, n), np.inf)
        for i in range(m):
            for j in range(n):
                if allocation[i, j] < 1e-9:  # свободная клетка
                    deltas[i, j] = costs[i, j] - u[i] - v[j]

        min_delta = np.min(deltas)
        if min_delta >= -1e-9:
            total_cost = np.sum(allocation * costs)
            return TransportPlan(allocation, total_cost, iterations)

        i0, j0 = np.unravel_index(np.argmin(deltas), deltas.shape)

        cycle = find_cycle(allocation, (i0, j0), basis)
        if not cycle:
            break

        theta = min(allocation[i, j] for idx, (i, j) in enumerate(cycle) if idx % 2 == 1)
        for idx, (i, j) in enumerate(cycle):
            if idx % 2 == 0:
                allocation[i, j] += theta
            else:  # нечётные – вычитаем
                allocation[i, j] -= theta

        allocation[np.abs(allocation) < 1e-9] = 0.0

        iterations.append({
            'step': len(iterations),
            'allocation': allocation.copy(),
            'deltas': deltas.copy(),
            'cycle': cycle,
            'theta': theta
        })

    # Если цикл прерван, возвращаем текущий план как последний
    total_cost = np.sum(allocation * costs)
    return TransportPlan(allocation, total_cost, iterations)

def find_cycle(allocation, start_cell, basis):
    m, n = allocation.shape
    i0, j0 = start_cell
    basis_set = set(basis)
    basis_set.add((i0, j0))

    row_neighbors = {i: [] for i in range(m)}
    col_neighbors = {j: [] for j in range(n)}
    for (i, j) in basis_set:
        row_neighbors[i].append(j)
        col_neighbors[j].append(i)

    stack = [(i0, j0, [(i0, j0)], 'row')]
    visited_edges = set()

    while stack:
        i, j, path, last_move = stack.pop()
        if last_move == 'row':
            for jj in row_neighbors[i]:
                if (i, jj) in visited_edges:
                    continue
                new_path = path + [(i, jj)]
                if (i, jj) == (i0, j0) and len(new_path) > 2:
                    # Возвращаем без последнего элемента (повтора стартовой)
                    return new_path[:-1]
                visited_edges.add((i, jj))
                stack.append((i, jj, new_path, 'col'))
        else:
            for ii in col_neighbors[j]:
                if (ii, j) in visited_edges:
                    continue
                new_path = path + [(ii, j)]
                if (ii, j) == (i0, j0) and len(new_path) > 2:
                    return new_path[:-1]
                visited_edges.add((ii, j))
                stack.append((ii, j, new_path, 'row'))
    return []

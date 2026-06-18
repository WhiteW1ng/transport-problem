from dataclasses import dataclass
import numpy as np

@dataclass
class TransportProblem:
    costs: np.ndarray          # матрица стоимостей (m x n)
    supplies: np.ndarray       # запасы (m)
    demands: np.ndarray        # потребности (n)

@dataclass
class TransportPlan:
    allocation: np.ndarray    # матрица перевозок (m x n)
    total_cost: float
    iterations: list          # логи итераций для пошагового вывода

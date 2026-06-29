import json
import numpy as np

def read_problem(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return TransportProblem(
        costs=np.array(data['costs']),
        supplies=np.array(data['supplies']),
        demands=np.array(data['demands'])
    )


import json
import numpy as np
from transport.src.core.models import TransportProblem


def read_problem(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Проверка наличия ключей
    required_keys = {'costs', 'supplies', 'demands'}
    if not required_keys.issubset(data.keys()):
        raise ValueError("JSON должен содержать ключи: costs, supplies, demands")

    costs = np.array(data['costs'], dtype=float)
    supplies = np.array(data['supplies'], dtype=float)
    demands = np.array(data['demands'], dtype=float)

    # Проверка размеров
    if costs.shape[0] != len(supplies):
        raise ValueError("Количество строк матрицы стоимостей не совпадает с числом поставщиков")
    if costs.shape[1] != len(demands):
        raise ValueError("Количество столбцов матрицы стоимостей не совпадает с числом потребителей")

    # Проверка баланса
    total_supply = np.sum(supplies)
    total_demand = np.sum(demands)
    if not np.isclose(total_supply, total_demand):
        raise ValueError(f"Сумма запасов ({total_supply}) не равна сумме потребностей ({total_demand})")

    return TransportProblem(costs, supplies, demands)

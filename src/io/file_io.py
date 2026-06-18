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

import os
import json
import numpy as np
from flask import Flask, render_template, request

# Импорты из нашего проекта
from transport.src.core.models import TransportProblem
from transport.src.core.northwest_corner import northwest_corner
from transport.src.core.potentials import potential_method

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Проверяем наличие файла
        if 'datafile' not in request.files:
            return render_template('index.html', error="Файл не выбран")
        file = request.files['datafile']
        if file.filename == '':
            return render_template('index.html', error="Файл не выбран")

        try:
            # Читаем JSON из файла
            data = json.load(file)
            # Проверяем обязательные поля
            if not all(key in data for key in ('costs', 'supplies', 'demands')):
                return render_template('index.html',
                                       error="Неверный формат JSON: требуются ключи costs, supplies, demands")

            problem = TransportProblem(
                costs=np.array(data['costs']),
                supplies=np.array(data['supplies']),
                demands=np.array(data['demands'])
            )

            # Выполняем алгоритмы
            initial = northwest_corner(problem)
            final = potential_method(problem, initial)

            # Подготавливаем итерации для шаблона
            iterations = []
            for idx, it in enumerate(final.iterations):
                iter_data = {
                    'step': idx + 1,
                    'allocation': it['allocation'].tolist() if isinstance(it['allocation'], np.ndarray) else it[
                        'allocation'],
                }
                if 'deltas' in it and it['deltas'] is not None:
                    iter_data['deltas'] = it['deltas'].tolist() if isinstance(it['deltas'], np.ndarray) else it[
                        'deltas']
                else:
                    iter_data['deltas'] = None
                if 'cycle' in it:
                    iter_data['cycle'] = it['cycle']
                if 'theta' in it:
                    iter_data['theta'] = it['theta']
                iterations.append(iter_data)

            return render_template('result.html',
                                   iterations=iterations,
                                   total_cost=final.total_cost,
                                   final_allocation=final.allocation.tolist() if isinstance(final.allocation,
                                                                                            np.ndarray) else final.allocation,
                                   enumerate=enumerate)
        except json.JSONDecodeError:
            return render_template('index.html', error="Ошибка парсинга JSON. Проверьте формат файла.")
        except Exception as e:
            return render_template('index.html', error=f"Ошибка при вычислениях: {str(e)}")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

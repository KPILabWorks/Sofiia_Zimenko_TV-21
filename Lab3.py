import time
import random

# Генерація великого набору даних енергоспоживання (кВт*год)
def generate_energy_data(num_records):
    data = []
    for _ in range(num_records):
        record = {
            'solar': random.randint(100, 1000),
            'wind': random.randint(50, 800),
            'hydro': random.randint(30, 500),
            'consumption': random.randint(800, 2000)
        }
        data.append(record)
    return data

expression = "(solar + wind + hydro) - consumption"

# Функція для обчислення виразу з використанням eval()
def calculate_with_eval(data, expression):
    results = []
    for record in data:
        try:
            result = eval(expression, {}, record)
            results.append(result)
        except Exception as e:
            print(f"Помилка при обчисленні виразу: {e}")
    return results

energy_data = generate_energy_data(1000)

start_time = time.time()
eval_results = calculate_with_eval(energy_data, expression)
eval_time = time.time() - start_time

print("Розрахунок суми енергоспоживання")
print(f"Час обчислення з eval(): {eval_time:.4f} секунд")
print(f"Перші 5 результатів: {eval_results[:5]}")

# Функція для обчислення виразу без eval()
def calculate_without_eval(data):
    results = []
    for record in data:
        solar = record['solar']
        wind = record['wind']
        hydro = record['hydro']
        consumption = record['consumption']
        result = (solar + wind + hydro) - consumption
        results.append(result)
    return results

start_time = time.time()
direct_results = calculate_without_eval(energy_data)
direct_time = time.time() - start_time

print(f"\nЧас обчислення без eval(): {direct_time:.4f} секунд")
print(f"Перші 5 результатів: {direct_results[:5]}")

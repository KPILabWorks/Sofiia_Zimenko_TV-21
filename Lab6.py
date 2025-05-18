import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

# Придушення попередження joblib
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Обмежити кількість ядер

# Список файлів із даними
files = {
    "Центр_День": "Центр_День.csv",
    "Центр_Ніч": "Центр_Ніч.csv",
    "Спальний_День": "Спальний_День.csv",
    "Спальний_Ніч": "Спальний_Ніч.csv",
    "Парк_День": "Парк_День.csv",
    "Парк_Ніч": "Парк_Ніч.csv"
}

# Зберігання статистики та даних для кластеризації
summary = {}
all_data = []
location_labels = []

# Обробка кожного файлу
plt.figure(figsize=(12, 8))
for name, file in files.items():
    # Завантаження даних
    data = pd.read_csv(file)

    # Згладжування даних (ковзне середнє, вікно = 5)
    data["Smoothed_Noise_dB"] = data["Noise_dB"].rolling(window=5, center=True).mean().bfill().ffill()

    # Розрахунок статистики
    mean_noise = data["Noise_dB"].mean()
    median_noise = data["Noise_dB"].median()
    std_noise = data["Noise_dB"].std()
    amplitude = data["Noise_dB"].max() - data["Noise_dB"].min()

    # Збереження статистики
    summary[name] = {
        "mean": mean_noise,
        "median": median_noise,
        "std": std_noise,
        "amplitude": amplitude
    }

    # Збереження даних для кластеризації
    all_data.append([mean_noise, std_noise, amplitude])
    location_labels.append(name)

    # Візуалізація: оригінальні та згладжені дані
    plt.plot(data["Time"], data["Noise_dB"], label=f"{name} (оригінал)", alpha=0.5)
    plt.plot(data["Time"], data["Smoothed_Noise_dB"], label=f"{name} (згладжено)", linewidth=2)

    print(f"\nЛокація: {name}")
    print(f"Середній рівень шуму: {mean_noise:.2f} дБ")
    print(f"Медіана: {median_noise:.2f} дБ")
    print(f"Стандартне відхилення: {std_noise:.2f} дБ")
    print(f"Амплітуда: {amplitude:.2f} дБ")

plt.xlabel("Час")
plt.ylabel("Рівень шуму (дБ)")
plt.title("Динаміка рівня шуму для всіх локацій")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Кластеризація (KMeans) із вибором оптимальної кількості кластерів
scaler = StandardScaler()
scaled_data = scaler.fit_transform(all_data)
best_silhouette = -1
best_n_clusters = 2
for n_clusters in [2, 3]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    silhouette = silhouette_score(scaled_data, clusters)
    print(f"\nSilhouette Score для {n_clusters} кластерів: {silhouette:.2f}")
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_n_clusters = n_clusters
        best_clusters = clusters

# Виведення результатів найкращої кластеризації
print(f"\nНайкраща кількість кластерів: {best_n_clusters}")
print(f"Найкращий Silhouette Score: {best_silhouette:.2f}")
for loc, cluster in zip(location_labels, best_clusters):
    print(f"Локація {loc}: Кластер {cluster}")

# Виявлення аномалій (Isolation Forest)
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(all_data)
for loc, anomaly in zip(location_labels, anomalies):
    if anomaly == -1:
        print(f"Локація {loc}: Аномалія виявлена")

# Прогнозування рівня шуму (Linear Regression) для Центр_День
data = pd.read_csv("Центр_День.csv")
X = np.arange(len(data)).reshape(-1, 1)  # Індекси часу
y = data["Noise_dB"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\nРегресія для Центр_День: R² = {r2:.2f}")

# Візуалізація регресії
plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="blue", label="Дані")
plt.plot(X, reg.predict(X), color="red", label="Лінійна регресія")
plt.xlabel("Час (індекс)")
plt.ylabel("Рівень шуму (дБ)")
plt.title("Прогнозування рівня шуму: Центральна частина, день")
plt.legend()
plt.tight_layout()
plt.show()

# Порівняння середніх значень
locations = ["Центр", "Спальний", "Парк"]
day_means = [summary["Центр_День"]["mean"], summary["Спальний_День"]["mean"], summary["Парк_День"]["mean"]]
night_means = [summary["Центр_Ніч"]["mean"], summary["Спальний_Ніч"]["mean"], summary["Парк_Ніч"]["mean"]]

plt.figure(figsize=(10, 5))
bar_width = 0.35
x = range(len(locations))
plt.bar(x, day_means, bar_width, label="День", color="#FF5722")
plt.bar([i + bar_width for i in x], night_means, bar_width, label="Ніч", color="#0288D1")
plt.xlabel("Локація")
plt.ylabel("Середній рівень шуму (дБ)")
plt.title("Порівняння рівня шуму в Ірпені: день vs ніч")
plt.xticks([i + bar_width / 2 for i in x], locations)
plt.legend()
plt.tight_layout()
plt.show()
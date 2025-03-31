import numpy as np
import matplotlib.pyplot as plt
import pywt

# 1. Генерація синтетичного часового ряду споживання енергії
np.random.seed(42)
t = np.linspace(0, 24, 1440)  # 24 години з вимірами кожну хвилину (1440 точок)
freq1, freq2 = 1/24, 1/2  # добова і погодинна періодичність
energy = (10 * np.sin(2 * np.pi * freq1 * t) +  # добові коливання
          5 * np.sin(2 * np.pi * freq2 * t) +    # погодинні коливання
          np.random.normal(0, 1, t.size))        # шум

# 2. Вейвлет-перетворення
wavelet = 'db4'  # Використаємо вейвлет Добеші 4
coeffs = pywt.wavedec(energy, wavelet, level=6)
cA6, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

# 3. Аналіз Фур'є
fft_result = np.fft.fft(energy)
frequencies = np.fft.fftfreq(len(t), d=t[1] - t[0])
positive_freqs = frequencies[:len(frequencies)//2]
fft_magnitude = np.abs(fft_result)[:len(frequencies)//2]

# 4. Візуалізація
plt.figure(figsize=(15, 10))

# Оригінальний сигнал
plt.subplot(3, 1, 1)
plt.plot(t, energy, label='Споживання енергії')
plt.title('Оригінальний часовий ряд')
plt.xlabel('Час (години)')
plt.ylabel('Споживання')
plt.legend()

# Вейвлет-перетворення (деталізація на кількох рівнях)
plt.subplot(3, 1, 2)
for i, detail in enumerate([cD1, cD2, cD3, cD4, cD5, cD6], 1):
    plt.plot(detail, label=f'cD{i}')
plt.title('Вейвлет-коефіцієнти (деталізація)')
plt.xlabel('Зразки')
plt.ylabel('Амплітуда')
plt.legend()

# Аналіз Фур'є
plt.subplot(3, 1, 3)
plt.plot(positive_freqs, fft_magnitude, label='Амплітуда')
plt.title('Фур’є-спектр')
plt.xlabel('Частота (1/год)')
plt.ylabel('Амплітуда')
plt.xlim(0, 1)  # Обмежимо частоти для кращої видимості
plt.legend()

plt.tight_layout()
plt.show()

# 5. Порівняння частотних складових
print("Основні частоти у Фур'є-аналізі:")
peak_freqs = positive_freqs[np.argsort(fft_magnitude)[-2:]]  # Дві найбільші частоти
print(f"Частоти: {peak_freqs} (1/год)")
print("\nВейвлет-перетворення дозволяє локалізувати частоти в часі:")
for i, detail in enumerate([cD6, cD5, cD4], 1):
    print(f"Рівень cD{i+3}: частотний діапазон залежить від масштабу")
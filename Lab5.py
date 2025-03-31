import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1. Створення синтетичного набору даних
data = {
    'Колір': ['червоний', 'синій', 'зелений', 'червоний', 'синій'],
    'Розмір': ['S', 'M', 'L', 'M', 'S'],
    'Матеріал': ['дерево', 'метал', 'пластик', 'дерево', 'метал'],
    'Ціна': [100, 150, 200, 120, 130]  # Числова змінна для прикладу
}
df = pd.DataFrame(data)
print("Оригінальний набір даних:")
print(df)
print("\n")

# 2. Label Encoding (перетворення категорій у числа)
label_encoder = LabelEncoder()

# Застосовуємо Label Encoding до кожної категоріальної колонки
df_label_encoded = df.copy()
for column in ['Колір', 'Розмір', 'Матеріал']:
    df_label_encoded[column] = label_encoder.fit_transform(df[column])

print("Дані після Label Encoding:")
print(df_label_encoded)
print("\n")

# 3. One-Hot Encoding (перетворення категорій у бінарні стовпці)
# Використаємо pandas get_dummies для простоти
df_one_hot = pd.get_dummies(df, columns=['Колір', 'Розмір', 'Матеріал'], prefix=['Колір', 'Розмір', 'Матеріал'])

print("Дані після One-Hot Encoding:")
print(df_one_hot)

# Альтернативний спосіб із sklearn OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' уникає мультиколінеарності
encoded_features = one_hot_encoder.fit_transform(df[['Колір', 'Розмір', 'Матеріал']])
feature_names = one_hot_encoder.get_feature_names_out(['Колір', 'Розмір', 'Матеріал'])
df_one_hot_sklearn = pd.DataFrame(encoded_features, columns=feature_names)
df_one_hot_sklearn['Ціна'] = df['Ціна']

print("\nДані після One-Hot Encoding (sklearn):")
print(df_one_hot_sklearn)
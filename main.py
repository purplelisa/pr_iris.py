# Импортируем необходимые библиотеки
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Загружаем набор данных о цветках Ирис
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris_data = pd.read_csv(url, header=None, names=columns)

# Отображение первых 5 строк
print(iris_data.head())

# Визуализация данных
sns.pairplot(iris_data, hue='species')
plt.title('Pairplot of Iris Dataset')
plt.show()

# Проверка описательной статистики
print(iris_data.describe())

# Разделяем данные на признаки и целевую переменную
X = iris_data.drop("species", axis=1)
y = iris_data["species"]

# Разбиваем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизируем данные
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Обучение модели классификации с использованием случайного леса
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Визуализация важности признаков
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(importances)), importances[indices], align='center')
plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

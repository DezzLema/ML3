import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def load_communities_data():
    # Загрузка и подготовка данных Communities and Crime
    # Загрузка данных
    column_names = [
        'state', 'county', 'community', 'communityname', 'fold',
        'population', 'householdsize', 'racepctblack', 'racePctWhite',
        'racePctAsian', 'racePctHisp', 'agePct12t21', 'agePct12t29',
        'agePct16t24', 'agePct65up', 'numbUrban', 'pctUrban',
        'medIncome', 'pctWWage', 'pctWFarmSelf', 'pctWInvInc',
        'pctWSocSec', 'pctWPubAsst', 'pctWRetire', 'medFamInc',
        'perCapInc', 'whitePerCap', 'blackPerCap', 'indianPerCap',
        'AsianPerCap', 'OtherPerCap', 'HispPerCap', 'NumUnderPov',
        'PctPopUnderPov', 'PctLess9thGrade', 'PctNotHSGrad',
        'PctBSorMore', 'PctUnemployed', 'PctEmploy', 'PctEmplManu',
        'PctEmplProfServ', 'PctOccupManu', 'PctOccupMgmtProf',
        'MalePctDivorce', 'MalePctNevMarr', 'FemalePctDiv',
        'TotalPctDiv', 'PersPerFam', 'PctFam2Par', 'PctKids2Par',
        'PctYoungKids2Par', 'PctTeen2Par', 'PctWorkMomYoungKids',
        'PctWorkMom', 'NumIlleg', 'PctIlleg', 'NumImmig',
        'PctImmigRecent', 'PctImmigRec5', 'PctImmigRec8',
        'PctImmigRec10', 'PctRecentImmig', 'PctRecImmig5',
        'PctRecImmig8', 'PctRecImmig10', 'PctSpeakEnglOnly',
        'PctNotSpeakEnglWell', 'PctLargHouseFam', 'PctLargHouseOccup',
        'PersPerOccupHous', 'PersPerOwnOccHous', 'PersPerRentOccHous',
        'PctPersOwnOccup', 'PctPersDenseHous', 'PctHousLess3BR',
        'MedNumBR', 'HousVacant', 'PctHousOccup', 'PctHousOwnOcc',
        'PctVacantBoarded', 'PctVacMore6Mos', 'MedYrHousBuilt',
        'PctHousNoPhone', 'PctWOFullPlumb', 'OwnOccLowQuart',
        'OwnOccMedVal', 'OwnOccHiQuart', 'RentLowQ', 'RentMedian',
        'RentHighQ', 'MedRent', 'MedRentPctHousInc', 'MedOwnCostPctInc',
        'MedOwnCostPctIncNoMtg', 'NumInShelters', 'NumStreet',
        'PctForeignBorn', 'PctBornSameState', 'PctSameHouse85',
        'PctSameCity85', 'PctSameState85', 'LemasSwornFT',
        'LemasSwFTPerPop', 'LemasSwFTFieldOps', 'LemasSwFTFieldPerPop',
        'LemasTotalReq', 'LemasTotReqPerPop', 'PolicReqPerOffic',
        'PolicPerPop', 'RacialMatchCommPol', 'PctPolicWhite',
        'PctPolicBlack', 'PctPolicHisp', 'PctPolicAsian',
        'PctPolicMinor', 'OfficAssgnDrugUnits', 'NumKindsDrugsSeiz',
        'PolicAveOTWorked', 'LandArea', 'PopDens', 'PctUsePubTrans',
        'PolicCars', 'PolicOperBudg', 'LemasPctPolicOnPatr',
        'LemasGangUnitDeploy', 'LemasPctOfficDrugUn', 'PolicBudgPerPop',
        'ViolentCrimesPerPop'
    ]

    # Загрузка данных
    data = pd.read_csv('communities.data', header=None, names=column_names, na_values='?')

    # Целевая переменная - ViolentCrimesPerPop
    X = data.drop('ViolentCrimesPerPop', axis=1)
    y = data['ViolentCrimesPerPop']

    # Удаляем нечисловые и идентификационные столбцы
    columns_to_drop = ['communityname']  # строковый столбец
    X = X.drop(columns_to_drop, axis=1)

    return X, y


def preprocess_data(X, y):
    # Предобработка данных
    # Заполнение пропущенных значений
    numeric_imputer = SimpleImputer(strategy='median')

    # Применяем импутацию ко всем числовым признакам
    X_imputed = numeric_imputer.fit_transform(X)
    X_processed = pd.DataFrame(X_imputed, columns=X.columns)

    # Обработка целевой переменной
    y_processed = y.copy()

    return X_processed, y_processed


# Загрузка и подготовка данных
print("Загрузка данных")
X, y = load_communities_data()
X, y = preprocess_data(X, y)

print(f"Размерность признаков: {X.shape}")
print(f"Размерность целевой переменной: {y.shape}")
print(f"Количество признаков: {X.shape[1]}")

# 1) Разделение на тестовую и обучающие выборки
n_samples = X.shape[0]  # Общее количество наблюдений
indices = np.arange(n_samples)  # Массив индексов от 0 до n_samples-1
np.random.shuffle(indices)  # Перемешиваем индексы случайным образом для того чтобы избежать упорядочности данных

# Применяем перемешанные индексы к X и y
X_shuffled = X.iloc[indices]
y_shuffled = y.iloc[indices]

# Разделяем перемешанные данные
train_size = int(n_samples * 0.8)
X_train = X_shuffled[:train_size]
X_test = X_shuffled[train_size:]
y_train = y_shuffled[:train_size]
y_test = y_shuffled[train_size:]

print(f"Обучающая выборка: {X_train.shape[0]} samples")
print(f"Тестовая выборка: {X_test.shape[0]} samples")

# 2) Обучение модели по линейной регрессии
print("\n--- Линейная регрессия ---")
regressor = LinearRegression().fit(X_train, y_train)

y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

print(f"Коэффициент детерминации (R²) на обучающей выборке: {r2_score(y_train, y_train_pred):.4f}")
print(f"Коэффициент детерминации (R²) на тестовой выборке: {r2_score(y_test, y_test_pred):.4f}")
print(f"MSE на тестовой выборке: {mean_squared_error(y_test, y_test_pred):.6f}")

# Визуализация результатов линейной регрессии
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color="blue", alpha=0.6, label="Фактические vs. Прогноз")
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle='--', linewidth=2,
         label="Идеальный прогноз (y=x)")
plt.xlabel("Истинные значения ViolentCrimesPerPop")
plt.ylabel("Предсказанные значения ViolentCrimesPerPop")
plt.title("Линейная регрессия: Истинные vs. Предсказанные значения")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 3) Полиномиальная регрессия
print("\n--- Полиномиальная регрессия ---")
degrees = range(1, 3)
r2_train_list = []
r2_test_list = []

for degree in degrees:
    print(f"Степень полинома: {degree}")
    pipeline = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("linear_regression", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r2_train_list.append(r2_train)
    r2_test_list.append(r2_test)

    print(f"  R² обучающая: {r2_train:.4f}")
    print(f"  R² тестовая: {r2_test:.4f}")

# Визуализация результатов полиномиальной регрессии
plt.figure(figsize=(8, 5))
plt.plot(degrees, r2_train_list, marker='o', label="Обучающая R²")
plt.plot(degrees, r2_test_list, marker='o', label="Тестовая R²")
plt.xlabel("Степень полинома")
plt.ylabel("R²")
plt.title("Полиномиальная регрессия: Производительность")
plt.legend()
plt.grid(True)
plt.show()

# 4) Ридж-регрессия с полиномиальными признаками
print("\n--- Ридж-регрессия с полиномиальными признаками ---")
degree = 2
alphas = np.logspace(-4, 3, 10)
r2_train_ridge = []
r2_test_ridge = []

for alpha in alphas:
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, max_iter=10000))
    ])

    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    r2_train_ridge.append(r2_score(y_train, y_train_pred))
    r2_test_ridge.append(r2_score(y_test, y_test_pred))

# Визуализация ридж-регрессии
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, r2_train_ridge, marker='o', label="Обучающая R²")
plt.semilogx(alphas, r2_test_ridge, marker='o', label="Тестовая R²")
plt.xlabel("Alpha (коэффициент регуляризации)")
plt.ylabel("R²")
plt.title(f"Ридж-регрессия (Полином степени={degree})")
plt.grid(True)
plt.legend()
plt.show()

# Нахождение лучшего alpha
best_index = np.argmax(r2_test_ridge)
best_alpha = alphas[best_index]
best_r2 = r2_test_ridge[best_index]

print(f"\nНаилучший alpha: {best_alpha:.4f}")
print(f"Наилучший R² на тестовой выборке: {best_r2:.4f}")

# Сравнение всех моделей
print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===")
print(f"Линейная регрессия - Тестовая R²: {r2_score(y_test, y_test_pred):.4f}")
print(f"Полиномиальная регрессия (степень 1) - Тестовая R²: {r2_test_list[0]:.4f}")
print(f"Полиномиальная регрессия (степень 2) - Тестовая R²: {r2_test_list[1]:.4f}")
print(f"Ридж-регрессия (лучшая) - Тестовая R²: {best_r2:.4f}")

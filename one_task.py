from clearml import Task
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes


# Создаем новый эксперимент
task = Task.init(project_name="Lab2_3", task_name="diabetes")

# Загрузка данных
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Параметры для итераций
iterations = [
    {"n_estimators": 50, "max_depth": 10},
    {"n_estimators": 100, "max_depth": 20},
    {"n_estimators": 150, "max_depth": 30},
    {"n_estimators": 200, "max_depth": None},
]

# Логирование метрик для каждой итерации
for i, params in enumerate(iterations):
    # Создание и обучение модели
    rf = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # Оценка модели
    predictions = rf.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Логирование метрик
    task.get_logger().report_scalar(
        title="MSE", series="mse", value=mse, iteration=i + 1
    )
    task.get_logger().report_scalar(title="R2", series="r2", value=r2, iteration=i + 1)

    print(
        f"Iteration {i+1}: n_estimators={params['n_estimators']},\
          max_depth={params['max_depth']}, MSE={mse}, R2={r2}"
    )

# Завершение эксперимента
task.close()

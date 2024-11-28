from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes
from clearml import Task

# Task.add_requirements("scikit-learn", package_version="1.5.2")

@PipelineDecorator.component(cache=True,
                             return_values=['X_train', 'X_test',
                                            'y_train', 'y_test'],
                             task_type=TaskTypes.data_processing)
def load_data():
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    # Загрузка данных
    data = load_diabetes()

    # Разделение данных на тестовую и обучающую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


@PipelineDecorator.component(cache=True, return_values=["model"],
                             task_type=TaskTypes.training)
def random_forest_model(X_train, y_train,
                        n_estimators, max_depth):

    # Импорт необходимых библиотек
    from sklearn.ensemble import RandomForestRegressor

    # Создание и обучение модели
    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    return model


@PipelineDecorator.component(cache=True, return_values=["model"],
                             task_type=TaskTypes.training)
def extra_trees_model(X_train, y_train,
                      n_estimators, max_depth):

    # Импорт необходимых библиотек
    from sklearn.ensemble import ExtraTreesRegressor

    # Создание и обучение модели
    model = ExtraTreesRegressor(n_estimators=n_estimators,
                                max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


@PipelineDecorator.component(cache=True, return_values=['mse', 'r2'],
                             task_type=TaskTypes.qc)
def evaluate_model(X_test, y_test, model):
    # Импорт необходимых библиотек
    from sklearn.metrics import mean_squared_error, r2_score

    # Оценка модели
    prediction = model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    return mse, r2


@PipelineDecorator.pipeline(name='main_pipeline', project='Lab2_3',
                            version='1.0', default_queue='services')
def main_pipeline():
    # Установка параметров для моделей
    parameters = {'n_estimators': 50, 'max_depth': 10}

    # Загрузка данных
    X_train, X_test, y_train, y_test = load_data()

    # Обработка ExtraTreesRegressor и оценка модели
    model = extra_trees_model(X_train, y_train, **parameters)
    mse, r2 = evaluate_model(X_test, y_test, model)
    print(f"MSE: {mse}, R2: {r2}")

    # Обработка RandomForestRegressor и оценка модели
    model = random_forest_model(X_train, y_train, **parameters)
    mse, r2 = evaluate_model(X_test, y_test, model)
    print(f"MSE: {mse}, R2: {r2}")


# Запуск эксперимента
if __name__ == '__main__':
    main_pipeline()

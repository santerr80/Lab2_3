from clearml import PipelineDecorator, TaskTypes, Logger


@PipelineDecorator.component(
    cache=True,
    return_values=["X_train", "X_test", "y_train", "y_test"],
    execution_queue="services",
    task_type=TaskTypes.data_processing,
)
def load_data():
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes

    # Загрузка данных
    data = load_diabetes()

    # Разделение данных на тестовую и обучающую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


@PipelineDecorator.component(
    cache=True, return_values=["model"], task_type=TaskTypes.training
)
def random_forest_model(X_train, y_train, n_estimators, max_depth):

    # Импорт необходимых библиотек
    from sklearn.ensemble import RandomForestRegressor

    # Создание и обучение модели
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    return model


@PipelineDecorator.component(
    cache=True, return_values=["model"], task_type=TaskTypes.training
)
def extra_trees_model(X_train, y_train, n_estimators, max_depth):

    # Импорт необходимых библиотек
    from sklearn.ensemble import ExtraTreesRegressor

    # Создание и обучение модели
    model = ExtraTreesRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    return model


@PipelineDecorator.component(
    cache=True, return_values=["mse", "r2"], task_type=TaskTypes.qc
)
def evaluate_model(X_test, y_test, model):
    # Импорт необходимых библиотек
    from sklearn.metrics import mean_squared_error, r2_score

    # Оценка модели
    prediction = model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    return mse, r2


@PipelineDecorator.pipeline(name="main_pipeline", project="Lab2_3",
                            version="2.0")
def pipeline_logic():
    # Вызов компонента load_data как части пайплайна
    X_train, X_test, y_train, y_test = load_data()

    iterations = [
        {"n_estimators": 50, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 20},
        {"n_estimators": 150, "max_depth": 30},
        {"n_estimators": 200, "max_depth": None},
    ]

    logger = Logger.current_logger()

    for i, params in enumerate(iterations):
        # Создание и обучение модели
        rf = random_forest_model(
            X_train,
            y_train,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
        )

        mse, r2 = evaluate_model(X_test, y_test, rf)
        # Логирование метрик
        filename = "report.txt"

        with open(filename, "a+") as f:
            print(f"Random Forest - Iteration {i+1}:\
                n_estimators={params['n_estimators']},\
                max_depth={params['max_depth']}, MSE={mse}, R2={r2}", file=f)
        f.close()

        logger.report_scalar(
            title="Random Forest MSE", series="mse", value=mse, iteration=i + 1
        )
        logger.report_scalar(
            title="Random Forest R2", series="r2", value=r2, iteration=i + 1
        )

        ext = extra_trees_model(
            X_train,
            y_train,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
        )
        mse, r2 = evaluate_model(X_test, y_test, ext)

        with open(filename, "a+") as f:
            print(f"Extra Trees - Iteration {i+1}:\
                n_estimators={params['n_estimators']},\
                max_depth={params['max_depth']}, MSE={mse}, R2={r2}", file=f)
        f.close()

        # Логирование метрик
        logger.report_scalar(
            title="Extra Trees MSE", series="mse", value=mse, iteration=i + 1
        )
        logger.report_scalar(
            title="Extra Trees R2", series="r2", value=r2, iteration=i + 1
        )


if __name__ == "__main__":
    # Запуск пайплайна
    PipelineDecorator.run_locally()
    pipeline_logic()

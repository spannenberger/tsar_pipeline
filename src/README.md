# Содержание папки src
:+1:    :metal: :metal:    :+1:
----

### Структура папки src
- callbacks - папка с общими колбэками
    - convert_callbacks - папка с колбэками конвертаций
        - onnx_save_callback - конвертация модели в onnx 
        - torchscript_save_callback - конвертация модели в torchscript
    - logger_callbacks - папка с колбэками для логгирования эксперимента, как во время эксперимента, так и после
        - mlflow_image_logger - колбэк с помощью которого происходит логгирование фотографий, моделей, конфигов в mlflow
        - tensorboard_image_logger - колбэк с помощью которого происходит логгирование фотографий в tensorboard
        - mlflow_logger - колбэк для логгирования метрик, лоссов и некоторой информации об обучении в mlflow 
    - save_metric_callback.py - колбэк, который позволяет сохранять переданные в него метрики
- classification - папка с реализацией решения задач multiclass, multilabel классификации
- metric_learning - папка с реализацией решения задач metric_learning
- metrics - папка с кастомными метриками
- datasets - папка с асбтрактной фабрикой датасетов
- nlp - папка с реализацией решения nlp задач: классификации с помощью сиамской сети, обучение языковой модели
- models - папка со всеми моделями, которые можно использовать
    - models_classes - файл с классами моделей для разных задач
    - models_fabrics - файл с фабрикой моделей
    - models - классы разных архитектур моделей
- utils - папка с необходимыми инструментами
- tests - папка с необходимыми тестами
### Структура папок classification, metric_learning, nlp
- callbacks - папка с кастомными колбэками
- \__init__.py - файл, в котором мы инициализируем, импортируем нужные нам инструменты как из каталиста, так и наши кастомные
- dataset.py - файл, в котором мы работаем с полученными данные из датасета
- dataset_fabric.py - фабрика датасетов для разных режимов обучения
- SupervisedRunner.py - Кастомный runner нашего эксперимента
- transform.py - файл-парсер, который применяет указанные аугментации к данным (light, medium, hard)
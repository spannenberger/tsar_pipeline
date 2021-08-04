# Содержание папки tests
:+1:    :metal: :metal:    :+1:
----

### Структура папки tests
- test_configs - конфиги экспериментов с разными колбэками
    - metric_learning -конфиги для задачи метрик лернинга
    - multilabel - конфиги для задачи мультилейбл классификации
    - multiclass - конфиги для задачи мультикласс классификации
- test_config_runner - файл для тестирования конфигов multilabel, multiclass, metric_learning
    - команда запуска тестирования metric learning: ```pytest -v -m test_metric_learning```
    - команда запуска тестирования multilabel: ```pytest -v -m test_multilabel_classification```
    - команда запуска тестирования multiclass: ```pytest -v -m test_multiclass_classification```
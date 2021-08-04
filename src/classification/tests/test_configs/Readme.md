# Содержание папки tests
:+1:    :metal: :metal:    :+1:
----

### Структура папки tests
- test_configs - конфиги экспериментов с разными колбэками
    - multilabel - конфиги для задачи мультилейбл классификации
    - multiclass - конфиги для задачи мультикласс классификации
- test_multiclass_classification - файл с тестированием конфигов задач мультикласс классификации 
    - команда запуска multiclass: ```pytest -v -m test_multiclass_classification```
- test_multilabel_classification - файл с тестированием конфигов задач мультилейбл классификации
    - команда запуска multilabel: ```pytest -v -m test_multiclass_classification```
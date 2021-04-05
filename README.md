# tsar_pipeline
Данный репозиторий содержит решение мультикласс и мультилейбл классификации
:muscle: :muscle: :muscle: :muscle:
----
### Table of Contents
- [User guide](#user-guide)
  * [Структура репозитория](#структура-репозитория)
  * [Инструкция по использования репозитория](#инструкция-по-использования-репозитория)
- [Training run](#training-run)
- [Test in docker](#test-in-docker)
# User guide
### Структура репозитория
- [classifications_shells](#training-run) - папка, содержащая скрипты запуска репозитория
- config - папка с конфигами эксперимента, в которых мы можем изменять: модель, путь до данных, шедулеры, коллбэки и тд
    * [Multiclass](config/classification/multiclass/train_multiclass.yml) - конфиг мультикласс классификации
    * [Multilabel](config/classification/multilabel/train_multilabel.yml) - конфиг мультилейбл класификаци
- [src](src/) - папка с основными файлами проекта, в которую добавляются новые шедулеры, модели, коллбэки и тд
- [docker-compose.yml](#test-in-docker) - конфиг файл для докера
- [model_converter.py](/model_converter.py) - файл для конвертации моделей в форматы torchscript, onnx и проверки корректности преобразованных файлов
- [requirements.txt](/requirements.txt) - файл с библиотеками и инструментами, которые нам нужны в проектах
---
### Инструкция по использования репозитория
- [Multiclass](#Запуск-и-изменение-multiclass-решения)
- [Multilabel](#Запуск-и-изменение-multilabel-решения)
 ### Запуск и изменение multiclass решения
   - Склонировать репозиторий
   -  Запустить команду ```pip install -r requirements.txt```
   -  Для изменения, подключения данных обучения:
       - Изменить файл ```./src/multilabel/dataset.py``` для мультикласса.
       - Изменить в папке ```./config/multiclass/train_multiclass.yml``` файл, прописав новые пути до данных в блоке ```data:```
       - По стандарту данные идут в формате: csv файл с лейблами и названием фото, папки с фотографиями(трейн и валидация)
       - Подготовка данных к эксперименту происходит в ```./src/multiclass/TTASupervisedRunner.py``` в методе get_datasets, 
       чтение данных во время эксперимента происходит в ```dataset.py```
   - Изменение моделей обучения
       - Изменить в файле ```./src/multiclass/__init__.py``` Registry модели, изменив на одну из импортированных
       - Изменить в ```train_multiclass.yml``` файле название модели в блоке ```model:```


 ### Запуск и изменение multilabel решения
   - Склонировать репозиторий
   - Запустить команду ```pip install -r requirements.txt```
   - Для изменения, подключения данных обучения:
       - Изменить файл ```./src/multilabel/dataset.py``` для мультикласса
       - Изменить в папке ```./config/multilabel/train_multilabel.yml``` файл, прописав новые пути до данных в блоке ```data:```
       - По стандарту данные идут в формате: csv файл с лейблами и названием фото, папки с фотографиями(трейн и валидация)
       - Подготовка данных к эксперименту происходит в ```./src/multilabel/TTASupervisedRunner.py``` в методе get_datasets,
       чтение данных во время эксперимента происходит в ```dataset.py```
   - Изменение моделей обучения
       - Изменить в файле ```./src/multilabel/__init__.py``` Registry модели, изменив на одну из импортированных
       - Изменить в ```train_multilabel.yml``` файле название модели в блоке ```model:```



# Training run 
```bash
# To check multiclass pipeline
sh classification_shells/check_multiclass.sh
# To usual multiclass train pipeline
sh classification_shells/train_multiclass.sh


# To check multilabel pipeline
sh classification_shells/check_multilabel.sh
# To usual multilabel train pipeline
sh classification_shells/train_multilabel.sh


# Run tensorflow for visualisation
tensorboard --logdir=logs/ui # for our pipeline
# Run mlflow 
mlfwlow ui

```
# Test in docker
```
# build ur project, u need to do this only once
docker-compose build

# run docker ur container
docker-compose up

# shutdown ur container
docker-compose stop
```
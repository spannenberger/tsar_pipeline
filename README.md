# tsar_pipeline
### Данный репозиторий содержит решение мультикласс, мультилейбл классификации и Metric Learning
----
### Содержание
- [User guide](#user-guide)
  * [Структура репозитория](#структура-репозитория)
  * [Инструкция по использованию репозитория для virtualenv/env](#инструкция-по-использованию-репозитория)
  * [Инструкция по использованию обучения metric learning с помощью LMDB](#использование-lmdb)
- [Информация о конвертации моделей](#информация-о-конвертации-моделей)
- [Информация о моделях](#информация-о-моделях)
- [Training run](#training-run)
- [Docker run](#docker-run)
# User guide
### Структура репозитория
- [classifications_shells](#training-run) - папка, содержащая скрипты запуска решений задач классификации
- [metric_learning_shells](#training-run) - папка, содержащая скрипты запуска решений задач metric learning
- [config](./config) - папка с конфигами эксперимента, в которых мы можем изменять: модель, путь до данных, шедулеры, коллбэки и тд
    * [Multiclass](config/classification/multiclass/train_multiclass.yml) - конфиг мультикласс классификации
    * [Multilabel](config/classification/multilabel/train_multilabel.yml) - конфиг мультилейбл класификаци
    * [MetricLearning](config/metric_learning/train_metric_learning.yml) - конфиг metric learning
- [src](src/) - папка с основными файлами проекта, в которую добавляются новые шедулеры, модели, коллбэки и тд
- [docker-compose.yml](#test-in-docker) - конфиг файл для докера
- [requirements.txt](/requirements.txt) - файл с библиотеками и инструментами, которые нам нужны в проектах
---
### Инструкция по использованию репозитория
- [Multiclass](#запуск-и-изменение-multiclass-решения)
- [Multilabel](#запуск-и-изменение-multilabel-решения)
- [MetricLearing](#запуск-и-изменение-metric-learning-решения)
- [Callbacks](#использование-колбэков-в-пайплайне)
 ### Запуск и изменение multiclass решения
   - Склонировать репозиторий
   -  Запустить команду ```pip install -r requirements.txt```
   -  Для изменения, подключения данных обучения:
       - По стандарту данные идут в формате:
       ```
          train_dataset/
            - images/
              - train/
                train_image_name_1.jpg
                train_image_name_2.jpg
                ...
                train_image_name_N.jpg

              - test/
                test_image_name_1.jpg
                test_image_name_2.jpg
                ...
                test_image_name_N.jpg

           test_metadata.csv
           train_metadata.csv
        ```
        - Структура csv файлов
        ```
        "image_path":
          train_image_name_1.jpg,
          train_image_name_2.jpg,
          ...
          train_image_name_N.jpg,
        "label":
          1,
          0,
          ...
          1
        ```
       - Изменить в папке ```./config/classification/multiclass/train_multiclass.yml``` файл, прописав новые пути до данных в блоке ```data:```
       - Подготовка данных к эксперименту происходит в ```./src/classification/SupervisedRunner.py``` в методе get_datasets класса ```MulticlassRunner```,
       чтение данных во время эксперимента происходит в ```dataset.py```
   - Изменение моделей обучения
       - Изменить в ```train_multiclass.yml``` файле название модели (доступные модели можно посмотреть в таблице([Информация о моделях](#информация-о-моделях))
   - Для отключения колбэков достаточно их закомментировать в config файле
   - Логирование эксперимента в mlflow
       - Изменить в папке ```./config/classification/multiclass/train_multiclass.yml``` файл, прописав новые url и название эксперимента в блоке 
       ```
       loggers:
            mlflow:
        ```
   - Для отключения колбэков достаточно их закомментировать в config файле
   - Для дообучения на своих моделях:
     - Проверить, что данная модель реализована в пайплайне
     - Создать в корне проекта папку ```our_models```
     - Загрузить в данную папку вашу модель в формате .pth с названием файла "best". Пример: ```best.pth```
     - Пример:
     ```
     model:
        _target_: Densenet121 # имя клаccа. Сам класс будет сконструирован в registry по этому имени
        mode: Classification
        num_classes: &num_classes 2
        path: 'our_models/best.pth' # Путь до расположения вашей локальной модели
        is_local: True # True если обучаете локально загруженную модель
        diff_classes_flag: True # True, если есть разница в кол-ве классов
        old_num_classes: 18 # Если diff_classes_flag=True, то указать кол-во классов в предобученной модели
     ```
 ### Запуск и изменение multilabel решения
   - Склонировать репозиторий
   - Запустить команду ```pip install -r requirements.txt```
   - Для изменения, подключения данных обучения:
       - По стандарту данные идут в формате:
       ```
          train_dataset/
            - images/
              - train/
                train_image_name_1.jpg
                train_image_name_2.jpg
                ...
                train_image_name_N.jpg

              - test/
                test_image_name_1.jpg
                test_image_name_2.jpg
                ...
                test_image_name_N.jpg

           test_metadata.csv
           train_metadata.csv
        ```
        - Структура csv файлов
        ```
        "image_path":
          train_image_name_1.jpg,
          train_image_name_2.jpg,
          ...
          train_image_name_N.jpg,
        "class_0":
          1.0,
          0.0,
          ...
          1.0
        "class_1":
          1.0,
          0.0,
          ...
          1.0
        "class_2":
          1.0,
          0.0,
          ...
          1.0
        ```
       - Изменить в папке ```./config/classification/multilabel/train_multilabel.yml``` файл, прописав новые пути до данных в блоке ```data:```
       - Подготовка данных к эксперименту происходит в ```./src/classification/SupervisedRunner.py``` в методе get_datasets класса ```MultilabelRunner```,
       чтение данных во время эксперимента происходит в ```dataset.py```
   - Изменение моделей обучения
       - Изменить в ```train_multilabel.yml``` файле название модели (доступные модели можно посмотреть в таблице([Информация о моделях](#информация-о-моделях)))
   - Логирование эксперимента в mlflow
       - Изменить в папке ```./config/classification/multilabel/train_multilabel.yml``` файл, прописав новые url и название эксперимента в блоке 
       ```
       loggers:
            mlflow:
       ```
   - Для отключения колбэков достаточно их закомментировать в config файле
   - Для дообучения на своих моделях:
     - Проверить, что данная модель реализована в пайплайне
     - Создать в корне проекта папку ```our_models```
     - Загрузить в данную папку вашу модель в формате .pth с названием файла "best". Пример: ```best.pth```
     - Поставить True в нужных пунктах конфига
     - Пример:
     ```
     model:
        _target_: Densenet121 # имя клаccа. Сам класс будет сконструирован в registry по этому имени
        num_classes: &num_classes 18
        mode: Classification
        path: 'our_models/best.pth' # Путь до расположения вашей локальной модели
        is_local: True # True если обучаете локально загруженную модель
        diff_classes_flag: True # True, если есть разница в кол-ве классов
        old_num_classes: 2 # Если diff_classes_flag=True, то указать кол-во классов в предобученной модели
     ```
 ### Запуск и изменение metric learning решения
   - Склонировать репозиторий
   - Запустить команду ```pip install -r requirements.txt```
   - Для изменения, подключения данных обучения:
       - По стандарту данные идут в формате:
       ```
       metric_learning_dataset
          - base
          - train
          - val 
       ```
   - Изменение моделей обучения
     - Изменить в ```train_metric_learning.yml``` файле название модели (доступные модели можно посмотреть в таблице([Информация о моделях](#информация-о-моделях)))
   - Для отключения колбэков достаточно их закомментировать в config файле
   - Логирование эксперимента в mlflow
       - Изменить в папке ```./config/metric_learning/train_metric_learning.yml``` файл, прописав новые url и название эксперимента в блоке 
       ```
       loggers:
            mlflow:
       ```
   - Для дообучения на своих моделях:
     - Проверить, что данная модель реализована в пайплайне
     - Создать в корне проекта папку ```our_models```
     - Загрузить в данную папку вашу модель в формате .pth с названием файла "best". Пример: ```best.pth```
     - Поставить флаг True в ```is_local`` модели
     - Пример:
     ```
      model:
        _target_: MobilenetV3Small # имя клаccа. Сам класс будет сконструирован в registry по этому имени
        mode: MetricLearning
        path: our_models/best.pth # Путь до расположения вашей локальной модели
        is_local: False # True если обучаете локально загруженную модель
     ```

### Использование колбэков в пайплайне
- prunning callback прунит параметры во время и/или после обучения.
:neutral_face:**Стоит отметить, что при использовании данного колбэка при обучении multilabel и metric learninng не будет работать конвертация моделей в onnx и torchscript**:neutral_face:
  ```
    Args:
        pruning_fn: функция из torch.nn.utils.prune module.
            Возможные prunning_fn в пайплайне: 'l1_unstructured', 'random_unstructured', 
            'ln_structured', 'random_structured'
            Смотри документацию pytorch: 
            https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
        amount: количество параметров для обрезки.
            Если с плавающей точкой, должно быть от 0,0 до 1,0 и
            представляют собой долю параметров, подлежащих сокращению.
            Если int, он представляет собой абсолютное число
            параметров для обрезки.
        keys_to_prune: Cписок строк. Определяет
            какой тензор в модулях будет обрезан.
        prune_on_epoch_end: флаг bool определяет вызвать или нет
            pruning_fn в конце эпохи.
        prune_on_stage_end: флаг bool определяет вызвать или нет
            pruning_fn в конце стейджа.
        remove_reparametrization_on_stage_end: если True тогда вся
            перепараметризация pre-hooks и tensors с маской
            будет удален в конце стейджа.
        layers_to_prune: список строк - имена модулей, которые нужно удалить.
            Если не указано ни одного, то будет обрезан каждый модуль в
            модели.
        dim: если вы используете structured pruning method вы должны
            указать dimension.
        l_norm: если вы используете ln_structured вы должны указать l_norm.
  ```

- quantization callback квантизирует модель :neutral_face:
  ```
    Args:
      logdir: путь до папки модели после квантизации
      qconfig_spec (Dict, optional): конфиг квантизации в PyTorch формате.
          Defaults to None.
      dtype (Union[str, Optional[torch.dtype]], optional):
          Тип весов после квантизации.
          Defaults to "qint8".
  ```
- CheckpointCallback - сохраняет n лучших моделей
  - best.pth - лучшая модель за обучение
  - last.pth - модель с последней эпохи
  - stage.1_full.pth, ..., stage.n_full.pth - лучшие n моделей за обучение 

## Использование LMDB
LMDB - Lightning Memory-Mapped Database - программная библиотека, представляющая собой высоко-производительную встроенную транзакционную базу данных, в основе которой - хранилище "ключ-значение". LMDB хранит произвольные пары "ключ-данные" как массивы байтов, поддерживает возможность записи нескольких блоков данных по одному ключу. С помощью LMDB мы можем ускорить процесс обучения в несколько раз. Использовать данную фичу есть возможность при обучении metric learning задач
- Как использовать:
  - У вас должен быть датасет со структурой:
      ```
      metric_learning_dataset
        - base:
        - train:
        - val 
      ```
  - Для создания LMDB датасета потребуется:
    - Перейти в [utils](./src/utils/)
    - Поменять конфиг - ```lmdb_config.yml```. Пример:
      ```
      # Параметры для resize
      width: 224
      height: 224
      channels: 3

      dataset_path: "./test_dataset" # Путь до папки датасета содержащей папки: train, val, base
      num_workers: 0
      batch_size: 32
      pin_memory: True
      output_dataset_path: "./lmdb_test_dataset" # Путь до папки в котором сохранится lmdb датасет
      ```
    - Запустить скрипт [lmdb_writer.py](./src/utils/lmdb_writer.py):
      ```
       - python lmdb_writer.py
      ```
    - По итогу ваш LMDB датасет сохраняется в папку, которую вы указали в конфиге
  - Меняем в конфиге в поле ```data:``` путь до новой папки с LMDB датасетом
  - указываем ```mode: LMDB```
  - Пример:
    ```
    data:
      dataset_path: lmdb_test_dataset # путь до датасета
      train_path: train # путь до папки с тренировочными фотографиями
      base_path: base # путь до папки с фотографиями галлереи
      val_path: val # путь до папки с фотографиями валидации(запроса/query)
      transform_path: config/augmentations/light.yml # Режим аугментаций данных (Возможны: light, medium, hard(./config/augmentations/))
      mode: LMDB # Для использования lmdb датасета указать - LMDB, иначе None
    ```
# Информация о моделях

| model | onnx  | torchscript | embedding_size |
| :---: | :-: | :-: | :-: |
| ResNet18 | True  | True  | 512 |
| ResNet34 | True  | True | 512 |
| ResNet50 | True  | True | 2048 |
| ResNet101 | True  | True | 2048 |
| MobilenetV3Small | False  | True | 576 |
| MobilenetV2 | True  | True | 576 |
| MobilenetV3Large | False  | True | 960 |
| ResNext101_32x8d | True  | True | 2048 |
| ResNext50_32x4d | True  | True | 2048 |
| WideResnet50_2 | True  | True | 2048 |
| WideResnet101_2 | True  | True | 2048 |
| EfficientNetb0 | True  | True | 1280 |
| EfficientNetb3 | True  | True | 1536 |
| EfficientNetb4 | True  | True | 1792 |
| Densenet201 | True  | True | 1920 |
| Densenet169 | True  | True | 1664 |
| Densenet161 | True  | True | 2208 |
| Densenet121 | True  | True | 1024 |
| sberbank-ai/rugpt3medium_based_on_gpt2 | False | False | 1024 | 
| sberbank-ai/rugpt3large_based_on_gpt2 | False | False | 1536 | 
| "sberbank-ai/rugpt3small_based_on_gpt2" | False | False | 768 |
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


# To train metric_learning pipeline
sh metric_learning_shells/train.sh
# To check metric_learning pipeline
sh metric_learning_shells/check.sh


# To train nlp siamese pipeline
nlp_shells/siamese_train.sh
# To check nlp siamse pipeline
sh nlp_shells/siamese_check.sh


# To train gpt language pipeline
sh nlp_shells/gpt_train.sh
# To check gpt language pipeline
sh nlp_shells/gpt_check.sh


# Run tensorflow for visualisation
tensorboard --logdir=logs/ui # for our pipeline
# Run mlflow 
mlflow ui

```
# Docker run 
```
# build ur project, u need to do this only once
docker-compose build

# run docker ur container
docker-compose up

# shutdown ur container
docker-compose stop
```
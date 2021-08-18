# tsar_pipeline
### Данный репозиторий содержит решение мультикласс, мультилейбл классификации и Metric Learning
----
### Содержание
- [User guide](#user-guide)
  * [Структура репозитория](#структура-репозитория)
  * [Инструкция по использования репозитория через CI/CD](./examples/Readme.md)
  * [Инструкция по использованию репозитория для virtualenv/env](#инструкция-по-использованию-репозитория)
- [Использование тритона после обучения](#использование-тритона-после-обучения)
- [Информация о конвертации моделей](#информация-о-конвертации-моделей)
- [Информация о моделях](#информация-о-моделях)
- [Training run](#training-run)
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
- [examples](examples/) - папка с примером и документацией использования репозитория для CI/CD
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
       - Изменить в папке ```./config/multiclass/train_multiclass.yml``` файл, прописав новые пути до данных в блоке ```data:```
   - Изменение моделей обучения
       - Изменить в ```train_multiclass.yml``` файле название модели (доступные модели можно посмотреть в ```src/classification/__init__.py``` в ```Registry(some_model)```) в блоке ```model:```
   - Логирование эксперимента в mlflow
   - Для отключения колбэков достаточно их закомментировать в config файле
    - Для дообучения на своих моделях:
       - Создать .env файл
       ```
       USER=YourWorkEmail@napoleonit.ru
       MLFLOW_TRACKING_URI=
       MLFLOW_S3_ENDPOINT_URL=
       AWS_ACCESS_KEY_ID=
       AWS_SECRET_ACCESS_KEY=
       ```
       - Изменить в папке ```./config/multiclass/train_multiclass.yml``` файл, прописав новые url и название эксперимента в блоке 
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
       - Изменить в папке ```./config/multilabel/train_multilabel.yml``` файл, прописав новые пути до данных в блоке ```data:```
       - Подготовка данных к эксперименту происходит в ```./src/multilabel/TTASupervisedRunner.py``` в методе get_datasets,
       чтение данных во время эксперимента происходит в ```dataset.py```
   - Изменение моделей обучения
       - Изменить в ```train_multilabel.yml``` файле название модели (доступные модели можно посмотреть в ```src/classification/__init__.py``` в ```Registry(some_model)```) в блоке ```model:```
   - Логирование эксперимента в mlflow
       - Создать .env файл
       ```
       USER=YourWorkEmail@napoleonit.ru
       MLFLOW_TRACKING_URI=
       MLFLOW_S3_ENDPOINT_URL=
       AWS_ACCESS_KEY_ID=
       AWS_SECRET_ACCESS_KEY=
       ```
       - Изменить в папке ```./config/multiclass/train_multilabel.yml``` файл, прописав новые url и название эксперимента в блоке 
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
     - Изменить в ```train_metric_learning.yml``` файле название модели (доступные модели можно посмотреть в ```src/metric_learning/__init__.py``` в ```Registry(some_model)```) в блоке ```model:```    
   - Для отключения колбэков достаточно их закомментировать в config файле
   - Логирование эксперимента в mlflow
       - Создать .env файл
       ```
       USER=YourWorkEmail@napoleonit.ru
       MLFLOW_TRACKING_URI=
       MLFLOW_S3_ENDPOINT_URL=
       AWS_ACCESS_KEY_ID=
       AWS_SECRET_ACCESS_KEY=
       ```
       - Изменить в папке ```./config/multiclass/train_multilabel.yml``` файл, прописав новые url и название эксперимента в блоке 
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
# Использование тритона после обучения
В репозитории реализован колбэк для создания минимального тритон конфига модели, который автоматически логируется в mlflow:
```
triton:
  _target_: TritonConfigCreator
  conf_path: "./logs/triton/config.pbtxt" # Путь создания конфига
  mode: multilabel # Выбрать один из режимов multiclass, multilabel
  # instance_group params
  count: 1 # Кол-во используемых gpu
  kind: None # Читай доку тритона
  gpus: [ 0 ] # номера используемых gpu
```
Также в таблице([Информация о моделях](#информация-о-моделях)) можно посмотреть возможность использования модели в тритоне
# Информация о моделях

| model | onnx  | torchscript  | Triton | embedding_size |
| :---: | :-: | :-: | :-: | :-: |
| ResNet18 | True  | True  | True | 512 |
| ResNet34 | True  | True  | True | 512 |
| ResNet50 | True  | True  | True | 2048 |
| ResNet101 | True  | True  | True | 2048 |
| MobilenetV3Small | False  | True  | False | 576 |
| MobilenetV2 | True  | True  | True | 576 |
| MobilenetV3Large | False  | True  | False | 960 |
| ResNext101_32x8d | True  | True  | True | 2048 |
| ResNext50_32x4d | True  | True  | True | 2048 |
| WideResnet50_2 | True  | True  | True | 2048 |
| WideResnet101_2 | True  | True  | True | 2048 |
| EfficientNetb0 | True  | True  | True | 1280 |
| EfficientNetb3 | True  | True  | True | 1536 |
| EfficientNetb4 | True  | True  | True | 1792 |
| Densenet201 | True  | True  | True | 1920 |
| Densenet169 | True  | True  | True | 1664 |
| Densenet161 | True  | True  | True | 2208 |
| Densenet121 | True  | True  | True | 1024 |

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

# Run tensorflow for visualisation
tensorboard --logdir=logs/ui # for our pipeline
# Run mlflow 
mlflow ui

```
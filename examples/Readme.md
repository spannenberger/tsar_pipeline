# Пример использования репозитория с CI/CD 

### Содержание
- [Как это работает?](#объяснение-как-это-работает)
- [Структура папки](#структуры-папки-examples)
- [Инструкция по использованию](#инструкция-по-использованию)
### Объяснение как это работет
---
У нас есть docker image, который лежит в [Registry нашего gitlab](https://gitlab.itnap.ru/machine-learning/computer-vision/tsar_pipeline/container_registry/eyJuYW1lIjoibWFjaGluZS1sZWFybmluZy9jb21wdXRlci12aXNpb24vdHNhcl9waXBlbGluZSIsInRhZ3NfcGF0aCI6Ii9tYWNoaW5lLWxlYXJuaW5nL2NvbXB1dGVyLXZpc2lvbi90c2FyX3BpcGVsaW5lL3JlZ2lzdHJ5L3JlcG9zaXRvcnkvNzU1L3RhZ3M%2FZm9ybWF0PWpzb24iLCJpZCI6NzU1fQ==). Этот image из себя представляет уже собранный докер контейнер, внутри которого уже лежит исходный код из нашего репозитория tsar_pipeline и установлены все зависимости. Благодаря этому теперь достаточно следовать еще более простым инструкциям, чтобы запустить обуение на ваших данных

### Структура папки examples
- your_local_config - папка, внутри которой лежит ваш конфиг для разных задач:
    - multiclass - папка с конфигом для задач мультиклассовой классификации
    - multilabel - папка с конфигом для задач мультилейбл классификации
    - metric_learning - папка с конфигом для задач metric learning 
- docker-compose.yml - конфиг для запуска докер контейнера на основе docker image из gitlab registry
- Также в этой папке должен лежать ваш датасет

### Инструкция по использованию
- Запустить команду: ```docker login reg.gitlab.itnap.ru --username="your_napoleon_username" --password="your_work_password"```
- Запустить команду ```docker pull reg.gitlab.itnap.ru/machine-learning/computer-vision/tsar_pipeline:b220273-release-tests```, где ссылкой должен служить docker image с [Registry gitlab](https://gitlab.itnap.ru/machine-learning/computer-vision/tsar_pipeline/container_registry/eyJuYW1lIjoibWFjaGluZS1sZWFybmluZy9jb21wdXRlci12aXNpb24vdHNhcl9waXBlbGluZSIsInRhZ3NfcGF0aCI6Ii9tYWNoaW5lLWxlYXJuaW5nL2NvbXB1dGVyLXZpc2lvbi90c2FyX3BpcGVsaW5lL3JlZ2lzdHJ5L3JlcG9zaXRvcnkvNzU1L3RhZ3M%2FZm9ybWF0PWpzb24iLCJpZCI6NzU1fQ==), который вам нужен
- Далее следуем пункту [Инструкция по использованию репозитория](/Readme.md)
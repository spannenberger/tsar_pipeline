# tsar_pipeline
This repository contains solution for multiclass and multilabel classification

## Training run 
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
## Test in docker
```
# build ur project, u need to do this only once
docker-compose build

# run docker ur container
docker-compose up

# shutdown ur container
docker-compose stop
```
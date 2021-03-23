# tsar_pipeline
This repository contains solution for multiclass and multilabel classification

## Training run 
```bash
# To check pipeline
sh ./check.sh

# To usual train pipeline
sh ./train.sh

# Run tensorflow for visualisation
tensorboard --logdir=logs/ui # for our pipeline
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
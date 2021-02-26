# cassava-classification
This reposetory contains solution for classification leaf of cassava

## Trining run 
```bash
# To check pipeline
sh ./check.sh

# To usual train pipeline
sh ./train.sh

# To kfold train pipeline
sh ./train_kfold.sh 10

# Run tensorflow for visualisation
tensorboard --logdir=logs # for usual pipeline
tensorboard --logdir=crossval_log # for kfold pipeline
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
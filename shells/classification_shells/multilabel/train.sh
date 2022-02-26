rm -r -f logs/> /dev/null
rm -r -f mlruns/> /dev/null
catalyst-dl run --config config/classification/multilabel/train_multilabel.yml
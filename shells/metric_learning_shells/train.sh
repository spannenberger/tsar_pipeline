rm -r -f logs/> /dev/null
rm -r -f mlruns/> /dev/null
python ./src/utils/lmdb_writer.py
catalyst-dl run --config config/metric_learning/train_metric_learning.yml

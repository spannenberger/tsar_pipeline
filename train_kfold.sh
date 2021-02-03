rm -r -f crossval_log> /dev/null

echo "Num folds is $1\n"
echo "================\n"
if [  "$1" ]
then
    for i in $(seq "$1")
        do
        echo "Train KFold[$i/$1]"
        catalyst-dl run -C config/train.yml --logdir=crossval_log/$i \
                        --stages/data_params/num_folds=$(($1)):int \
                        --stages/data_params/fold_index=$(($i - 1)):int \
                        --stages/callbacks_params/infer/fold_index=$(($i - 1)):int \
                        --stages/callbacks_params/infer/num_folds=$(($1)):int
        python fold_metric.py $i
        echo "================\n"
        done

    echo "Final metric: "
    python metrics.py crossval_log/metrics.csv crossval_log/preds.csv
else
    echo "Set number of folds"
fi

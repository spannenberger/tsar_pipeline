echo "Num folds is $1\n"
echo "================\n"
if [[  "$1" ]]
then
    for i in $(seq "$1")
        do
        echo "Train KFold[$i/$1]"
        catalyst-dl run -C config/train.yml --logdir=crossval_log/$i \
                        --stages/data_params/num_folds=$(($1)):int \
                        --stages/data_params/fold_index=$(($i - 1)):int \
                        --check
        python -c "import torch;c=torch.load(\"crossval_log/$i/checkpoints/best.pth\");print(\"Fold metric:\", c[\"valid_metrics\"][c[\"main_metric\"]]);"
        echo "================\n"
        done

    echo "Final metric: "
else
    echo "Set number of folds"
fi
from sklearn.metrics import accuracy_score
import pandas as pd
import sys

if __name__ == "__main__":
    final_metrics_file = sys.argv[1]
    preds_file = sys.argv[2]
    print(preds_file)
    pred = pd.read_csv(preds_file)
    predict = pred['class_id']
    target = pred['target']
    with open(final_metrics_file, 'w') as f:
        main_metric = accuracy_score(target, predict)
        f.write('name_metric, value\n')
        f.write(f'main_metric, {main_metric}')
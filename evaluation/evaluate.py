import argparse
import jsonlines
import os

from feverous_scorer import feverous_score
from my_utils import compute_metrics

# os.chdir("{dir}/DCUF_code")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--use_gold_verdict', action='store_true', default=False)

    args = parser.parse_args()
    predictions = []

    preds_epoch = []
    golds_epoch = []

    label2idx = {
        "NOT ENOUGH INFO": 0,
        "SUPPORTS": 1,
        "REFUTES": 2,
    }

    with jsonlines.open(os.path.join(args.input_path)) as f:
         for i,line in enumerate(f.iter()):
            if i == 0:
                 continue
            if args.use_gold_verdict:
                line['predicted_label'] = line['label']
                # if not (line["label"] == "NOT ENOUGH INFO"):
                #     continue

            preds_epoch.append(label2idx[line["predicted_label"]])
            golds_epoch.append(label2idx[line["label"]])

            line['evidence'] = [el['content'] for el in line['evidence']]
            for j in range(len(line['evidence'])):
                # line['evidence'][j] = [[el.split('_')[0], el.split('_')[1], '_'.join(el.split('_')[2:])] for el in  line['evidence'][j]]
                line['evidence'][j]= [[el.split('_')[0], el.split('_')[1] if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[1:3]), '_'.join(el.split('_')[2:]) if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[3:])] for el in  line['evidence'][j]]

            try:
                line['predicted_evidence']= [[el.split('_')[0], el.split('_')[1] if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[1:3]), '_'.join(el.split('_')[2:]) if 'table_caption' not in el and 'header_cell' not in el else '_'.join(el.split('_')[3:])] for el in line['predicted_evidence']]
            except:
                a = 1
            # line['predicted_evidence'] = [[el.split('_')[0], el.split('_')[1], '_'.join(el.split('_')[2:])] for el in  line['predicted_evidence']]
            # print(line['predicted_evidence'])
            # line['label'] = line['verdict']
            predictions.append(line)

    # scores = compute_metrics(preds_epoch, golds_epoch)

    print('Feverous scores...')
    strict_score, label_accuracy, precision, recall, f1 = feverous_score(predictions)
    print("feverous score:", strict_score)     #0.5
    print("label accuracy:", label_accuracy)   #1.0
    print("evidence precision:", precision)    #0.833 (first example scores 1, second example scores 2/3)
    print("evidence recall:", recall)          #0.5 (first example scores 0, second example scores 1)
    print("evidence f1:", f1)                  #0.625

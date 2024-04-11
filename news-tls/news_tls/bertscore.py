from datetime import datetime
import os
import json
import argparse
from bert_score import score
from pathlib import Path
from tilse.data.timelines import Timeline as TilseTimeline
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.evaluation import rouge
from news_tls.data import Timeline
from pprint import pprint

def get_scores(metric_desc, pred_tl, groundtruth, evaluator):

    if metric_desc == "concat":
        return evaluator.evaluate_concat(pred_tl, groundtruth)
    elif metric_desc == "agreement":
        return evaluator.evaluate_agreement(pred_tl, groundtruth)
    elif metric_desc == "align_date_costs":
        return evaluator.evaluate_align_date_costs(pred_tl, groundtruth)
    elif metric_desc == "align_date_content_costs":
        return evaluator.evaluate_align_date_content_costs(
            pred_tl, groundtruth)
    elif metric_desc == "align_date_content_costs_many_to_one":
        return evaluator.evaluate_align_date_content_costs_many_to_one(
            pred_tl, groundtruth)

def evaluate_dates(pred, ground_truth):
    pred_dates = pred.get_dates()
    ref_dates = ground_truth.get_dates()
    shared = pred_dates.intersection(ref_dates)
    n_shared = len(shared)
    n_pred = len(pred_dates)
    n_ref = len(ref_dates)
    prec = n_shared / n_pred
    rec = n_shared / n_ref
    if prec + rec == 0:
        f_score = 0
    else:
        f_score = 2 * prec * rec / (prec + rec)
    return {
        'precision': prec,
        'recall': rec,
        'f_score': f_score,
    }

def zero_scores():
    return {'f_score': 0., 'precision': 0., 'recall': 0.}

def get_average_bert_results(tmp_results):
    ret = zero_scores()
    for bert_res in tmp_results:
        metrics = [m for m in ret.keys() if m != 'f_score']
        for m in metrics:
            ret[m] += bert_res[m]
    n = len(tmp_results)
    for k in ['precision', 'recall']:
        ret[k] /= n
    prec = ret['precision']
    rec = ret['recall']
    if prec + rec == 0:
        ret['f_score'] = 0.
    else:
        ret['f_score'] = (2 * prec * rec) / (prec + rec)
    return ret

def get_average_results(tmp_results):
    rouge_1 = zero_scores()
    rouge_2 = zero_scores()
    bert_scores = zero_scores()
    date_prf = zero_scores()
    for rouge_res, date_res, bert_res, _ in tmp_results:
        metrics = [m for m in date_res.keys() if m != 'f_score']
        for m in metrics:
            rouge_1[m] += rouge_res['rouge_1'][m]
            rouge_2[m] += rouge_res['rouge_2'][m]
            date_prf[m] += date_res[m]
            bert_scores[m] += bert_res[m]
    n = len(tmp_results)
    for result in [rouge_1, rouge_2, date_prf, bert_scores]:
        for k in ['precision', 'recall']:
            result[k] /= n
        prec = result['precision']
        rec = result['recall']
        if prec + rec == 0:
            result['f_score'] = 0.
        else:
            result['f_score'] = (2 * prec * rec) / (prec + rec)
    return rouge_1, rouge_2, date_prf, bert_scores

def bertscore(ref, pred):
    print('computing BertScore...\n')
    P, R, F1 = score(pred, ref, lang='en', verbose=True)
    result = {'precision': P.mean().item(), 'recall': R.mean().item(), 'F1': F1.mean().item()}
    print('BERTScore: ', result)
    return result

def concatenate(tl):
    ret = []
    for item in tl:
        content = ' '.join(item[1])
        ret.append(content)
    return ' '.join(ret)

def main(args):
    results = []
    metric = 'align_date_content_costs_many_to_one'
    #metric = 'concat'
    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])
    file_dir = Path(args.dir)
    for f in os.listdir(file_dir):
        print('evaluating {}......'.format(f))
        with open(file_dir / f, "r") as fp:
            timelines = json.load(fp)
        ref_timeline_, pred_timeline_ = timelines

        pred_timeline = {datetime.strptime(t, "%Y-%m-%d %H:%M:%S").date(): s for (t, s) in pred_timeline_}
        ref_timeline = Timeline([[datetime.strptime(t, "%Y-%m-%d"), s] for (t, s) in ref_timeline_])

        pred_timeline = TilseTimeline(pred_timeline)
        sys_len = len(pred_timeline.get_dates())
        ground_truth = TilseGroundTruth([TilseTimeline(ref_timeline.date_to_summaries)])

        rouge_scores = get_scores(metric, pred_timeline, ground_truth, evaluator)
        date_scores = evaluate_dates(pred_timeline, ground_truth)

        concat_ref = concatenate(ref_timeline_)
        concat_pred = concatenate(pred_timeline_)
        bert_scores = bertscore([concat_ref], [concat_pred])
        print(rouge_scores)
        results.append((rouge_scores, date_scores, bert_scores, pred_timeline_))

    avg_results = get_average_results(results)
    output_dir = Path(args.output)
    with open(output_dir/args.name, 'w') as fp:
        fp.write(str(avg_results))
        fp.write('\n')
        fp.write(str(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        required=True,
                        help='the directory of the generated output and the reference')
    parser.add_argument('--output',
                        required=True,
                        help='the directory of results')
    parser.add_argument('--name',
                        required=True,
                        help='the name of results file')
    main(parser.parse_args())
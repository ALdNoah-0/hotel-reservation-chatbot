"""Reproduce NLP results. Exposed notebook data is a regression benchmark."""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from sklearn.metrics import classification_report, precision_recall_fscore_support
from hotel_chatbot.intents import IntentRecognizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'evaluation')
    args = parser.parse_args()
    dataset = json.loads((ROOT/'data'/'regression_holdout.json').read_text(encoding='utf-8'))
    examples = dataset['examples']
    recognizer = IntentRecognizer()
    rows = []
    for item in examples:
        ml = recognizer.predict_ml(item['text'])
        hybrid = recognizer.predict(item['text'])
        rows.append({'text':item['text'],'expected':item['intent'],'classifier':ml.intent,
                     'hybrid':hybrid.intent,'method':hybrid.source,'score':round(hybrid.score,4)})
    truth = [r['expected'] for r in rows]
    labels = sorted(set(truth))
    def metrics(key):
        guesses = [r[key] for r in rows]
        correct = sum(a==b for a,b in zip(truth,guesses))
        answered = sum(p!='fallback' for p in guesses)
        precision,recall,f1,_ = precision_recall_fscore_support(truth,guesses,labels=labels,average='macro',zero_division=0)
        return {'correct':correct,'incorrect':len(rows)-correct,'accuracy':correct/len(rows),
                'macro_precision':float(precision),'macro_recall':float(recall),'macro_f1':float(f1),
                'abstentions':len(rows)-answered,'coverage':answered/len(rows),
                'selective_accuracy':correct/answered if answered else 0,
                'per_class':classification_report(truth,guesses,labels=labels+['fallback'],output_dict=True,zero_division=0)}
    report = {'dataset_status':dataset['provenance'],'sample_count':len(rows),
              'training_examples':len(recognizer.examples),'generated_at':datetime.now(timezone.utc).isoformat(),
              'historical_notebook':{'templated_accuracy':.9423,'original_holdout_accuracy':.4688},
              'classifier':metrics('classifier'),'hybrid':metrics('hybrid'),'predictions':rows,
              'limitations':['Benchmark was inspected before implementation; this is not independent generalization evidence.',
                             'No human usability study or real hotel integration was conducted.',
                             'Abstentions count as incorrect in ordinary accuracy. Rule scores are not calibrated probabilities.']}
    report['independent'] = independent(recognizer)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (args.output_dir/'errors.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(r for r in rows if r['hybrid']!=r['expected'])
    for label in ('classifier','hybrid'):
        score = report[label]
        print(f"{label}: {score['correct']}/{len(rows)} ({score['accuracy']:.2%}), macro-F1 {score['macro_f1']:.4f}, abstentions {score['abstentions']}")
    print('Exposed notebook regression benchmark; not an independent accuracy guarantee.')
    ind = report['independent']
    print(f"Independent test ({ind['intent_count']} intent messages, never used for training or tuning):")
    for label in ('classifier','hybrid'):
        score = ind[label]
        print(f"  {label}: {score['correct']}/{ind['intent_count']} ({score['accuracy']:.2%}), macro-F1 {score['macro_f1']:.4f}, abstentions {score['abstentions']}")
    print(f"  FAQ topics ({ind['faq_count']} questions): hybrid {ind['faq_hybrid_accuracy']:.2%}, retrieval only {ind['faq_retrieval_accuracy']:.2%}")
    print(f'Results written to {args.output_dir}')


def independent(recognizer):
    """Score the held-out set written before the MVP expansion; it is never used for fitting."""
    data = json.loads((ROOT/'data'/'independent_test.json').read_text(encoding='utf-8'))
    items = data['intent_examples']
    truth = [i['intent'] for i in items]
    rows = [{'text':i['text'],'expected':i['intent'],'classifier':recognizer.predict_ml(i['text']).intent,
             'hybrid':recognizer.predict(i['text']).intent} for i in items]
    labels = sorted(set(truth))
    def score(key):
        guesses = [r[key] for r in rows]
        correct = sum(a==b for a,b in zip(truth,guesses))
        _,_,f1,_ = precision_recall_fscore_support(truth,guesses,labels=labels,average='macro',zero_division=0)
        return {'correct':correct,'accuracy':correct/len(rows),'macro_f1':float(f1),'abstentions':sum(g=='fallback' for g in guesses)}
    faq = []
    for item in data['faq_examples']:
        hybrid = recognizer.predict(item['text'])
        retrieved,similarity = recognizer.faq.match(item['text'])
        faq.append({'text':item['text'],'expected':item['topic'],'hybrid':hybrid.topic,'retrieval':retrieved,'similarity':round(similarity,4)})
    return {'provenance':data['provenance'],'intent_count':len(rows),'faq_count':len(faq),
            'classifier':score('classifier'),'hybrid':score('hybrid'),
            'faq_hybrid_accuracy':sum(f['hybrid']==f['expected'] for f in faq)/len(faq),
            'faq_retrieval_accuracy':sum(f['retrieval']==f['expected'] for f in faq)/len(faq),
            'predictions':rows,'faq_predictions':faq}


if __name__=='__main__':
    main()

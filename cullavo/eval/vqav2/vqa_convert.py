
from tqdm import tqdm
import json
from cullavo.eval.vqav2.vqaEval import EvalAIAnswerProcessor
with open('vqav2_test_results.json') as f:
    json_object = json.load(f)
ai = EvalAIAnswerProcessor()
new_json = []

for x in tqdm(json_object):
    new_json.append({'question_id': x['question_id'], 'answer': ai(x['answer'])})
    
with open('new.json', 'w') as f:
    json.dump(new_json, f)
import json
from tqdm import tqdm
from evaluation import evaluate

mode = 'test'
data = json.load(open(f'covid-qa/covid-qa-{mode}.json'))['data']
print ('{mode} data size: ',len(data))


qac_triples = [[],[],[]]

for did, document in enumerate(data):
    context = document['paragraphs'][0]['context']
    qas = document['paragraphs'][0]['qas']
    
    for qid, qa in enumerate(qas):
        qac_triples[0].append(context)
        qac_triples[1].append(qa['question'])
        
        cur_ans = []
        for ans in qa['answers']:
            cur_ans.append(ans['text'])
        qac_triples[2].append(cur_ans)
    assert len(qac_triples[0])==len(qac_triples[1])==len(qac_triples[2])
print (len(qac_triples[0]))



from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

pred_answers = []

for i,(context,ques,ans) in tqdm(enumerate(zip(qac_triples[0],qac_triples[1],qac_triples[2]))):
    qa_input = {'question':ques, 'context':context}
    pred_ans = nlp(qa_input)
    pred_answers.append(pred_ans['answer'])


with open('pred_zeroshot.txt','w') as f:
    for i in pred_answers:
        f.write(i+'\n')

print (evaluate(qac_triples[2],pred_answers,f'{mode}-result.txt'))




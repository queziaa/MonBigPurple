# import random
from tqdm import tqdm
# from MonBigTool import IS_levenshtein_distance_and_operations
# from MonBigTool import levenshtein_distance_and_operations
from MonBigTool import colon,process_list
from MonBigTool import MonBigTool,MASKmodel 
# WordsDict = monBigTool.getWordsDict()
# MODELNAME = 'tugstugi/bert-large-mongolian-uncased'
MODELNAME = './mongolian'

from transformers import BertForTokenClassification, AutoTokenizer

class fineTuningClass():
    def __init__(self,fineT,tokenizer):
        self.tokenizer = tokenizer
        # 加载模型和分词器
        self.model = BertForTokenClassification.from_pretrained(fineT)
        # model = BertForTokenClassification.from_pretrained('./RRRRRRRRRRR')
        # self.tokenizer = AutoTokenizer.from_pretrained(MODELNAME, use_fast=False)

    def predict(self,sen):
        sentence = ' '.join(sen)
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(-1).tolist()
        # print(predictions)
        tokenizer = AutoTokenizer.from_pretrained(MODELNAME, use_fast=False)
        sen = sentence.split()
        Tindex = 0
        SecondaryTreatment = []
        
        for j in range(len(sen)):
            aAaaa = tokenizer.tokenize(sen[j])
            # print(sen[j],len(aAaaa),predictions[0][Tindex:Tindex+len(aAaaa)])
            if all(v == 0 for v in predictions[0][Tindex:Tindex+len(aAaaa)]) == False:
                SecondaryTreatment.append(j)
            Tindex = Tindex + len(aAaaa)
        return SecondaryTreatment
    



class MonBigPurple():
    def __init__(self,fineTuningClass = None,fineFile = None):
        self.WINDOW = 3
        self.mASKmodel = MASKmodel(MODELNAME)
        self.monBigTool = MonBigTool()
        self.MASK = self.monBigTool.getMASK()
        self.fineTuningClass = fineTuningClass
        # if self.fineTuningClass != None:
            # self.fineTuningClasspredict = fineTuningClass(fineFile,self.mASKmodel.tokenizer)


    
    def predict(self,fsen):

        # inode,tsen,fsen,sen,freq,pairs = monBigTool.Decode(i)


        # SecondaryTreatment = [0]
        # if self.fineTuningClass != None:
        #     SecondaryTreatment = self.fineTuningClasspredict.predict(fsen)
        # # else:
        # temp = self.mASKmodel.tomask(fsen,self.WINDOW,False)
        # for i in range(len(fsen)):
        #     if not fsen[i].isalpha():
        #         continue
        #     if fsen[i] in temp[i] :
        #         continue
        #     if self.monBigTool.mysterious(fsen[i]):
        #         continue
        #     # if i in inode:
        #     #     LOSS_O_T += 1
        #     #     inode.remove(i)
        #     # else:
        #     #     LOSS_O_F += 1
        #     SecondaryTreatment.append(i)

        SecondaryTreatment = [i for i in range(len(fsen))]

        OUT = []
        print(SecondaryTreatment)
        for i in SecondaryTreatment:
            candidate,score = self.monBigTool.FuzzySearch(fsen[i])
            Target = fsen[i]
            TTEMP = self.mASKmodel.hybridPrediction(fsen,i,3,candidate,score)
            if len(TTEMP) == 0:
                continue
            OUT.append({'i':i,'f':Target,'t':TTEMP[0]})
        return OUT



te = MonBigPurple(fineTuningClass,'./results_A/checkpoint-5000')
test_spell = open("test_spell_error.txt", "r")
test_spell = test_spell.read().split("\n")
OUT = []


for i in tqdm(range(len(test_spell))):
    temp = process_list(test_spell[i])
    temp = colon(temp)
    predict = te.predict(temp)
    for j in predict:
        t = j['t']
        ii = j['i']

        temp[ii] = t
    temp = ' '.join(temp)
    temp = temp.replace(' : ',': ')
    OUT.append(temp)
    print('-----------------')    
    print(temp)
    print(test_spell[i])
    print('-----------------')    

with open('test_spell_error_out_EEEEE.txt', 'w') as f:
    for item in OUT:
        f.write("%s\n" % item)
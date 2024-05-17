# import random
from tqdm import tqdm
# from MonBigTool import IS_levenshtein_distance_and_operations
# from MonBigTool import levenshtein_distance_and_operations
from MonBigTool import colon,process_list
from MonBigTool import MonBigTool,MASKmodel 
# WordsDict = monBigTool.getWordsDict()
# MODELNAME = 'tugstugi/bert-large-mongolian-uncased'
MODELNAME = '.mongolian'

class MonBigPurple():
    def __init__(self):
        self.WINDOW = 3
        self.mASKmodel = MASKmodel(MODELNAME)
        self.monBigTool = MonBigTool()
        self.MASK = self.monBigTool.getMASK()

        # self.LOSS_PScore = 0
        # self.LOSS_PError = 0
        # self.LOSS_PMiss = 0
        # self.LOSS_PTrue = 0
        # self.LOSS_PTrue_errFix = 0
        # self.LOSS_PTrue_succFix = 0
        # self.LOSS_O_T = 0
        # self.LOSS_O_F = 0
        # self.LOSS_O_MISS = 0
    
    def predict(self,fsen):

        # inode,tsen,fsen,sen,freq,pairs = monBigTool.Decode(i)

        # ########################################
        # # 得到词对 信息  ########################
        # for g in pairs:
        #     if not g[0].isalpha():
        #         continue


        #     # te = monBigTool.letterSim(g[0],g[1])
        #     # if te < 0.8:
        #         # print(g[0],g[1],te)
        # ########################################


        temp = self.mASKmodel.tomask(fsen,self.WINDOW,False)
        SecondaryTreatment = []
        for i in range(len(fsen)):
            if not fsen[i].isalpha():
                continue
            if fsen[i] in temp[i] :
                continue
            if self.monBigTool.mysterious(fsen[i]):
                continue

            # if i in inode:
            #     LOSS_O_T += 1
            #     inode.remove(i)
            # else:
            #     LOSS_O_F += 1
            SecondaryTreatment.append(i)

        
        # LOSS_O_MISS += len(inode)
        # print('假----',' '.join(fsen),'----')
        # print('真----',' '.join(tsen),'----')
        # print('MASK----',' '.join(sen),'----')
        # print('怀疑----',SecondaryTreatment,'----',inode)
        OUT = []
        for i in SecondaryTreatment:
            candidate,score = self.monBigTool.FuzzySearch(fsen[i])
            Target = fsen[i]
            TTEMP = self.mASKmodel.hybridPrediction(fsen,i,3,candidate,score)
            if len(TTEMP) == 0:
                continue
            OUT.append({'i':i,'f':Target,'t':TTEMP[0]})
        return OUT
            # TTEMP = candidate
        #     print('目标',Target,':  候选',candidate)
        #     print('结果:',TTEMP)


        #     if len(TTEMP) == 0:
        #         continue

        #     if fsen[i] == TTEMP[0]:
        #         continue
        
        #     if i not in inode:
        #         LOSS_PError += 1
        #         print('*ERR'*20)
        #         print(Target,TTEMP)
        #     elif i in inode:
        #         LOSS_PTrue += 1
        #         inode.remove(i)
        #         if tsen[i] == TTEMP[0]:
        #             print('*SUCFix'*20)
        #             LOSS_PTrue_succFix += 1
        #         else:
        #             print('*ERRFix'*20)
        #             LOSS_PTrue_errFix += 1
        #         print(Target,TTEMP)
        
        # LOSS_PMiss += len(inode)
        # for i in inode:
        #     print('@MISS'*20)
        #     print('----',' '.join(fsen),'----')
        #     print('----',' '.join(tsen),'----')
        #     print('----',' '.join(sen),'----')
        #     print(fsen[i])
        #     print(fsen[i])

        # if COUNT % 50 == 0:
        # # print('-----------------')
        #     print('O_T:',LOSS_O_T)
        #     print('O_F:',LOSS_O_F)
        #     print('O_MISS:',LOSS_O_MISS)
        # print('PError:',LOSS_PError)
        # print('PMiss:',LOSS_PMiss)
        # print('PTrue:',LOSS_PTrue)
        # print('PTrue_succFix:',LOSS_PTrue_succFix)
        # print('PTrue_errFix:',LOSS_PTrue_errFix)
te = MonBigPurple()
test_spell = open("test_spell_error.txt", "r")
test_spell = test_spell.read().split("\n")
OUT = []


for i in tqdm(range(len(test_spell))):
    temp = process_list(test_spell[i])
    temp = colon(temp)
    predict = te.predict(temp)
    for j in predict:
        t = j['t']
        f = j['f']
        ii = j['i']
        if temp[ii] != f:
            print('ERROR@@@@@@@@@@@@@@@@@@@')
            print(temp,i,f,t)
            print('ERROR@@@@@@@@@@@@@@@@@@@')
        temp[ii] = t
    temp = ' '.join(temp)
    temp = temp.replace(' : ',': ')
    OUT.append(temp)
    print('-----------------')    
    print(temp)
    print(test_spell[i])
    print('-----------------')    

with open('test_spell_error_out.txt', 'w') as f:
    for item in OUT:
        f.write("%s\n" % item)
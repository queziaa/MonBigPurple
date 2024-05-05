import pickle
import math
import torch
from tqdm import tqdm

class Mondata:
    def __init__(self):
        print("Mondata")
    
    def cleanse(sen):


        temp = sen.split(" ")
        sen = []
        for j in range(len(temp)):
            # print(temp)
            # print('*',temp[j],'*')
            if len(temp[j]) == 0:
                continue
            if temp[j][-1] == ":":
                sen.append(temp[j][:-1])
                sen.append(":")
            else:
                sen.append(temp[j])

        for j in range(len(sen)):
            # 如果单词含有“:”且不是最后一个元素
            if sen[j].find(":") != -1 and sen[j][-1] != ":":
                # sen[j]位置赋予删除“:”后的单词
                sen[j] = sen[j].replace(":", "")
                # sen[j]之后插入“:”
                sen.insert(j+1, ":")



        tol = 0  #79791
        indexx = 0
        tempDict = {}
        tempDict['sen'] = ''
        tempDict['word'] = []
        templist = []
        if indexx != 0:
            # 深拷贝
            tempDict['sen'] = clean[i].copy()
            for j in range(len(templist)):
                tempDict['sen'][templist[j]] = '<>'
            return tempDict






class MonBigTool:
    def __init__(self):
        self.MASK = None
        self.wordsDict = None
        # self.MASKmodel = mASKmodel
        # self.pipe = pipe
        # self.device = device
        self.nroW = None
        self.errW = None


    def getMASK(self):
        if self.MASK is None:
            with open('mask.pickle', 'rb') as f:
                self.MASK = pickle.load(f)
        return self.MASK
    def getWordsDict(self):
        if self.wordsDict is None:
            with open('wordsDict.pickle', 'rb') as f:
                self.wordsDict = pickle.load(f)
        return self.wordsDict
    def getNroW(self):
        if self.nroW is None:
            self.nroW = []
            for i in self.MASK:
                for j in range(int(len(i['word']) / 2)):
                    if i['word'][j * 2] not in self.nroW:
                        self.nroW.append(i['word'][j * 2])
                    if i['word'][j * 2 + 1] not in self.errW:
                        self.errW.append(i['word'][j * 2 + 1])
        return self.nroW
    
    def getErrW(self):
        if self.errW is None:
            self.errW = []
            for i in self.MASK:
                for j in range(int(len(i['word']) / 2)):
                    if i['word'][j * 2] not in self.nroW:
                        self.nroW.append(i['word'][j * 2])
                    if i['word'][j * 2 + 1] not in self.errW:
                        self.errW.append(i['word'][j * 2 + 1])
        return self.errW
    
    def getWordsDictNum(self,word):
        wordDict = self.getWordsDict()
        if word in wordDict:
            return wordDict[word]
        else:
            return 0
        
    def Decode(self, ma_i):
        Tsum = 0
        inode = []
        tsen = ma_i['sen'].copy()
        fsen = ma_i['sen'].copy()
        freq = []
        wordDict = self.getWordsDict()
        for index in range(len(ma_i['sen'])):
            if ma_i['sen'][index] == '<>':
                ma_i['sen'][index] = '[MASK]'
                Tsum += 1
                inode.append(index)
                temp = (Tsum-1) * 2
                tsen[index] = ma_i['word'][temp]
                fsen[index] = ma_i['word'][temp+1]

                if fsen[index] in wordDict:
                    freq.append(wordDict[fsen[index]])
                else:
                    freq.append(0)
            else:
                if ma_i['sen'][index] in wordDict:
                    freq.append(wordDict[ma_i['sen'][index]])
                else:
                    freq.append(0)
        pairs = [(ma_i['word'][i], ma_i['word'][i+1]) for i in range(0, len(ma_i['word']), 2)]
        sen = ma_i['sen']
        return inode,tsen,fsen,sen,freq,pairs
    
    def FuzzySearch(self,word):
        llll = len(word)
        out = []
        wordDict = self.getWordsDict()
        for i in wordDict:
            if IS_levenshtein_distance_and_operations(word,i,llll):
                out.append(i)
        return out





    # def tomask(self,sen,i):
    #     input_ = sen.copy()
    #     input_[i] = '[MASK]'
    #     input_ = " ".join(input_)
    #     # input_ = input_.to(self.device)
    #     return self.pipe(input_)
    
    # def tomask2_ishit(self,fsen,tsen,i):
    #     te = self.tomask(fsen,i)
    #     for maski in te:
    #         if maski['token_str'] == tsen[i]:
    #             return maski['score']
    #     return -1
    
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

class MASKmodel:
    def __init__(self):




        # self.tokenizer = AutoTokenizer.from_pretrained("hfl/cino-large-v2")
        # self.model = AutoModelForMaskedLM.from_pretrained("hfl/cino-large-v2")


        # self.tokenizer = AutoTokenizer.from_pretrained("tugstugi/bert-base-mongolian-cased")
        # self.model = AutoModelForMaskedLM.from_pretrained("tugstugi/bert-base-mongolian-cased")


        # self.tokenizer = AutoTokenizer.from_pretrained("tugstugi/bert-base-mongolian-uncased")
        # self.model = AutoModelForMaskedLM.from_pretrained("tugstugi/bert-base-mongolian-uncased")


        # self.tokenizer = AutoTokenizer.from_pretrained("tugstugi/bert-large-mongolian-cased")
        # self.model = AutoModelForMaskedLM.from_pretrained("tugstugi/bert-large-mongolian-cased")


        # self.tokenizer = AutoTokenizer.from_pretrained("tugstugi/bert-large-mongolian-uncased")
        # self.model = AutoModelForMaskedLM.from_pretrained("tugstugi/bert-large-mongolian-uncased")


        # self.tokenizer = AutoTokenizer.from_pretrained("phjhk/hklegal-xlm-r-base")
        # self.model = AutoModelForMaskedLM.from_pretrained("phjhk/hklegal-xlm-r-base")


        # self.tokenizer = AutoTokenizer.from_pretrained("Twitter/twhin-bert-large")
        # self.model = AutoModelForMaskedLM.from_pretrained("Twitter/twhin-bert-large")



        # self.tokenizer = AutoTokenizer.from_pretrained("3ebdola/Dialectal-Arabic-XLM-R-Base")
        # self.model = AutoModelForMaskedLM.from_pretrained("3ebdola/Dialectal-Arabic-XLM-R-Base")


        # self.tokenizer = AutoTokenizer.from_pretrained("phjhk/hklegal-xlm-r-large")
        # self.model = AutoModelForMaskedLM.from_pretrained("phjhk/hklegal-xlm-r-large")


        # self.tokenizer = AutoTokenizer.from_pretrained("phjhk/hklegal-xlm-r-large-t")
        # self.model = AutoModelForMaskedLM.from_pretrained("phjhk/hklegal-xlm-r-large-t")


        # self.tokenizer = AutoTokenizer.from_pretrained("plAIground/xlmr-bert-multilingual-base-merge")
        # self.model = AutoModelForMaskedLM.from_pretrained("plAIground/xlmr-bert-multilingual-base-merge")

# 1200
        # self.tokenizer = AutoTokenizer.from_pretrained("hfl/cino-base-v2")
        # self.model = AutoModelForMaskedLM.from_pretrained("hfl/cino-base-v2")
# 1500
        # self.tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-multilingual-cased")
        # self.model = AutoModelForMaskedLM.from_pretrained("distilbert/distilbert-base-multilingual-cased")
# 1500
        # self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
        # self.model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-multilingual-cased")
    # 200-700
        # self.tokenizer = AutoTokenizer.from_pretrained("bayartsogt/albert-mongolian")
        # self.model = AutoModelForMaskedLM.from_pretrained("bayartsogt/albert-mongolian")
            # 200 -1000
        # self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base", use_fast=False)
        # self.model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base")

#       蒙古人 发表 模型 有论文  2000-3500
        self.tokenizer = AutoTokenizer.from_pretrained('tugstugi/bert-large-mongolian-uncased', use_fast=False)
        self.model = AutoModelForMaskedLM.from_pretrained('tugstugi/bert-large-mongolian-uncased')

        # pipe = pipeline(task="fill-mask", model=model, tokenizer=tokenizer,device= (0 if torch.cuda.is_available() else -1))
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)



    def tomask_hid(self,_input):
        # 使用分词器将输入文本转换为模型可以接受的格式
        inputs = self.tokenizer(' '.join(_input), return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        # 使用模型进行预测
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.logits
                

    def token_count(self,_input):
        return [len(self.tokenizer.tokenize(word)) for word in _input]

    def tomask(self,_input,predQuery,model):

        # 使用分词器将输入文本转换为模型可以接受的格式
        inputs = self.tokenizer(' '.join(_input), return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        # 使用模型进行预测
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取每个词对应的token数量
        word_token_counts = self.token_count(_input)

        out = []
        token_index = 1
        for count in word_token_counts:
            # 只考虑每个词的第一个token

            predicted_token_ids = torch.topk(outputs.logits[..., token_index, :], predQuery, dim=-1).indices[0]
            predicted_scores = torch.topk(outputs.logits[..., token_index, :], predQuery, dim=-1).values[0]



            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_token_ids)
            predicted_tokens = [token.replace("▁", "") for token in predicted_tokens]

   
            if model == True:
                if count > 1:
                    # 如果词有多个token，取最后一个token的预测结果
                    predicted_token_ids_two = torch.topk(outputs.logits[..., token_index + 1, :], predQuery, dim=-1).indices[0]
                    predicted_scores_two = torch.topk(outputs.logits[..., token_index + 1, :], predQuery, dim=-1).values[0]
                    predicted_tokens_two = self.tokenizer.convert_ids_to_tokens(predicted_token_ids_two)
                    predicted_tokens_two = [token.replace("▁", "") for token in predicted_tokens_two]
                    predicted_tokens,predicted_scores = cross(predicted_tokens,predicted_tokens_two,predicted_scores,predicted_scores_two)
                    
                if count == 3:
                    predicted_token_ids_three = torch.topk(outputs.logits[..., token_index + 2, :], predQuery, dim=-1).indices[0]
                    predicted_scores_three = torch.topk(outputs.logits[..., token_index + 2, :], predQuery, dim=-1).values[0]
                    predicted_tokens_three = self.tokenizer.convert_ids_to_tokens(predicted_token_ids_three)
                    predicted_tokens_three = [token.replace("▁", "") for token in predicted_tokens_three]
                    predicted_tokens,predicted_scores = cross(predicted_tokens,predicted_tokens_three,predicted_scores,predicted_scores_three)

                if count == 4:
                    predicted_token_ids_four = torch.topk(outputs.logits[..., token_index + 3, :], predQuery, dim=-1).indices[0]
                    predicted_scores_four = torch.topk(outputs.logits[..., token_index + 3, :], predQuery, dim=-1).values[0]
                    predicted_tokens_four = self.tokenizer.convert_ids_to_tokens(predicted_token_ids_four)
                    predicted_tokens_four = [token.replace("▁", "") for token in predicted_tokens_four]
                    predicted_tokens,predicted_scores = cross(predicted_tokens,predicted_tokens_four,predicted_scores,predicted_scores_four)

                if count == 5:
                    predicted_token_ids_five = torch.topk(outputs.logits[..., token_index + 4, :], predQuery, dim=-1).indices[0]
                    predicted_scores_five = torch.topk(outputs.logits[..., token_index + 4, :], predQuery, dim=-1).values[0]
                    predicted_tokens_five = self.tokenizer.convert_ids_to_tokens(predicted_token_ids_five)
                    predicted_tokens_five = [token.replace("▁", "") for token in predicted_tokens_five]
                    predicted_tokens,predicted_scores = cross(predicted_tokens,predicted_tokens_five,predicted_scores,predicted_scores_five)


            out.append(predicted_tokens)

            # 更新token索引
            token_index += count

        return out
    
    def tomask2(self,fsen,tsen,i):
        te = self.tomask(fsen,5)
        for maski in te:
            if maski[0] == tsen[i]:
                return maski[1]
        return -1
    

    def AttFix(self,fsen,WINDOW,index):
        temp2 = self.tomask(fsen,WINDOW,True)
        TTEMP = []
        i = index
        for j in temp2[i]:
            if IS_levenshtein_distance_and_operations(fsen[i],j):
                TTEMP.append(j)
            if len(TTEMP) == WINDOW:
                break
        return TTEMP
    





def cross(Alist, Blist, Ascor, Bscor):
    #Softmax 

    if isinstance(Ascor, torch.Tensor):
        Ascor = Ascor.numpy()
    if isinstance(Bscor, torch.Tensor):
        Bscor = Bscor.numpy()
# softmax
    Ascor = [math.exp(i) for i in Ascor]
    Bscor = [math.exp(i) for i in Bscor]
    Ascor = [i/sum(Ascor) for i in Ascor]
    Bscor = [i/sum(Bscor) for i in Bscor]


    Alist_copy = Alist
    Ascor_copy = [i*2 for i in Ascor]

    # 删除两个list中  不满足 isalpha 但保留空的
    for index in range(len(Alist)):
        if not Alist[index].isalpha() and Alist[index] != '':
            Alist[index] = ''
            Ascor[index] = 0
    for index in range(len(Blist)):
        if not Blist[index].isalpha() and Blist[index] != '':
            Blist[index] = ''
            Bscor[index] = 0

    OUTlist = [a + b for a in Alist for b in Blist]
    OUTscor = [Ascor[i] + Bscor[j] for i in range(len(Ascor)) for j in range(len(Bscor))]


    OUTlist = OUTlist + Alist_copy
    OUTscor = OUTscor + Ascor_copy


    # 对于OUTlist中的重复 是得它们的OUTscor相加
    score_dict = {}
    for i in range(len(OUTlist)):
        if OUTlist[i] in score_dict:
            score_dict[OUTlist[i]] += OUTscor[i]
        else:
            score_dict[OUTlist[i]] = OUTscor[i]
    # 根据分数排序
    OUTlist, OUTscor = zip(*sorted(score_dict.items(), key=lambda x: x[1], reverse=True))
    return list(OUTlist), list(OUTscor)


def create_cyrillic_vector(text):
    # 西里尔字母表的小写字母
    alphabet = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё'
    letter_count = {'а': 0, 'б': 0, 'в': 0, 'г': 0, 'д': 0, 'е': 0, 'ж': 0, 'з': 0, 'и': 0, 'й': 0, 'к': 0, 'л': 0, 'м': 0, 'н': 0, 'о': 0, 'п': 0, 'р': 0, 'с': 0, 'т': 0, 'у': 0, 'ф': 0, 'х': 0, 'ц': 0, 'ч': 0, 'ш': 0, 'щ': 0, 'ъ': 0, 'ы': 0, 'ь': 0, 'э': 0, 'ю': 0, 'я': 0, 'ё': 0}
    
    # 遍历文本中的每个字符，并更新频率
    for char in text:
        if char in letter_count:
            letter_count[char] += 1
    
    # 从字典中提取频率值，构成向量
    vector = [letter_count[letter] for letter in alphabet]
    # 转为torch
    return torch.tensor(vector, dtype=torch.float32)

def levenshtein_distance_and_operations(s1, s2):
    # 创建一个矩阵来存储从 s1 转换到 s2 的操作过程中的步骤
    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        dist[i][0] = (i, i, 0, 0)  # 需要 i 次删除操作来从 s1 的前 i 个字符变为空字符串
    for j in range(1, cols):
        dist[0][j] = (j, 0, j, 0)  # 需要 j 次插入操作来从空字符串变为 s2 的前 j 个字符

    for col in range(1, cols):
        for row in range(1, rows):
            if s1[row - 1] == s2[col - 1]:
                cost = 0
            else:
                cost = 1
            delete = (dist[row - 1][col][0] + 1, dist[row - 1][col][1] + 1, dist[row - 1][col][2], dist[row - 1][col][3])
            insert = (dist[row][col - 1][0] + 1, dist[row][col - 1][1], dist[row][col - 1][2] + 1, dist[row][col - 1][3])
            substitute = (dist[row - 1][col - 1][0] + cost, dist[row - 1][col - 1][1], dist[row - 1][col - 1][2], dist[row - 1][col - 1][3] + cost)

            # 选择最小编辑距离的操作
            if cost == 0:  # 字符相同，无需操作
                dist[row][col] = dist[row - 1][col - 1]
            else:
                dist[row][col] = min((delete, insert, substitute), key=lambda x: x[0])

    # 结果位于矩阵的最后位置，包含总编辑距离和各操作次数
    return dist[-1][-1]



def IS_levenshtein_distance_and_operations(s1, s2,s1len=None):
    if s1len == None:
        s1len = len(s1)

    dif = abs(s1len-len(s2))
    
    if dif > 1:
        return False
    
    # if s1len+len(s2) <= 3:
        # return True
    

    if not s2.isalpha():
        return False
    
    if letterSim(s1,s2) < 0.7:
        return False
    
    temp = levenshtein_distance_and_operations(s1, s2)

    if temp[3] == 1:
        if temp[2] == 0 and temp[1] == 0:
            return True
        else:
            return False
    elif temp[3] == 0:
        if temp[0] <= 2:
            return True
        else:
            return False
    else:
        return False
        



def letterSim(A,B):
    tedict = {}
    for i in range(len(A)):
        if A[i] not in tedict:
            tedict[A[i]] = 1
        else:
            tedict[A[i]] += 1
    for i in range(len(B)):
        if B[i] not in tedict:
            tedict[B[i]] = -1
        else:
            tedict[B[i]] -= 1
    sum = 0
    for i in tedict:
        sum += abs(tedict[i])
    le = (len(A) + len(B))
    lediff = abs(len(A) - len(B))
    te = (le - sum + math.exp(-lediff)) / le  
    # if te > 1:
    return  te



def senAvg(sen):
    # 计算所有字符串的长度总和
    total_length = sum(len(s) for s in sen)
    
    # 计算字符串的数量
    num_sen = len(sen)
    
    # 计算并返回平均长度
    return total_length / num_sen




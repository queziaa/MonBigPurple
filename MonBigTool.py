import pickle

class MonBigTool:
    def __init__(self,pipe):
        self.MASK = None
        self.wordsDict = None
        self.pipe = pipe
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
    
    def Decode(self, ma_i):
        Tsum = 0
        inode = []
        tsen = ma_i['sen'].copy()
        fsen = ma_i['sen'].copy()
        freq = []
        wordDict = self.getWordsDict()
        for index in range(len(ma_i['sen'])):
            if ma_i['sen'][index] == '<>':
                Tsum += 1
                inode.append([])
                inode[-1].append(index)
                inode[-1].append((Tsum-1) * 2)
                tsen[index] = ma_i['sen'][inode[-1][0]]
                fsen[index] = ma_i['word'][inode[-1][1]+1]

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

        return inode,tsen,fsen,freq,pairs
    


    def letterSim(self,A,B):
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
        le = ((len(A) + len(B)))
        return  (le - sum) / le  


    def tomask(self,sen,i):
        input_ = sen.copy()
        input_[i] = '[MASK]'
        input_ = " ".join(input_)
        return self.pipe(input_)
    
    def tomask2_ishit(self,fsen,tsen,i):
        te = self.tomask(fsen,i)
        for maski in te:
            if maski['token_str'] == tsen[i]:
                return maski['score']
        return -1
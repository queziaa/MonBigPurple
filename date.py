# train_spell_error.txt
# train_clean.txt
spell_error = open("train_spell_error.txt", "r")
clean = open("train_clean.txt", "r")
spell_error = spell_error.read().split("\n")
clean = clean.read().split("\n")

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
    to = len(A) + len(B)
    to = to / 2
    return to,sum

# 将spell_error和clean的数据每个句子以空格切分，视为单词，如果单词最后一个元素为“:”则将其单独切分
for i in range(len(spell_error)):
    temp = spell_error[i].split(" ")
    spell_error[i] = []
    for j in range(len(temp)):
        # print(temp)
        # print('*',temp[j],'*')
        if len(temp[j]) == 0:
            continue
        if temp[j][-1] == ":":
            spell_error[i].append(temp[j][:-1])
            spell_error[i].append(":")
        else:
            spell_error[i].append(temp[j])

for i in range(len(clean)):
    temp = clean[i].split(" ")
    clean[i] = []
    for j in range(len(temp)):
        if len(temp[j]) == 0:
            continue
        if temp[j][-1] == ":":
            clean[i].append(temp[j][:-1])
            clean[i].append(":")
        else:
            clean[i].append(temp[j])

for i in range(len(spell_error)):
    if len(spell_error[i]) != len(clean[i]):
        for j in range(len(spell_error[i])):
            # 如果单词含有“:”且不是最后一个元素
            if spell_error[i][j].find(":") != -1 and spell_error[i][j][-1] != ":":
                # spell_error[i][j]位置赋予删除“:”后的单词
                spell_error[i][j] = spell_error[i][j].replace(":", "")
                # spell_error[i][j]之后插入“:”
                spell_error[i].insert(j+1, ":")


OUT = []

ta = {}
tb = {}
errerDict = {}
norDict = {}
tol = 0  #79791
tttt=  0
ttttt = 0
for i in range(len(spell_error)):
    indexx = 0
    tempDict = {}
    tempDict['sen'] = ''
    tempDict['word'] = []
    templist = []
    for  j in range(len(spell_error[i])):
        if len(spell_error[i]) == len(clean[i]) and spell_error[i][j] != clean[i][j]: #舍弃16个句子 缺失":" 的
            tempDict['word'].append(clean[i][j])
            tempDict['word'].append(spell_error[i][j])
            templist.append(j)
            tol += 1
            indexx+= 1
    if indexx != 0:
        # 深拷贝
        tempDict['sen'] = clean[i].copy()
        for j in range(len(templist)):
            tempDict['sen'][templist[j]] = '<>'
        OUT.append(tempDict)
        print(clean[i])
        print(spell_error[i])
        print(tempDict)
    # else:
    #     ttttt +=1 
    #     print(clean[i])
    #     print(spell_error[i])
    #     print(i)


#保存OUT 到二进制文件 mask.pickle
import pickle

with open('mask.pickle', 'wb') as f:
    pickle.dump(OUT, f)


    # if indexx ==0:
        # print(spell_error[i])
        # print(clean[i])
        # tttt += 1
        #     if spell_error[i][j] not in errerDict:
        #         errerDict[spell_error[i][j]] = 1
        #     else:
        #         errerDict[spell_error[i][j]] += 1

        #     if clean[i][j] not in norDict:
        #         norDict[clean[i][j]] = 1
        #     else:
        #         norDict[clean[i][j]] += 1
        # else:
        #     if clean[i][j] not in norDict:
        #         norDict[clean[i][j]] = 1
        #     else:
        #         norDict[clean[i][j]] += 1
        


            # if len(spell_error[i][j]) == 1:
            #     print(spell_error[i][j],clean[i][j])
            #     if spell_error[i][j] not in ta:
            #         ta[spell_error[i][j]] = 1
            #     else:
            #         ta[spell_error[i][j]] += 1

            # if len(clean[i][j]) == 1:
            #     print(spell_error[i][j],clean[i][j])
            #     if clean[i][j] not in tb:
            #         tb[clean[i][j]] = 1
            #     else:
            #         tb[clean[i][j]] += 1

            # a,b = letterSim(clean[i][j],spell_error[i][j])
            # if b > a :
            #     print('#',clean[i][j],'#',spell_error[i][j],'#',letterSim(clean[i][j],spell_error[i][j]))


            # print(spell_error[i])
            # print(clean[i])
            # print(" ".join(clean[i]))
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # if len(spell_error[i]) != len(clean[i]):
        # print(spell_error[i])
        # print(clean[i])
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # senlenserr +=1 
    # if indexx > 1:
        # onesenhavemulerrors += 1



# # 寻找所有":"之前的单词  
# te = {}
# for i in range(len(clean)):
#     for j in range(len(clean[i])-1):
#         if clean[i][j+1] == ":":
#             if clean[i][j] not in te:
#                 te[clean[i][j]] = 1
#             else:
#                 te[clean[i][j]] += 1

# for i in range(len(clean)):
#     for j in range(len(clean[i])):
#         if clean[i][j] in te and len(clean[i]) > j+1:
#             print(clean[i][j+1])
# 单字错误情况
# я	8	——t
# ю	1	
# ю	97	——t
# э	1	
# э	72	——t
# ь	258	——t
# ы	1	
# ы	10	——t
# щ	22	——t
# ш	6	
# ш	23	——t
# ч	300	
# ч	26	——t
# ц	8	
# ц	13	——t
# х	4	
# х	83	——t
# ф	3	
# ф	18	——t
# ү	59	——t
# у	1	
# у	64	——t
# т	10	
# т	37	——t
# с	13	
# с	46	——t
# р	123	
# р	44	——t
# п	2	
# п	14	——t
# ө	6	
# ө	4	——t
# о	2	
# о	7	——t
# н	9	
# н	305	——t
# м	8	
# м	121	——t
# л	221	
# л	28	——t
# к	2	
# к	16	——t
# й	35	——t
# и	3	
# и	115	——t
# з	2	
# з	29	——t
# ж	4	
# ж	38	——t
# ё	6	——t
# е	2	
# е	16	——t
# д	50	
# д	28	——t
# г	22	
# г	49	——t
# в	6	
# в	37	——t
# б	13	
# б	102	——t
# а	3	
# v	2	
# s	1	
# n	1	
# i	1	
# 5	2	
# 错误绝大部分都是对应位置的单词的拼写错误 
# errrepeat = 0  #9578个词是句子中错误拼写但是在正确的词典中
# for i in errerDict:
#     if i in norDict:
#         errrepeat += errerDict[i]
#         print(i,errerDict[i],norDict[i])
# errrepeat = 0  #41968 个词是句子中错误拼写但是在正确的词典中 这次是更大词典
# for i in errerDict:
#     if i not in wordsDict:
#         print(i,errerDict[i])
#         errrepeat += 1
# wordsDict.pickle  读取
# import pickle
# with open('wordsDict.pickle', 'rb') as f:
#     wordsDict = pickle.load(f)
# tol #79813 处拼写错误
# spell_error    50000 个句子
# tttt 6916 个句子在错误集合中但是和同行的正确集合一样
# onesenhavemulerrors = 0   #22883 含有多个错误的句子
# senlenserr = 0  #34  经过简单分词后长度不一致的句子，因为":"

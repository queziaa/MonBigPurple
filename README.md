# 实验报告

## 简介
拼写检查（拼写纠错）任务是一项悠久的自然语言处理任务，旨在检测文本中的拼写错误并提供正确的拼写建议。拼写检查系统通常由两个主要组件组成：错误检测器和纠错器。当然如今也有些工作是将两个任务合并在一起，使用深度学习完成端到端的拼写纠错任务。

在机器学习之前就有通过规则的方式来进行拼写检查，例如使用词典。但是这种方法有一个缺点，就是无法处理未知词，以及当错误的词与另外一个正确的词相似时，可能会出现错误的纠正。尤其对于中文来说，使用拼写输入法导致拼写错误常表现为同音字、同形字、形近字等，这些错误很难通过规则来纠正。

之后随着机器学习算法的发展，几乎每一个当时火热的机器学习算法都被应用到了拼写检查任务中，例如贝叶斯方法、最大熵模型、条件随机场、神经网络等。其中神经网络在近年来取得了很大的进展，例如使用循环神经网络（RNN）、长短时记忆网络（LSTM）、卷积神经网络（CNN）等。这些方法通常会使用大量的数据进行训练，以提高模型的泛化能力。

下一个世代是以transformer为代表的预训练模型，这种模型在拼写检查任务中也取得了很好的效果。这种模型通常会使用大量的数据进行预训练，然后在特定任务上进行微调。这种方法的优点是可以很好的处理未知词，以及可以通过大量的数据进行训练，提高模型的泛化能力。如BERT、GPT、RoBERTa等。

这里为了开发的便捷性，我们在Hugging Face中寻找蒙古语的预训练模型，以此为基础开发拼写检查系统。

## 前期准备工作
这里对Huagging Face中的蒙古语预训练模型和多语言预训练模型进行了筛选，选择了一些适合拼写检查任务的模型。这些模型包括：
来自Baljinnyam[^1]的roberta
来自Blgn94[^2]的bert,roberta
来自bayartsogt[^3]的roberta,albert,bert
来自Dakie的[^4]的roberta,bert
来自tugstugi[^5]的bert
这些模型都是在蒙古语数据上进行了预训练的，在之后的各项任务中测试了这些模型的性能，最后选择了tugstugi的bert模型作为拼写检查系统的基础模型。


为了补充语料库，我们还使用了来自tugstugi整理的蒙古语新闻数据400M[^5]。

## 


[^1]: [Baljinnyam Dayan](https://huggingface.co/Baljinnyam)
[^2]: [Blgn94](https://huggingface.co/Blgn94)
[^3]: [bayartsogt](https://huggingface.co/bayartsogt)
[^4]: [Dakie](https://huggingface.co/Dakie)
[^5]: [tugstugi](https://huggingface.co/tugstugi)


## 


在拼写纠错任务中，WER（Word Error Rate）和CER（Character Error Rate）是两个常用的性能指标，用于评估模型的纠错效果。

WER（Word Error Rate）：

定义：WER 是衡量系统输出与参考文本之间差异的指标，计算公式为：

WER = (S + D + I) / N
其中，S 是替换错误（substitutions），D 是删除错误（deletions），I 是插入错误（insertions），N 是参考文本中的单词总数。
用途：WER 主要用于评估语音识别系统和文本纠错系统的性能，反映了系统在单词层面上的错误率。
CER（Character Error Rate）：

定义：CER 是衡量系统输出与参考文本之间差异的指标，计算公式为：

CER = (S + D + I) / N
其中，S 是替换错误（substitutions），D 是删除错误（deletions），I 是插入错误（insertions），N 是参考文本中的字符总数。
用途：CER 主要用于评估文本纠错系统的性能，反映了系统在字符层面上的错误率。
这两个指标都用于量化模型的纠错能力，但侧重点不同：WER 关注单词层面的错误，而 CER 关注字符层面的错误。选择使用哪一个指标取决于具体应用场景和需求。


https://spellcheck.gov.mn/



https://github.com/shibing624/pycorrector/tree/master/examples/macbert
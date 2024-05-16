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

@ https://huggingface.co/bayartsogt/mongolian-gpt2
https://huggingface.co/Dorjzodovsuren/mongolian-gpt2
@ https://huggingface.co/flax-community/mongolian-gpt2
@ https://huggingface.co/Baljinnyam/mongolian-gpt2-ner-finetuning
@ https://huggingface.co/Baljinnyam/mongolian-gpt-2
Baljinnyam/gpt-2-10000
Baljinnyam/gpt-2-2000
Baljinnyam/gpt-2




@ https://huggingface.co/bayartsogt/mongolian-roberta-large 可 2023



@ https://huggingface.co/tugstugi/bert-base-mongolian-cased   
@ https://huggingface.co/tugstugi/bert-large-mongolian-cased   OR UNCASED  可 有论文 用UNCASED 数据集合全小写


@ Blgn94    不确定如何使用
Blgn94/mongolian-Davlan-distilbert-base-multilingual-cased-ner-hrl
Blgn94/mongolian-davlan-xlm-roberta-base-ner-hrl
Blgn94/mongolian-twitter-roberta-base-sentiment-ner
Blgn94/mongolian-roberta-large-mnli-ner
Blgn94/mongolian-bert-base-multilingual-cased-ner
Blgn94/mongolian-xlm-roberta-base-ner
Blgn94/roberta-base-ner-demo



@ Baljinnyam
<!-- Baljinnyam/xlm-roberta-base-ner-hrl-ner-finetuning -->
<!-- Baljinnyam/bayartsogt-albert-mongolain-ner-finetuning -->
<!-- Baljinnyam/bert-base-multilingual-uncased-mongolian-ner -->
<!-- Baljinnyam/roberta-large-mnli-ner-2000 -->
<!-- Baljinnyam/roberta-large-mnli-ner-1000 -->
<!-- Baljinnyam/Roberta-large-mnli-mongolian-ner -->
<!-- Baljinnyam/xlm-roberta-base-mongolian-ner-finetuning -->





<!-- Dakie  多语言 微调 -->
Dakie/mongolian-xlm-roberta-large
Dakie/mongolian-roberta-base
Dakie/mongolian-distilbert-base-multilingual-cased
Dakie/mongolian-bert-base-multilingual-cased
Dakie/mongolian-roberta-base-mn
Dakie/mongolian-xlm-roberta-base
Dakie/finetuned-xlm-roberta-large


https://github.com/nghuyong/Chinese-text-correction-papers
https://github.com/bedapudi6788/deepcorrect
https://github.com/shibing624/pycorrector

https://huggingface.co/shibing624/macbert4csc-base-chinese
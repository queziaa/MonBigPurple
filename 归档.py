# https://huggingface.co/Dorjzodovsuren/mongolian-gpt2/tree/main
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("Dakie/mongolian-xlm-roberta-base")
model = AutoModelForTokenClassification.from_pretrained("Dakie/mongolian-xlm-roberta-base")

# prepare input
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

# forward pass
output = model(**encoded_input)

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bayartsogt/mongolian-gpt2")
model = AutoModelForCausalLM.from_pretrained("bayartsogt/mongolian-gpt2")
# 准备输入文本，这是模型生成续写的起点
prompt_text = "Өглөөний нар мандаж,"  # 假设这意味着“早晨的太阳升起，”

# 将文本编码为模型可以理解的格式
input_ids = tokenizer.encode(prompt_text, return_tensors='pt')

# 使用模型生成文本
# num_return_sequences=1 表示生成一个续写
# max_length 指定生成文本的最大长度，包括输入文本的长度
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
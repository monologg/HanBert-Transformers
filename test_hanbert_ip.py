import torch
from transformers import BertModel
from tokenization_hanbert import HanBertTokenizer

# Load model and tokenizer
model = BertModel.from_pretrained('HanBert-54kN-IP-torch')
tokenizer = HanBertTokenizer.from_pretrained('HanBert-54kN-IP-torch')

text = "나는 걸어가고 있는 중입니다. 나는걸어 가고있는 중입니다. 잘 분류되기도 한다. 잘 먹기도 한다."

inputs = tokenizer.encode_plus(
    text=text,
    text_pair=None,
    add_special_tokens=True,  # This add [CLS] on front, [SEP] at last
    pad_to_max_length=True,
    max_length=40
)

print("--------------------------------------------------------")
print("tokens: ", " ".join(tokenizer.tokenize("[CLS] " + text + " [SEP]")))
print("input_ids: {}".format(" ".join([str(x) for x in inputs['input_ids']])))
print("token_type_ids: {}".format(" ".join([str(x) for x in inputs['token_type_ids']])))
print("attention_mask: {}".format(" ".join([str(x) for x in inputs['attention_mask']])))
print("--------------------------------------------------------")

# Make the input with batch size 1
input_ids = torch.LongTensor(inputs['input_ids']).unsqueeze(0)
token_type_ids = torch.LongTensor(inputs['token_type_ids']).unsqueeze(0)
attention_mask = torch.LongTensor(inputs['attention_mask']).unsqueeze(0)

sequence_output, pooled_output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
print("<Last layer hidden state (batch_size, max_seq_len, dim)>")
print(sequence_output)

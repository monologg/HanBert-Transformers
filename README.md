# HanBert-Transformers

[HanBert](https://github.com/tbai2019/HanBert-54k-N) on 🤗 Huggingface Transformers 🤗

## Details

- HanBert Tensorflow ckpt를 Pytorch로 변환
  - 기존의 Optimizer 관련 Parameter는 제거하여 기존의 1.43GB에서 488MB로 줄였습니다.
  - **변환 시 Optimizer 관련 파라미터를 Skip 하지 못하는 이슈가 있어 해당 부분을 고쳐서 변환했습니다.** ([해당 이슈 관련 PR](https://github.com/huggingface/transformers/pull/2652))

```bash
# transformers bert TF_CHECKPOINT TF_CONFIG PYTORCH_DUMP_OUTPUT
$ transformers bert HanBert-54kN/model.ckpt-3000000 \
                    HanBert-54kN/bert_config.json \
                    HanBert-54kN/pytorch_model.bin
```

- Tokenizer를 위하여 `tokenization_hanbert.py` 파일을 새로 제작
  - Transformers의 **tokenization 관련 함수 지원** (`convert_tokens_to_ids`, `convert_tokens_to_string`, `encode_plus`...)

## How to Use

1. **관련 라이브러리 설치**

   - torch>=1.1.0
   - transformers>=2.2.2

2. **모델 다운로드 후 압축 해제**

   - 기존의 HanBert에서는 tokenization 관련 파일을 `/usr/local/moran`에 복사해야 했지만, 해당 폴더 이용 시 그래도 사용 가능합니다.
   - **다운로드 링크 (Pretrained weight + Tokenizer tool)**
     - [HanBert-54kN-torch](https://drive.google.com/open?id=1LUyrnhuNC3e8oD2QMJv8tIDrXrxzmdu4)
     - [HanBert-54kN-IP-torch](https://drive.google.com/open?id=1wjROsuDKoJQx4Pu0nqSefVDs3echKSXP)

3. **tokenization_hanbert.py 준비**

   - Tokenizer의 경우 **Ubuntu** 환경에서만 사용 가능합니다.
   - 하단의 형태로 디렉토리가 세팅이 되어 있어야 합니다.

```
.
├── ...
├── HanBert-54kN-torch
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── vocab_54k.txt
│   ├── libmoran4dnlp.so
│   ├── moran.db
│   ├── udict.txt
│   └── uentity.txt
├── tokenization_hanbert.py
└── ...
```

## Example

### 1. Model

```python
>>> import torch
>>> from transformers import BertModel

>>> model = BertModel.from_pretrained('HanBert-54kN-torch')
>>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
>>> token_type_ids = torch.LongTensor([[0, 0, 0], [0, 0, 0]])
>>> attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
>>> sequence_output, pooled_output = model(input_ids, attention_mask, token_type_ids)
>>> sequence_output
tensor([[[-0.0938, -0.5030,  0.3765,  ..., -0.4880, -0.4486,  0.3600],
         [-0.6036, -0.1008, -0.2344,  ..., -0.6606, -0.5762,  0.1021],
         [-0.4353,  0.0970, -0.0781,  ..., -0.7686, -0.4418,  0.4109]],

        [[-0.7117,  0.2479, -0.8164,  ...,  0.1509,  0.8337,  0.4054],
         [-0.7867, -0.0443, -0.8754,  ...,  0.0952,  0.5044,  0.5125],
         [-0.8613,  0.0138, -0.9315,  ...,  0.1651,  0.6647,  0.5509]]],
       grad_fn=<AddcmulBackward>)
```

### 2. Tokenizer

```python
>>> from tokenization_hanbert import HanBertTokenizer
>>> tokenizer = HanBertTokenizer.from_pretrained('HanBert-54kN-torch')
>>> text = "나는 걸어가고 있는 중입니다. 나는걸어 가고있는 중입니다. 잘 분류되기도 한다. 잘 먹기도 한다."
>>> tokenizer.tokenize(text)
['나', '~~는', '걸어가', '~~고', '있', '~~는', '중', '~~입', '~~니다', '.', '나', '##는걸', '##어', '가', '~~고', '~있', '~~는', '중', '~~입', '~~니다', '.', '잘', '분류', '~~되', '~~기', '~~도', '한', '~~다', '.', '잘', '먹', '~~기', '~~도', '한', '~~다', '.']
```

### 3. Test with python file

```bash
$ python3 test_hanbert.py
$ python3 test_hanbert_ip.py
```

## Result on Subtask

`max_seq_len = 50`으로 설정

|                   | **NSMC** (acc) | **Naver-NER** (F1) |
| ----------------- | -------------- | ------------------ |
| HanBert-54kN      | **90.16**      | **84.84**          |
| HanBert-54kN-IP   | 88.72          | 84.45              |
| KoBERT            | 89.63          | 84.23              |
| Bert-multilingual | 87.07          | 81.78              |

- NSMC (Naver Sentiment Movie Corpus) ([Implementation of HanBert-nsmc](https://github.com/monologg/HanBert-nsmc))
- Naver NER (NER task on Naver NLP Challenge 2018) ([Implementation of HanBert-NER](https://github.com/monologg/HanBert-NER))

## Reference

- [HanBert Github](https://github.com/tbai2019/HanBert-54k-N)
- [KoBERT Github](https://github.com/SKTBrain/KoBERT)
- [Kobert-transformers](https://pypi.org/project/kobert-transformers/)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [From Tensorflow to Pytorch](https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28)

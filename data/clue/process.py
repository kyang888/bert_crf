import json
import os
import sys
sys.path.insert(0, "../../")
from tokenization.tokenization import FullTokenizer
tokenizer = FullTokenizer("../../resources/vocab.txt", do_lower_case=True)

def exist_nested_entity(line):
    res = False
    entity_char_span = []
    for d in line['label'].values():
        for entity_value, char_spans in d.items():
            entity_char_span.extend(char_spans)
    entity_char_span = sorted(entity_char_span, key=lambda x: x[0])
    start = 0
    for s, e in entity_char_span:
        if start <= s and s <= e:
            start = e
        else:
            res = True
            break
    return res


input_dir = "raw"
# for data_name in ["train.json", "dev.json"]:
#     ner2id = json.load(open("../../resources/ner2id.json", "r", encoding="utf-8"))
#     data = []
#     for line in open(os.path.join(os.path.join(input_dir, data_name)), "r", encoding="utf-8"):
#         line = line.strip()
#         if line:
#             line = json.loads(line)
#             if exist_nested_entity(line):
#                 print(line)
#                 continue
#             text = line['text']
#             label = line['label']
#
#             type2char_span = {}
#             for type, d in label.items():
#                 type2char_span[type] = []
#                 for char_spans in d.values():
#                     for char_span in char_spans:
#                         type2char_span[type].append([char_span[0], char_span[-1]+1])
#
#             assert len(text.lower()) == len(text)
#             token = list(text.lower())
#             assert len(token) < 300
#             input_ids =  tokenizer.convert_tokens_to_ids(["[CLS]"] + token + ["[SEP]"])
#             labels = [0] * len(input_ids)
#             for type, char_spans in type2char_span.items():
#                 for s, e in char_spans:
#                     if e - s == 1:
#                         labels[s+1] = ner2id["S-" + type]
#                     else:
#                         labels[s+1] = ner2id["B-" + type]
#                         for i in range(s+1, e):
#                             labels[i+1] = ner2id["I-" + type]
#
#             data.append({"input_ids":input_ids,"labels":labels, "text": text, "token":token})
#     print(len(data))
#     with open(data_name, "w", encoding="utf-8") as f:
#         for line in data:
#             f.write(json.dumps(line, ensure_ascii=False) + "\n")





input_dir = "raw"
for data_name in ["train.json", "dev.json"]:
    ner2id = json.load(open("../../resources/ner2id.json", "r", encoding="utf-8"))
    data = []
    for line in open(os.path.join(os.path.join(input_dir, data_name)), "r", encoding="utf-8"):
        line = line.strip()
        if line:
            line = json.loads(line)
            if exist_nested_entity(line):
                print(line)
                continue
            text = line['text']
            label = line['label']

            toks, i2jk = tokenizer.tokenize(text)
            toks = ['[CLS]'] + toks + ["[SEP]"]
            i2jk = [[-1, -1]] + i2jk + [[-1, -1]]

            i2j = {}
            i2k = {}
            for token_index, mapping in enumerate(i2jk):
                start, end = mapping[0], mapping[-1]
                if start == end == -1:
                    continue
                i2j[start] = token_index
                i2k[end] = token_index
            entity_list = []
            for type, value2indexes in label.items():
                for value, indexes in value2indexes.items():
                    for start, end in indexes:
                        entity_list.append({"type": type, "char_start": start, "char_end": end, "value": value})
            entity_list_new = []
            for entity in entity_list:
                if entity['char_start'] not in i2j:
                    print("实体的char start找不到对应的token start")
                elif entity['char_end'] not in i2k:
                    print("实体的char end找不到对应的token end")
                else:
                    value_tok = tokenizer.tokenize(entity['value'])[0]
                    entity['token_start'], entity['token_end'] = i2j[entity['char_start']], i2k[entity['char_end']]
                    entity['token'] = toks[entity['token_start']:entity['token_end'] + 1]
                    if entity['token'] != value_tok:
                        print("文本tokenizer后，实体的token 不等于 实体值得tokens")
                    else:
                        entity_list_new.append(entity)

            assert len(toks) < 128
            input_ids = tokenizer.convert_tokens_to_ids(toks)
            labels = [0] * len(toks)
            for entity in entity_list_new:
                s = entity['token_start']
                e = entity['token_end']
                type = entity['type']
                if e - s == 0:
                    labels[s] = ner2id["S-" + type]
                else:
                    labels[s] = ner2id["B-" + type]
                    for i in range(s+1, e+1):
                        labels[i] = ner2id["I-" + type]

            data.append({"input_ids":input_ids,"labels":labels, "text": text, "token":toks})
    print(len(data))
    with open(data_name, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")










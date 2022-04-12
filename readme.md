# bert crf
## train
```angular2
bash train.sh
```

- 数据集使用clue ner数据集，采用bertTokenizer分词时，存在token边界与实体边界矛盾，
 例如 "朴茨茅斯队vsac米兰队"，"ac米兰"是一个实体，但是tokenizer的结果是【 'vsa', '##c'】，所以这里采用list(str)的分词方式。
 
- 验证结果，使用哈工大的 wwm_ext， 基于char span dev的f1是80.15， 基于token span dev的f1是80.4。

## TODO
- 部署，目前crf层转onnx总是失败，后续重新研究下crf层的代码。
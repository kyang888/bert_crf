 # Bert Tokenzation
 - 回顾bert的tokenizer，并实现分词后token到raw_text_char的映射关系。
 - tokenization_bert.py 原始的bert分词方式
   ```
      tokenizer = FullTokenizer("vocab.txt")
      text = '哈哈，abn\u0303o'
      tokens = tokenizer.tokenize(text)
      #tokens:['哈', '哈', '，', 'ab', '##no']
   ```
   长度为500的样本，totokenizer.tokenize耗时2.3ms
 - tokenization.py 
   ```
     tokenizer = FullTokenizer("vocab.txt")
     text = '哈哈，abn\u0303o'
     tokens, index_map = tokenizer.tokenize(text)
     print(tokens, index_map)
     # tekens:['哈', '哈', '，', 'ab', '##no']
     # index_map:[[0], [1], [2], [3, 4], [5, 7]]
   ```
   长度为500的样本，totokenizer.tokenize耗时5.1ms
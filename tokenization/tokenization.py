"""
tokenizaion return offsetmapping
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six

# WHITE_SPACE = []
# # 遍历目前所有的unicode字符
# for n in range(0, 65535 * 17):
#   if chr(n).split() == []:
#     WHITE_SPACE.append(n)

WHITE_SPACE = [9, 10, 11, 12, 13, 28, 29, 30, 31, 32, 133, 160, 5760, 8192, 8193, 8194, 8195, 8196, 8197, 8198, 8199,
               8200, 8201, 8202, 8232, 8233, 8239, 8287, 12288]
WHITE_SPACE = [chr(i) for i in WHITE_SPACE]

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if isinstance(text, str):
    return text
  elif isinstance(text, bytes):
    return text.decode("utf-8", "ignore")
  else:
    raise ValueError("Unsupported string type: %s" % (type(text)))


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with open(vocab_file, "r", encoding="utf-8") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    if item in vocab:
      output.append(vocab[item])
    else:
      output.append(vocab['[UNK]'])
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text, index_map):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  # 原始的实现
  # text = text.strip()
  # if not text:
  #   return []
  # tokens = text.split()
  # return tokens
  start = 0
  while start < len(text) and text[start] in WHITE_SPACE:
    start += 1
  text = text[start:]
  index_map = index_map[start:]
  end = len(text) - 1
  while end > -1 and text[end] in WHITE_SPACE:
    end -= 1
  text = text[:end+1]
  index_map = index_map[:end+1]
  tokens, new_index_map = [], []
  cur_token, cur_index_map = "", []
  for index, char in enumerate(text):
    if char in WHITE_SPACE:
      tokens.append(cur_token)
      new_index_map.append(cur_index_map)
      cur_token, cur_index_map = "", []
    else:
      cur_token = cur_token + char
      cur_index_map.append(index_map[index])
  tokens.append(cur_token)
  new_index_map.append(cur_index_map)
  return [t for t in tokens if t], [idx for idx in new_index_map if idx]







class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens,index_map = [],[]
    for token, token_index_map in zip(*self.basic_tokenizer.tokenize(text)):
      for sub_token, sub_token_index_map in zip(*self.wordpiece_tokenizer.tokenize(token, token_index_map)):
        split_tokens.append(sub_token)
        index_map.append(sub_token_index_map)


    return split_tokens, index_map

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    """Constructs a BasicTokenizer.
    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    # token2chars Index
    index_map = list(range(len(text)))
    text, index_map = self._clean_text(text, index_map)
    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text, index_map = self._tokenize_chinese_chars(text, index_map)
    orig_tokens, index_map = whitespace_tokenize(text, index_map)
    split_tokens, new_index_map = [], []
    for token, token_index_map in zip(orig_tokens, index_map):
      if self.do_lower_case:
        # bug例子: "İ" 长度是1， "İ".lower()后长度是2
        # token = token.lower()
        token_temp, token_index_map_temp = "", []
        for index, char in enumerate(token):
          chars = char.lower()
          for c in chars:
            token_temp += c
            token_index_map_temp.append(token_index_map[index])
        token, token_index_map = token_temp, token_index_map_temp

        token, token_index_map = self._run_strip_accents(token, token_index_map)
      token, token_index_map = self._run_split_on_punc(token, token_index_map)
      split_tokens.extend(token)
      new_index_map.extend(token_index_map)
    new_text, new_index_map_flat = "", []
    for index, tok in enumerate(split_tokens):
      new_text = new_text + tok + " "
      new_index_map_flat.extend(new_index_map[index])
      # -1对应上面的" "
      new_index_map_flat.append(-1)
    if new_text:
      new_text = new_text[:-1]
      new_index_map_flat = new_index_map_flat[:-1]
    output_tokens, new_index_map = whitespace_tokenize(new_text, new_index_map_flat)
    return output_tokens,new_index_map

  def _run_strip_accents(self, text, index_map):
    """Strips accents from a piece of text."""
    # text = unicodedata.normalize("NFD", text)
    # output = []
    # for char in text:
    #   cat = unicodedata.category(char)
    #   if cat == "Mn":
    #     continue
    #   output.append(char)
    # return "".join(output)
    output, new_index_map = [], []
    for index, char in enumerate(text):
      # "ñ" 会变成 "ñ`"
      chars = unicodedata.normalize("NFD", char)
      for c in chars:
        cat = unicodedata.category(c)
        if cat == "Mn":
          continue
        output.append(c)
        new_index_map.append(index_map[index])
    return "".join(output), new_index_map



  def _run_split_on_punc(self, text, index_map):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output, new_index_map = [], []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        new_index_map.append([index_map[i]])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
          new_index_map.append([])
        start_new_word = False
        output[-1].append(char)
        new_index_map[-1].append(index_map[i])
      i += 1

    return ["".join(x) for x in output], new_index_map

  def _tokenize_chinese_chars(self, text, index_map):
    """Adds whitespace around any CJK character."""
    output,new_index_map = [], []
    for index, char in enumerate(text):
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
        new_index_map.append(-1)
        new_index_map.append(index_map[index])
        new_index_map.append(-1)
      else:
        output.append(char)
        new_index_map.append(index_map[index])
    return "".join(output), new_index_map

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text, index_map):
    """Performs invalid character removal and whitespace cleanup on text."""
    output, new_index_map = [], []
    for index, char in enumerate(text):
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
      new_index_map.append(index_map[index])
    return "".join(output), new_index_map


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text, index_map):
    """Tokenizes a piece of text into its word pieces.
    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.
    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]
    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.
    Returns:
      A list of wordpiece tokens.
    """

    text = convert_to_unicode(text)
    output_tokens, output_index_map = [], []
    for token, token_index_map in zip(*whitespace_tokenize(text, index_map)):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        output_index_map.append(token_index_map)
        continue

      is_bad = False
      start = 0
      sub_tokens, sub_tokens_index_map = [], []
      while start < len(chars):
        end = len(chars)
        cur_substr, cur_substr_index_map = None, []
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            cur_substr_index_map = token_index_map[start:end]
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        sub_tokens_index_map.append(cur_substr_index_map)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
        output_index_map.append(token_index_map)
      else:
        output_tokens.extend(sub_tokens)
        output_index_map.extend(sub_tokens_index_map)
    return output_tokens, output_index_map


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  # 原始bert的判断逻辑
  #if cat in ("Cc", "Cf"):

  # transformers.BertTokenizer的判断逻辑
  # 这两种不同的判断方式，导致结果略有差别，例如text: "好￴内容"
  if cat.startswith("C"):
    return True
  return False


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False

if __name__ == "__main__":
  tokenizer = FullTokenizer("vocab.txt")

  r = tokenizer.tokenize("好￴内容形")
  print(r)

  text = "".join(["好￴内容形" for _ in range(100)])
  print(len(text))
  import time
  start =time.time()
  n = 1000
  for _ in range(n):
    tokenizer.tokenize(text)
  print((time.time() - start) /n )






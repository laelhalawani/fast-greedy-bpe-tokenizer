# BPE Tokenizer Documentation

- [Table of Contents](#table-of-contents)
- [Introduction and Overview](#introduction-and-overview)
- [Installation](#installation)
- [BPE Tokenizer Class Usage Guide](#bpe-tokenizer-class-usage-guide)
  - [Train and Save or Load Tokenizer](#train-and-save-or-load-tokenizer)
  - [Encode Text to Integers](#encode-text-to-integers)
  - [Decode Integers to Text](#decode-integers-to-text)
- [API Reference](#api-reference)
  - [train](#train)
  - [encode](#encode)
  - [decode](#decode)
  - [save_vocab_file](#save-vocab-file)
  - [load_vocab_file](#load-vocab-file)
- [Advanced Topics](#advanced-topics)
- [License](#license)
- [Contributing](#contributing)

## Introduction and Overview

The BPE Tokenizer is a fast and greedy Byte Pair Encoding tokenizer for Python. It allows you to tokenize text into subword units by iteratively merging the most frequent pairs of adjacent characters or tokens to build up a vocabulary. This documentation provides an overview of the BPE Tokenizer and how to use it effectively.

## Installation

To install the BPE Tokenizer, you can download the repository and open a command prompt in the extracted directory. Then, run the following command:

```python
pip install .
```

After installation, you can import the tokenizer using `import bpe_tokenizer` or `from bpe_tokenizer import BPETokenizer`.

## BPE Tokenizer Class Usage Guide

The BPE Tokenizer class allows trainable subword tokenization of text using the Byte Pair Encoding algorithm. Here are some examples of how to use the class:

### Train and Save or Load Tokenizer

**Train and save the tokenizer from scratch:**
```python
train_file = "./training_data.txt"
saved_vocab = "./vocab.json"
tokenizer = BPETokenizer()
tokenizer.train_from_file(train_file, 5000, True)
tokenizer.save_vocab_file(saved_vocab)
```

**Load a pretrained tokenizer:**
```python
saved_vocab = "./vocab.json"
tokenizer = BPETokenizer(vocab_or_json_path=saved_vocab)
```
**Encode Text to Integers**

Encode text to integers using a trained or loaded tokenizer:
```python
Copy code
saved_vocab = "./vocab.json"
tokenizer = BPETokenizer(saved_vocab)
tokens = tokenizer.encode("This is some example text")
print(tokens)
```

**Decode Integers to Text**
Decode integers back into text:
```python
saved_vocab = "./vocab.json"
tokenizer = BPETokenizer(saved_vocab)
text = tokenizer.decode([1,0, -1, -999])
print(text)
```

## API Reference
### train
*Method Signature:*
```python
def train(self, corpus, desired_vocab_size:int, word_level=True)
```
Performs the core BPE training algorithm on the provided corpus. Iteratively merges the most frequent pair of adjacent symbols in the corpus text to build up vocabulary.

### train_from_file
*Method Signature:*
```python
def train_from_file(self, corpus_file_path:str, desired_vocab_size:int, word_level=True)
```
Trains the BPE model from a corpus file. It reads the corpus text file file line by line and performs the core BPE training algorithm on each line.

### encode
*Method Signature:*
```python
def encode(self, text, pad_to_tokens=0)
```
Encodes text into corresponding integer tokens using the trained vocabulary. It iteratively takes the longest matching substring from the vocabulary and emits the integer token. Unknown symbols are replaced with the UNK token.

### decode
*Method Signature:*
```python
def decode(self, tokens)
```
Decodes a list of integer tokens into the corresponding text. It looks up each integer token in the decoder dictionary to recover the symbol string.

### save_vocab_file
*Method Signature:*
```python
def save_vocab_file(self, json_file_path:str)
```
Saves the current vocabulary dictionary to the provided file path as a JSON file. It serializes the vocabulary dict as JSON using the json module.

### load_vocab_file
*Method Signature:*
```python
def load_vocab_file(self, json_file_path:str)
```
Loads the vocabulary dictionary from a JSON file at the given path using the json module.

### Advanced Topics
Using `word_level = False` will enable the use of a character level BPE model. It is significantly slower for training than a word level model, however it might be more accurate for complex tasks. Character level training were used in GPT tokenizers. The default value of `word_level` for this BPETokenizer implementation is `True`.
## License
GNU AGPLv3 2023, [laelhalawani@gmail.com](https://github.com/laelal.halawani).

## Contributing
Any and all is welcome, thank you!

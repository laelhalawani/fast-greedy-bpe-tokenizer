## Installation
Download repository, open command prompt in the extracted directory and type `pip install .`
now you should be able to `import bpe_tokenizer` or `from bpe_tokenizer import BPETokenizer`

## Fast and greedy BPETokenizer

The BPETokenizer
This a fast and greedy BPE (Byte Pair Encoding) Tokenizer class can be used to tokenize text into subword units. It works by iteratively merging the most frequent pairs of adjacent characters or tokens to build up a vocabulary. 

The tokenizer can be trained on a corpus of text to generate a vocabulary of a desired size. Alternatively, it can be initialized with a pre-trained vocabulary. After training, it can encode text into integer tokens and decode those tokens back into text.

This implementation focuses on speed, readability and ease of use and adjustment. Sports detailed docstrings and documentation. 

*See example usage in example.py*

Some potential applications:

- Train a tokenizer on a large corpus to get a generalized vocabulary, save to file and load for later use
- Train a domain-specific tokenizer to get better representations of rare words in that domain
- Use pretrained tokenizer from file for encoding/decoding text 


### The key methods are:

**train**
- Overview: Trains the tokenizer on a corpus to generate a vocabulary
- Details: Counts symbol pairs, finds most frequent, adds to vocab until reaching desired size

**encode** 
- Overview: Encodes text into integer tokens
- Details: Iteratively checks longest match from vocab, encodes, moves to next character

**decode**
- Overview: Decodes integer tokens back into text
- Details: Looks up each token integer in the decoder dictionary 

**save_vocab_file/load_vocab_file**
- Overview: Save current vocabulary to json file and load vocabulary from json file
- Details: Uses json module to serialize vocabulary dict to json format

## BPE Tokenizer Class Usage Guide

The BPE Tokenizer class allows trainable subword tokenization of text using the Byte Pair Encoding algorithm. Here are some examples of how to use the class:


### Train and save or load lokenizer

**Train and save the tokenizer from scratch:**
```python
train_file = "./training_data.txt"
saved_vocab = "./vocab.json"
tokenizer = BPETokenizer()
tokenizer.train_from_file(train_file, 5000, True)
tokenizer.save_vocab_file(saved_vocab)
```

**Load pretrained tokenizer:**

```python 
saved_vocab = "./vocab.json"
tokenizer = BPETokenizer(vocab_or_json_path=saved_vocab)
```

**Encode text to integers using a trained or loaded tokenizer:**

```python
saved_vocab = "./vocab.json"
tokenizer = BPETokenizer(saved_vocab)
tokens = tokenizer.encode("This is some example text") 
```

**Decode integers to text:**

```python
saved_vocab = "./vocab.json"
tokenizer = BPETokenizer(saved_vocab)
text = tokenizer.decode([123, 456, 789])
```

The tokenizer generates a vocabulary by iteratively merging frequent symbol pairs from the training corpus. Encoding works by matching the longest substrings from text to vocabulary items. Decoding reverses this mapping.

Training on a large, representative corpus produces a vocabulary optimized for general usage. Domain-specific vocabularies may produce better encodings for text in that domain.

The BPE Tokenizer class allows flexible reuse of pretrained vocabularies while also supporting creation of new ones when needed.

If you have any issues or questions please submit in the repo and I will do my best to help out :)

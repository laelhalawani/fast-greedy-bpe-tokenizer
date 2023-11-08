
import json
import os
import string

class BPETokenizer:
    def __init__(self, corpus_or_txt_path=None, vocab_size=None, vocab_or_json_path=None):
        """Constructor to initialize BPE Tokenizer.

        Supports three initialization modes:
        1) Unconfigured: Leave arguments unspecified
        2) Pretrained: Provide pretrained vocab_or_json_path
        3) New vocab: Provide corpus_or_txt_path and vocab_size

        Initializes class attributes including vocab dict with default
        symbols. Encoder and decoder dicts left empty until vocab configured.

        Args:
            corpus_or_txt_path: Path to corpus text file to train vocab on.
            vocab_size: Desired size of new vocabulary to train. 
            vocab_or_json_path: Path to pre-trained vocab json file.

        """
        self._UNK = "<|UNK|>"
        self._PAD = "<|PAD|>"
        self._SOS = "<|SOS|>"
        self._EOS = "<|EOS|>"
        self._default_chars = [self._PAD] + list(string.ascii_letters + string.digits + string.punctuation + " ") + [self._UNK] + [self._SOS] + [self._EOS]
        self.vocab_size = None
        self.vocab = None
        self.encoder_dict = None
        self.decoder_dict = None
        self._init_vocab()
        if vocab_or_json_path:
            if os.path.isfile(vocab_or_json_path):
                if not vocab_or_json_path.endswith(".json"):
                    raise ValueError("Vocab file must be a json file")
                with open(vocab_or_json_path, "r") as f:
                    vocab_dict = json.load(f)
                    vocab_or_json_path = vocab_dict
            self._update_vocab(vocab_or_json_path)
        elif vocab_size and corpus_or_txt_path:
            self._set_vocab_size(vocab_size)
            if os.path.isfile(corpus_or_txt_path):
                if not corpus_or_txt_path.endswith(".txt"):
                    raise ValueError("Corpus file must be a text file")
                self.train_from_file(corpus_or_txt_path, vocab_size)
            else:
                self.train(corpus_or_txt_path, vocab_size)
        else:
            print(f"Initialized unconfigured BPE Tokenizer.")
            self._rebuild_token_dict()

    def _init_vocab(self):
        """Populates vocab dictionary with default special symbols.
        
        Adds PAD, UNK, SOS, EOS tokens as well as ASCII letters, digits,
        punctuation and space character. Initializes all with frequency of 0.
        Also initializes vocab_size to match size of vocab dictionary.

        This provides starting set of symbols for first stage of BPE training.

        """
        vocab_elements = self._default_chars
        vocab = {}
        for element in vocab_elements:
            vocab[element] = 0
        self.vocab_size = len(vocab.keys()) if not self.vocab_size or self.vocab_size < len(vocab.keys()) else self.vocab_size
        self.vocab = vocab
        self._set_vocab_size(len(self.vocab.keys()))
    
    def _update_vocab(self, vocab):
        """Updates internal vocabulary dictionary from external vocab.
        
        For each key/value pair in provided vocab, updates interal vocabulary
        accordingly. If key already exists, value is added to current. New keys
        are added.

        Also updates internal vocab_size attribute to match new dict size.

        Used to load pretrained vocabulary.

        Args:
            vocab: External vocabulary dictionary.

        """
        for key, value in vocab.items():
            self.vocab[key] = value
        print(f"Updated vocab with {len(self.vocab.keys())} elements")
        self._set_vocab_size(len(self.vocab.keys()))
        self._rebuild_token_dict()

    def _set_vocab_size(self, vocab_size):
        if vocab_size >= len(self.vocab.keys()):
            self.vocab_size = vocab_size
        else:
            self.vocab_size = len(self.vocab.keys())
            print(f"Given vocab size {vocab_size} is too small to accomodate the default vocab of {len(self.vocab.keys())}, set to {len(self.vocab.keys())} instead")

    def _rebuild_token_dict(self):
        """Rebuilds encoder and decoder dictionaries based on vocabulary.
        
        Iterates through vocab dictionary to assign unique integer indices 
        incrementally to each symbol. 

        Populates encoder dict that maps symbols to indices.
        Populates decoder dict that maps indices to symbols.

        Special tokens PAD, UNK, SOS, EOS get fixed reserved indices.

        Called whenever vocabulary is updated to refresh mappings.

        """
        encoder_dict = {}
        decoder_dict = {}
        encoder_dict[self._PAD] = 0
        decoder_dict[0] = self._PAD
        encoder_dict[self._UNK] = -999
        decoder_dict[-999] = self._UNK
        encoder_dict[self._SOS] = -1
        decoder_dict[-1] = self._SOS
        encoder_dict[self._EOS] = -2
        decoder_dict[-2] = self._EOS

        i = 1 
        for token_str in self.vocab.keys():
            if token_str in [self._PAD, self._UNK, self._SOS, self._EOS]:
                continue
            encoder_dict[token_str] = i
            decoder_dict[i] = token_str
            i += 1
        self.encoder_dict = encoder_dict
        self.decoder_dict = decoder_dict
        print(f"Token dict created with {len(self.vocab.keys())} elements, encoding and decoding active.")
    
    def _include_corpus_chars(self, corpus):
        """Scans corpus text and adds any new characters to vocabulary.
    
        Checks each character of provided corpus text(s). Any newly encountered
        characters are added to the vocabulary dict with a frequency of 1.

        This is used at start of training to initialize vocabulary with corpus
        characters before beginning BPE merge operations.

        Args:
            corpus: Training corpus text(s) to scan for new characters.

        """
        if isinstance(corpus, list):
            corpus = "".join(corpus)     
        for c in corpus:
            if c in self.vocab.keys():
                self.vocab[c] += 1
            else:
                self.vocab[c] = 1
        print(f"Updated vocab with corpus chars. Vocab size is now {len(self.vocab.keys())}")

    def train(self, corpus, desired_vocab_size:int, word_level = True):
        """Performs core BPE training algorithm on provided corpus.

        Iteratively merges the most frequent pair of adjacent symbols in the
        corpus text to build up vocabulary. 

        If word_level=True, merges are done at space-separated word boundaries.
        Otherwise, all symbol pairs are considered across word boundaries.

        Continues merges until target vocab_size is reached.

        Updates internal vocabulary dictionary with learned symbols and frequencies.

        Args:
            corpus: Training corpus text(s).
            vocab_size: Target size for final vocabulary.
            word_level: Whether to merge within space-separated words.

        """
        _counted_words = {}
        _temp_vocab = self.vocab.copy()
        self._set_vocab_size(desired_vocab_size)
        self._include_corpus_chars(corpus)
        if isinstance(corpus, str):
            corpus = [corpus]
        if word_level:
            words = []
            for c in corpus:
                words.extend(c.split())
            for word in words:
                if word in _counted_words.keys():
                    _counted_words[word] += 1
                else:
                    _counted_words[word] = 1
            corpus = _counted_words.keys()

        while len(_temp_vocab) < desired_vocab_size:
            word_freqs = {}
            for text in corpus:  
                i = 0
                j = 1 
                if word_level:
                    inc = _counted_words[text]
                else:
                    inc = 1          
                while j < len(text) and len(text) > 2:
                    curr_word = text[i:j]
                    next_char = text[j]
                    if curr_word + next_char in _temp_vocab.keys():
                        j += 1
                    else:
                        if curr_word+next_char in word_freqs.keys():
                            word_freqs[curr_word+next_char] += inc
                        else:
                            word_freqs[curr_word+next_char] = inc
                        i = j
                        j += 1
            if word_freqs == {}:
                if len(_temp_vocab) < desired_vocab_size:
                    print(f"Ending early at {len(_temp_vocab)}/{desired_vocab_size}, all combinations exhausted")
                    self._set_vocab_size(len(_temp_vocab))
                break
            best_pair_str = max(word_freqs, key=word_freqs.get)
            occurences = word_freqs[max(word_freqs, key=word_freqs.get)]
            _temp_vocab[best_pair_str] = occurences
            print(f"Target vocab size: {desired_vocab_size} | Current vocab size: {len(_temp_vocab.keys())}")
        self._update_vocab(_temp_vocab)

    def train_from_file(self, corpus_file_path, desired_vocab_size, word_level = True):
        """Helper to perform training directly from a corpus text file.

        Loads training text file, extracts full corpus text, and passes 
        to train() method along with provided vocab_size and word_level settings.

        Args:
            file_path: Path to training corpus text file.
            vocab_size: Target size for final vocabulary.
            word_level: Whether to merge within space-separated words.

        """
        with open(corpus_file_path, "r") as f:
            corpus = [l.strip() for l in f.readlines()]
            self.train(corpus, desired_vocab_size, word_level)    
    
    def save_vocab_file(self, json_file_path:str):
        """Saves current vocabulary dictionary to provided file path as JSON.

        Serializes vocabulary dict as JSON using json module and writes to 
        specified file path.

        Directory is created if it does not exist.

        Args:
            path: Path to output JSON vocabulary file.

        """
        f_dir = os.path.dirname(json_file_path)
        if not os.path.exists(f_dir):
            os.makedirs(f_dir)
        with open(json_file_path, "w") as f:
            json.dump(self.vocab, f)
    
    def load_vocab_file(self, json_file_path:str):
        """Loads vocabulary dictionary from a JSON file at given path.

        Uses json module to deserialize vocabulary dictionary stored in 
        JSON format at specified file path. 

        Loads dictionary into internal vocabulary attribute, overwriting any
        existing vocabulary.

        Args:
            path: Path to input JSON vocabulary file.

        """
        with open(json_file_path, "r") as f:
            vocab = json.load(f)
            self._update_vocab(vocab)

    def decode(self, tokens):
        """Decodes a list of integer tokens into corresponding text.

        Looks up each integer token in the decoder dictionary to recover 
        the corresponding symbol string.

        Concatenates all recovered symbols and returns final text.

        Args:
            tokens: List of integer vocabulary indices.

        Returns:
            Decoded text recovered from integer tokens.

        """
        return "".join(self._decode_chunks(tokens))

    def _decode_chunks(self, tokens):
        """Decodes integer tokens into chunks of text symbols.

        Looks up each integer token in the decoder dictionary
        to recover the corresponding symbol string. 

        Args:
            tokens: List of integer tokens to decode.

        Returns:
            List of decoded symbol strings.

        """
        if not self.decoder_dict:
            raise ValueError(f"Cannot decode without a decoder dictionary!")        
        decoded = []
        for t in tokens:
            decoded.append(self.decoder_dict[t])
        return decoded

    def encode(self, text, pad_to_tokens=0):
        """Encodes text into corresponding integer tokens.

        Iteratively takes the longest matching substring from the vocabulary
        and emits the integer token. Advances to next character and repeats.

        Unknown symbols are replaced with UNK token.

        If pad_to_tokens specified, pads output to given length.

        Args:
            text: Input text to encode to integers.
            pad_to_tokens: If given, pads output to this length.

        Returns:
            List of integer vocabulary indices encoding input text.

        """
        if not self.encoder_dict:
            raise ValueError(f"Cannot encode without an encoder dictionary!")
        tokens = []
        i = 0
        j = 0
        
        end_string = False

        while j <= len(text):  
            if text[i:].startswith(self._PAD):
                tokens.append(self.encoder_dict[self._PAD])
                i += len(self._PAD)
                continue
            elif text[i:].startswith(self._EOS):
                tokens.append(self.encoder_dict[self._EOS])
                i += len(self._EOS)
                continue
            curr_word = text[i:j]
            if j < len(text):
                next_char = text[j]
            else:
                next_char = ""
                end_string = True

            if curr_word+next_char != "" and (curr_word + next_char not in self.vocab or end_string):
                try:
                    tokens.append(self.encoder_dict[curr_word])
                except KeyError:
                    print(f"'{curr_word}' in '{text}' not found in vocab, setting to UNK token")
                    tokens.append(self.encoder_dict[self._UNK])
                i = j
            j += 1
        tokens = self.pad(tokens, pad_to_tokens) if pad_to_tokens > 0 else tokens
        return tokens
    
    def pad(self, tokens:list, desired_length):
        """Pads token list to desired length by adding PAD tokens.
    
        Adds SOS and EOS tokens to beginning and end. 
        Calculates number of PAD tokens needed to reach desired length.
        PAD tokens are appended to fill to desired length.

        Args:
            tokens: List of tokens to pad.
            desired_length: Desired length to pad tokens to.

        Returns:
            Padded token list with SOS, EOS, and PAD tokens added.

        Raises:
            ValueError if tokens exceed desired length.

        """
        start_token = self.encoder_dict[self._SOS]
        end_token = self.encoder_dict[self._EOS]
        pad_token = self.encoder_dict[self._PAD]
        if len(tokens)+2 >= desired_length:
            raise ValueError(f"Cannot pad {len(tokens)} tokens to desired length of {desired_length}!")
        else:
            tokens.insert(0, start_token)
            tokens.extend([end_token])
            [tokens.extend([pad_token]) for _ in range (desired_length - len(tokens))]
        return tokens
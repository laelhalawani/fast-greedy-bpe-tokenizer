from bpe_tokenizer import BPETokenizer

#To use this example easiest
# 1) Put your corpus of text as training_data.txt in the same dir 
# 2) Run the script
# 3) Comment out lines 19 and 22 (train and save)
# 4) Run the script again to use your trained tokenizer


# Path to training data 
train_file = "./training_data.txt"  

# Output path for saving trained vocabulary
vocab_file = "./trained_vocab/vocab5000.json"

# Create tokenizer instance, unconfigured (you can also pass a vocab_file to load configured BPE or do it later)
tok = BPETokenizer()  

# Train new vocabulary (comment this line after training and saving)
tok.train_from_file(train_file, 5000, True) 

# Save newly trained vocabulary (comment this line after training and saving)
tok.save_vocab_file(vocab_file)

# Load vocabulary
tok.load_vocab_file(vocab_file)

# Input text to encode
inp = 'TEST INPUT GOES HERE'  

# Encode text to integer tokens
enc = tok.encode(inp, pad_to_tokens=64)

# Decode integer tokens to symbol strings 
dec = tok._decode_chunks(enc)

# Decode tokens back to full text
dec_str = tok.decode(enc)

# Print results
print(f"Input: {inp}\nEncoded: {enc} | l: {len(enc)}\nDecoded: {dec}\nDecoded String: {dec_str}")
import torch
from transformers import BertTokenizer, BertModel

# Load a pretrained BERT model and tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)

# Sample sentence
sentence = "Transformers are powerful models for NLP tasks."

### Step 1: Tokenize and Encode the Sentence
tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
input_ids = tokens["input_ids"]

print("\nTokenized Input IDs:", input_ids)

### Step 2: Extract Word Embeddings
with torch.no_grad():  # Disable gradient calculations (for inference)
    outputs = model(**tokens)

# Extract last hidden states (word embeddings)
hidden_states = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

print("\nShape of Hidden States:", hidden_states.shape)  # Example: (1, 10, 768)

### Step 3: Convert Token IDs to Words (Decoded Tokens)
decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print("\nDecoded Tokens:", decoded_tokens)

### Step 4: Extract Embeddings for Each Token
token_embeddings = hidden_states.squeeze(0)  # Remove batch dimension
embeddings_dict = {token: embedding.tolist() for token, embedding in zip(decoded_tokens, token_embeddings)}

# Display first few token embeddings
print("\nSample Token Embeddings:")
for token, embedding in list(embeddings_dict.items())[:5]:  # Show first 5 tokens
    print(f"{token}: {embedding[:5]} ... (Truncated)")


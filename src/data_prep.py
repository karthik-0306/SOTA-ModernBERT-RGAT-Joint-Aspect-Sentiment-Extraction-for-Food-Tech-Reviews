import spacy
import torch
from transformers import AutoTokenizer

class ABSAPreprocessor:
    def __init__(self, model_name="answerdotai/ModernBERT-base"):
        """
        Initializes the ModernBERT tokenizer and spaCy linguistic model.
        """
        print(f"Loading {model_name} and spaCy...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_lg")
        
        # 7 key grammatical relations for sentiment flow in Food-Tech reviews
        self.rel_map = {
            'nsubj': 0,    # Subject
            'amod': 1,     # Adjective modifier
            'obj': 2,      # Direct object
            'advmod': 3,   # Adverb modifier
            'neg': 4,      # Negation (not, never)
            'compound': 5, # Multi-word concepts (ice cream)
            'conj': 6      # Connections (and, but)
        }

    def get_adj_tensor(self, text, max_len=128):
        """
        Converts text into a fixed-size 3D Adjacency Tensor.
        Shape: [7, max_len, max_len]
        """
        # 1. Linguistic Analysis
        doc = self.nlp(text)
        
        # 2. ModernBERT Tokenization with Offset Mapping
        encoding = self.tokenizer(
            text, 
            return_offsets_mapping=True, 
            add_special_tokens=True, 
            truncation=True, 
            max_length=max_len
        )
        
        tokens = encoding.input_ids
        offsets = encoding.offset_mapping
        
        # 3. Initialize Fixed-Size Tensor (Zeros)
        adj = torch.zeros((len(self.rel_map), max_len, max_len))
        
        # 4. Alignment Map: Link spaCy word indices to BERT token indices
        spacy_to_bert = [[] for _ in range(len(doc))]
        
        for b_idx, (start, end) in enumerate(offsets):
            if start == end: continue 
            
            for word in doc:
                # Check if the BERT token's character range falls within the spaCy word
                if word.idx <= start < (word.idx + len(word.text)):
                    spacy_to_bert[word.i].append(b_idx)
                    break

        # 5. Populate the Adjacency Matrix
        for token in doc:
            if token.dep_ in self.rel_map:
                rel_idx = self.rel_map[token.dep_]
                
                child_sub_tokens = spacy_to_bert[token.i]
                head_sub_tokens = spacy_to_bert[token.head.i]
                
                # Create a bridge between all sub-tokens of the related words
                for c_i in child_sub_tokens:
                    for h_i in head_sub_tokens:
                        if c_i < max_len and h_i < max_len:
                            adj[rel_idx, c_i, h_i] = 1
        
        return adj

if __name__ == "__main__":
    # Internal Sanity Check
    prep = ABSAPreprocessor()
    example = "The spicy ramen was incredibly delicious."
    output = prep.get_adj_tensor(example)
    print(f"Data Prep Sanity Check: SUCCESS")
    print(f"Final Tensor Shape: {output.shape}")
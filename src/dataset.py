import torch
from torch.utils.data import Dataset
from src.data_prep import ABSAPreprocessor

class AspectDataset(Dataset):
    def __init__(self, data_list, preprocessor):
        """
        data_list: A list of dictionaries, e.g., 
                   [{'text': 'Great food', 'aspect': 'food', 'label': 1}]
        preprocessor: Your ABSAPreprocessor instance
        """
        self.data = data_list
        self.preprocessor = preprocessor
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2, 'conflict': 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # 1.3D Adjacency Tensor from preprocessor
        adj_tensor = self.preprocessor.get_adj_tensor(text)
        
        # 2. Get ModernBERT input IDs
        encoding = self.preprocessor.tokenizer(
            text, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=128
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'adj_matrix': adj_tensor,
            'label': torch.tensor(self.label_map[item['sentiment']])
        }
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class RGATLayer(nn.Module):
    """
    Relational Graph Attention Layer: 
    Refines word embeddings based on the 7 grammatical relations.
    """
    def __init__(self, in_dim, out_dim, num_relations=7):
        super().__init__()
        self.num_relations = num_relations
        self.out_dim = out_dim
        
        # Linear transformations for each relation
        self.weight_mats = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)
        ])
        
        # Attention mechanism
        self.attn_linear = nn.Linear(out_dim * 2, 1)
        self.gate = nn.Linear(in_dim, out_dim)

    def forward(self, h, adj):
        """
        h: BERT embeddings [batch, 128, 768]
        adj: Adjacency matrices [batch, 7, 128, 128]
        """
        batch_size, seq_len, _ = h.size()
        outputs = []

        # Process each grammatical relation separately
        for i in range(self.num_relations):
            # Transform nodes for this specific relation
            rel_h = self.weight_mats[i](h) # [batch, 128, out_dim]
            
            # Apply adjacency filter
            # (batch, 128, 128) @ (batch, 128, out_dim)
            context = torch.matmul(adj[:, i, :, :], rel_h)
            outputs.append(context)

        # Aggregate all relations (Sum)
        combined = torch.stack(outputs, dim=0).sum(dim=0)
        
        # Apply a Gated Skip-Connection (Keeps BERT's original knowledge)
        res = F.relu(combined + self.gate(h))
        return res

class ModernBERT_RGAT(nn.Module):
    def __init__(self, model_name="answerdotai/ModernBERT-base", num_classes=4):
        super().__init__()
        # 1. Semantic Backbone
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 2. Structural Layer (RGAT)
        self.rgat = RGATLayer(in_dim=768, out_dim=768)
        
        # 3. Final Prediction Head (Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, adj_matrix):
        # A. Get BERT Embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # [batch, 128, 768]
        
        # B. Enhance with RGAT (Graph logic)
        graph_output = self.rgat(sequence_output, adj_matrix)
        
        # C. Pooling: Take the [CLS] token (index 0) which now has graph info
        cls_output = graph_output[:, 0, :]
        
        # D. Classify
        logits = self.classifier(cls_output)
        return logits
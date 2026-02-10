import torch
from torch.utils.data import DataLoader
from src.utils import parse_absa_xml
from src.data_prep import ABSAPreprocessor
from src.dataset import AspectDataset

def main():
    print("--- Starting Data Check ---")
    
    xml_path = "restaurants-trial.xml" 
    
    # 2. Parse the XML
    raw_data = parse_absa_xml(xml_path)
    print(f"Total Aspect Examples found: {len(raw_data)}")

    # 3. Safety check: Exit if no data found
    if len(raw_data) == 0:
        print("Error: No data found! Check if restaurants-trial.xml has content.")
        return

    # 4. Initialize Preprocessor (This loads ModernBERT)
    print("Initializing Preprocessor...")
    preprocessor = ABSAPreprocessor()
    
    # 5. Create Dataset and DataLoader
    dataset = AspectDataset(raw_data, preprocessor)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 6. Test the loop
    print("Attempting to load the first batch...")
    for batch in train_loader:
        print("\n--- XML Data Check SUCCESS ---")
        print(f"Input IDs Shape (Batch, SeqLen): {batch['input_ids'].shape}")
        print(f"Adj Matrix Shape (Batch, Rel, Seq, Seq): {batch['adj_matrix'].shape}")
        print(f"Labels in this batch: {batch['label']}")
        break

if __name__ == "__main__":
    main()
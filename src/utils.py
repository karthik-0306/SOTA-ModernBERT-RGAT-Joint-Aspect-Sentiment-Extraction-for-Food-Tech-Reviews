import xml.etree.ElementTree as ET

def parse_absa_xml(file_path):
    print(f"--- Debug: Opening {file_path} ---")
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"--- Debug Error: Could not read XML file: {e} ---")
        return []

    data = []
    sentences = root.findall('sentence')
    print(f"--- Debug: Found {len(sentences)} <sentence> tags ---")

    for sentence in sentences:
        text_elem = sentence.find('text')
        if text_elem is None:
            continue
        text = text_elem.text
        
        aspect_terms = sentence.find('aspectTerms')
        if aspect_terms is not None:
            terms = aspect_terms.findall('aspectTerm')
            for at in terms:
                data.append({
                    'text': text,
                    'aspect': at.get('term'),
                    'sentiment': at.get('polarity')
                })
        else:
            # If no aspect terms, we can use aspect categories as a backup
            categories = sentence.find('aspectCategories')
            if categories is not None:
                for cat in categories.findall('aspectCategory'):
                    data.append({
                        'text': text,
                        'aspect': cat.get('category'),
                        'sentiment': cat.get('polarity')
                    })
    
    print(f"--- Debug: Successfully extracted {len(data)} total items ---")
    return data
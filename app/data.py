from datasets import load_dataset


def load_isic_dataset(split="train", num_samples=None):
    """
    Load ISIC 2019 dataset from Hugging Face.
    
    Args:
        split (str): Dataset split to load ("train", "validation", "test")
        num_samples (int): Number of samples to load (None for all)
        
    Returns:
        Dataset: Loaded ISIC dataset
    """
    print(f"Loading ISIC 2019 dataset ({split} split)...")
    
    try:
        dataset = load_dataset("MKZuziak/ISIC_2019_224", split=split)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
        print(f"Loaded {len(dataset)} samples from ISIC dataset")
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have internet connection and the datasets library installed")
        return None


def get_comprehensive_dermatology_concepts():
    """
    Get a comprehensive list of dermatology concepts for annotation.
    
    Returns:
        list: List of dermatology concepts
    """
    return [
        # Primary skin cancers
        "melanoma",
        "basal cell carcinoma", 
        "squamous cell carcinoma",
        
        # Precancerous lesions
        "actinic keratosis",
        "Bowen's disease",
        "atypical nevus",
        
        # Benign lesions
        "seborrheic keratosis",
        "dermatofibroma",
        "nevus",
        "solar lentigo",
        "cherry angioma",
        
        # Vascular lesions
        "vascular lesion",
        "hemangioma",
        "port wine stain",
        
        # Morphological features
        "pigmented lesion",
        "asymmetric lesion",
        "irregular border",
        "color variation",
        "ulceration",
        "scaling",
        "crusting",
        
        # Inflammatory changes
        "inflammatory changes",
        "erythema",
        "papule",
        "nodule",
        
        # Infectious conditions
        "bacterial infection",
        "fungal infection",
        "viral wart",
        
        # Other conditions
        "eczema",
        "psoriasis",
        "contact dermatitis"
    ]
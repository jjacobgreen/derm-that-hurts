#!/usr/bin/env python3
"""
MONET Inference Script for Dermatology Images (Hugging Face Version)
====================================================================

This script demonstrates how to use the MONET model from Hugging Face for automatic 
concept annotation on dermatology images. It loads the ISIC 2019 dataset from 
Hugging Face and performs inference on a single image.
"""
from model import MONETInference
from data import load_isic_dataset, get_comprehensive_dermatology_concepts
from utils import analyze_prediction_confidence
import numpy as np
import torch

def main():
    """Main function demonstrating MONET inference."""
    
    # Initialize MONET
    monet = MONETInference(device="auto")
    
    # Load ISIC dataset
    dataset = load_isic_dataset(split="train", num_samples=100)
    
    if dataset is None:
        print("Failed to load dataset. Creating demo with placeholder...")
        # You could add code here to use a local image instead
        return
    
    # Select a random image from the dataset
    sample_idx = np.random.randint(0, len(dataset))
    sample = dataset[sample_idx]
    
    # Get image and any available metadata
    image = sample['image']
    print(f"Processing sample {sample_idx}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    
    # Print any available labels/metadata
    for key in sample.keys():
        if key != 'image':
            print(f"{key}: {sample[key]}")
    
    # Get comprehensive dermatology concepts
    dermatology_concepts = get_comprehensive_dermatology_concepts()
    
    print(f"\nAnnotating {len(dermatology_concepts)} dermatology concepts...")
    
    # Perform concept annotation
    results = monet.annotate_concepts(image, dermatology_concepts)
    
    # Analyze results
    analysis = analyze_prediction_confidence(results, threshold=0.1)
    
    # Print results
    print(f"\n" + "="*60)
    print(f"MONET CONCEPT ANNOTATION RESULTS")
    print(f"="*60)
    print(f"Max score: {analysis['max_score']:.4f}")
    print(f"Mean score: {analysis['mean_score']:.4f}")
    print(f"High confidence concepts (>0.1): {analysis['high_confidence_count']}")
    
    print(f"\nTop 10 Concept Annotations:")
    print("-" * 50)
    for i, item in enumerate(results['ranked_concepts'][:10]):
        confidence = "HIGH" if item['score'] > 0.1 else "LOW"
        print(f"{i+1:2d}. {item['concept']:<25} | {item['score']:.4f} | {confidence}")
    
    # Visualize results
    print("\nGenerating visualization...")
    monet.visualize_results(image, results, top_k=8)
    
    # Show high-confidence concepts
    high_confidence_concepts = [
        item for item in results['ranked_concepts'] 
        if item['score'] > 0.1
    ]
    
    if high_confidence_concepts:
        print(f"\nHigh-confidence concepts (score > 0.1):")
        for item in high_confidence_concepts:
            print(f"  • {item['concept']}: {item['score']:.4f}")
    else:
        print(f"\nNo high-confidence concepts found (all scores < 0.1)")
        print("Top 3 concepts by score:")
        for item in analysis['top_3_concepts']:
            print(f"  • {item['concept']}: {item['score']:.4f}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 70)
    print("MONET Dermatology Concept Annotation (Hugging Face Version)")
    print("=" * 70)
    print()
    
    main()
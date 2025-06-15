import numpy as np


def analyze_prediction_confidence(results, threshold=0.1):
    """
    Analyze prediction confidence and provide insights.
    
    Args:
        results (dict): Results from concept annotation
        threshold (float): Minimum score threshold for high confidence
        
    Returns:
        dict: Analysis results
    """
    scores = [item['score'] for item in results['ranked_concepts']]
    
    analysis = {
        'max_score': max(scores),
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'high_confidence_count': len([s for s in scores if s > threshold]),
        'top_3_concepts': results['ranked_concepts'][:3]
    }
    
    return analysis
"""
Confusion matrix analysis utilities for Classification Boundary Warning
"""

import numpy as np
import os
from config import CONFUSION_MATRIX_DATA_PATH, CLASS_NAMES


# Tips for commonly confused class pairs
CONFUSION_TIPS = {
    ('brown-glass', 'green-glass'): {
        'tip': '💡 Tip: The model is uncertain between [brown glass] and [green glass]. Please check: What is the actual color of the glass? Is it brown/amber or green?',
        'checks': ['Color: Brown/amber vs Green', 'Transparency level', 'Container type']
    },
    ('brown-glass', 'white-glass'): {
        'tip': '💡 Tip: The model is uncertain between [brown glass] and [white/clear glass]. Please check: What is the actual color? Is it brown/amber or clear/transparent?',
        'checks': ['Color: Brown/amber vs Clear', 'Transparency', 'Container type']
    },
    ('green-glass', 'white-glass'): {
        'tip': '💡 Tip: The model is uncertain between [green glass] and [white/clear glass]. Please check: What is the actual color? Is it green or clear/transparent?',
        'checks': ['Color: Green vs Clear', 'Transparency level', 'Container type']
    },
    ('brown-glass', 'plastic'): {
        'tip': '💡 Tip: The model is uncertain between [brown glass] and [plastic]. Please check: Is the item fully transparent? Does it have a risk of breaking? Can you feel if it\'s glass (hard, brittle) or plastic (flexible)?',
        'checks': ['Transparency: Fully transparent?', 'Rigidity: Hard/brittle (glass) vs Flexible (plastic)', 'Sound: Glass makes a "ting" sound when tapped']
    },
    ('green-glass', 'plastic'): {
        'tip': '💡 Tip: The model is uncertain between [green glass] and [plastic]. Please check: Is the item fully transparent? Does it have a risk of breaking? Can you feel if it\'s glass (hard, brittle) or plastic (flexible)?',
        'checks': ['Transparency: Fully transparent?', 'Rigidity: Hard/brittle (glass) vs Flexible (plastic)', 'Sound: Glass makes a "ting" sound when tapped']
    },
    ('white-glass', 'plastic'): {
        'tip': '💡 Tip: The model is uncertain between [white/clear glass] and [plastic]. Please check: Is the item fully transparent? Does it have a risk of breaking? Can you feel if it\'s glass (hard, brittle) or plastic (flexible)?',
        'checks': ['Transparency: Fully transparent?', 'Rigidity: Hard/brittle (glass) vs Flexible (plastic)', 'Sound: Glass makes a "ting" sound when tapped', 'Weight: Glass is typically heavier']
    },
    ('paper', 'cardboard'): {
        'tip': '💡 Tip: The model is uncertain between [paper] and [cardboard]. Please check: What is the thickness? Is it thin (paper) or thick/corrugated (cardboard)?',
        'checks': ['Thickness: Thin (paper) vs Thick (cardboard)', 'Structure: Flat (paper) vs Corrugated (cardboard)', 'Rigidity: Flexible (paper) vs Stiff (cardboard)']
    },
    ('clothes', 'shoes'): {
        'tip': '💡 Tip: The model is uncertain between [clothes] and [shoes]. Please check: Is it a wearable item? Does it cover the feet (shoes) or body (clothes)?',
        'checks': ['Item type: Footwear (shoes) vs Clothing (clothes)', 'Shape and structure', 'Material composition']
    },
    ('metal', 'plastic'): {
        'tip': '💡 Tip: The model is uncertain between [metal] and [plastic]. Please check: Is it magnetic? Does it feel heavy and cold (metal) or light and warm (plastic)?',
        'checks': ['Magnetism: Metal is magnetic', 'Weight: Metal is typically heavier', 'Temperature: Metal feels cold, plastic feels warm', 'Sound: Metal makes a "clang" sound']
    },
}


def load_confusion_matrix():
    """
    Load confusion matrix data from saved file.
    
    Returns:
        confusion_matrix: numpy array of confusion matrix
        class_names: list of class names
    """
    if not os.path.exists(CONFUSION_MATRIX_DATA_PATH):
        return None, None
    
    try:
        data = np.load(CONFUSION_MATRIX_DATA_PATH, allow_pickle=True)
        cm = data['confusion_matrix']
        class_names = data['class_names'].tolist()
        return cm, class_names
    except Exception as e:
        print(f"Error loading confusion matrix: {e}")
        return None, None


def get_confused_pairs(confusion_matrix, class_names, threshold=0.15):
    """
    Identify pairs of classes that are commonly confused based on confusion matrix.
    
    Args:
        confusion_matrix: Normalized confusion matrix
        class_names: List of class names
        threshold: Minimum normalized confusion value to consider (default: 0.15 = 15%)
        
    Returns:
        List of tuples: [(class1, class2, confusion_value), ...]
    """
    if confusion_matrix is None or class_names is None:
        return []
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float')
    row_sums = cm_normalized.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm_normalized = cm_normalized / row_sums[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    confused_pairs = []
    
    # Find pairs where confusion is above threshold (excluding diagonal)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:  # Skip diagonal (correct predictions)
                confusion_value = cm_normalized[i, j]
                if confusion_value >= threshold:
                    # Add both directions (i->j and j->i) if significant
                    pair = tuple(sorted([class_names[i], class_names[j]]))
                    confused_pairs.append((class_names[i], class_names[j], confusion_value))
    
    # Sort by confusion value (descending)
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return confused_pairs


def check_boundary_warning(all_probs, idx_to_class, dataset_classes, confidence_threshold=0.4, diff_threshold=0.15):
    """
    Check if prediction is in a boundary region between confused classes.
    
    Args:
        all_probs: Array of probabilities for all classes
        idx_to_class: Dictionary mapping index to class name
        dataset_classes: List of class names in dataset order
        confidence_threshold: Minimum confidence for both classes (default: 0.4 = 40%)
        diff_threshold: Maximum difference between top 2 predictions (default: 0.15 = 15%)
        
    Returns:
        Dictionary with warning info if boundary detected, None otherwise
    """
    # Get top 2 predictions
    top2_indices = np.argsort(all_probs)[-2:][::-1]
    top2_probs = all_probs[top2_indices]
    
    # Check if both are above threshold and close together
    if top2_probs[0] >= confidence_threshold and top2_probs[1] >= confidence_threshold:
        prob_diff = top2_probs[0] - top2_probs[1]
        if prob_diff <= diff_threshold:
            # Get class names
            class1_idx = top2_indices[0]
            class2_idx = top2_indices[1]
            
            if idx_to_class is not None:
                class1 = idx_to_class.get(class1_idx, dataset_classes[class1_idx] if class2_idx < len(dataset_classes) else "Unknown")
                class2 = idx_to_class.get(class2_idx, dataset_classes[class2_idx] if class2_idx < len(dataset_classes) else "Unknown")
            else:
                class1 = dataset_classes[class1_idx] if class1_idx < len(dataset_classes) else "Unknown"
                class2 = dataset_classes[class2_idx] if class2_idx < len(dataset_classes) else "Unknown"
            
            # Check if this is a known confused pair
            pair_key1 = tuple(sorted([class1, class2]))
            pair_key2 = (class1, class2)
            pair_key3 = (class2, class1)
            
            tip_info = None
            if pair_key1 in CONFUSION_TIPS:
                tip_info = CONFUSION_TIPS[pair_key1]
            elif pair_key2 in CONFUSION_TIPS:
                tip_info = CONFUSION_TIPS[pair_key2]
            elif pair_key3 in CONFUSION_TIPS:
                tip_info = CONFUSION_TIPS[pair_key3]
            else:
                # Generic tip for any confused pair
                tip_info = {
                    'tip': f'💡 Tip: The model is uncertain between [{class1}] and [{class2}]. Please check the item characteristics carefully.',
                    'checks': ['Visual appearance', 'Material properties', 'Physical characteristics']
                }
            
            return {
                'class1': class1,
                'class2': class2,
                'prob1': top2_probs[0],
                'prob2': top2_probs[1],
                'prob_diff': prob_diff,
                'tip': tip_info['tip'],
                'checks': tip_info.get('checks', [])
            }
    
    return None


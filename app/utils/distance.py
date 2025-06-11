import numpy as np
from typing import List, Tuple, Dict
from models.sequences import ParsedSequence

def calculate_jukes_cantor_distance(sequences: List[ParsedSequence]) -> Tuple[np.ndarray, Dict, Dict]:
    """Calculate Jukes-Cantor distance matrix with proper model parameters"""
    n_seqs = len(sequences)
    distance_matrix = np.zeros((n_seqs, n_seqs))
    
    # Model parameters
    total_comparisons = 0
    total_transitions = 0
    total_transversions = 0
    
    for i in range(n_seqs):
        for j in range(i+1, n_seqs):
            seq1 = sequences[i].sequence.upper()
            seq2 = sequences[j].sequence.upper()
            
            # Calculate p-distance and substitution types
            min_len = min(len(seq1), len(seq2))
            differences = 0
            transitions = 0
            transversions = 0
            
            for k in range(min_len):
                if seq1[k] != seq2[k] and seq1[k] in 'ATGC' and seq2[k] in 'ATGC':
                    differences += 1
                    
                    # Count transitions (A<->G, C<->T) and transversions
                    if (seq1[k] == 'A' and seq2[k] == 'G') or (seq1[k] == 'G' and seq2[k] == 'A') or \
                       (seq1[k] == 'C' and seq2[k] == 'T') or (seq1[k] == 'T' and seq2[k] == 'C'):
                        transitions += 1
                    else:
                        transversions += 1
            
            p_distance = differences / min_len if min_len > 0 else 0
            
            # Jukes-Cantor correction
            if p_distance < 0.75:
                jc_distance = -0.75 * np.log(1 - 4/3 * p_distance)
            else:
                jc_distance = 3.0  # Maximum distance for highly diverged sequences
            
            distance_matrix[i, j] = distance_matrix[j, i] = jc_distance
            
            # Accumulate for model parameters
            total_comparisons += 1
            total_transitions += transitions
            total_transversions += transversions
    
    # Calculate model parameters
    model_parameters = {
        'alpha': 1.0,  # Shape parameter (assumed for Jukes-Cantor)
        'substitution_rate': np.mean(distance_matrix[np.triu_indices_from(distance_matrix, k=1)])
    }
    
    substitution_rates = {
        'transition_rate': total_transitions / total_comparisons if total_comparisons > 0 else 0.0,
        'transversion_rate': total_transversions / total_comparisons if total_comparisons > 0 else 0.0,
        'ts_tv_ratio': total_transitions / total_transversions if total_transversions > 0 else 2.0
    }
    
    return distance_matrix, model_parameters, substitution_rates
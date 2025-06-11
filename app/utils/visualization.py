"""
Visualization utilities for creating phylogenetic and genetic analysis plots
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import io
from typing import List, Tuple, Dict
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from app.models.sequences import ParsedSequence

def extract_species_groups(sequence_ids: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Extract species groups and assign colors"""
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A2B', 
              '#6A994E', '#BC4749', '#7209B7', '#F72585', '#4CC9F0',
              '#7B68EE', '#32CD32', '#FF6347', '#FFD700', '#DDA0DD']
    
    species_groups = {}
    species_colors = {}
    color_index = 0
    
    for seq_id in sequence_ids:
        # Extract species name from sequence ID
        parts = seq_id.replace('_', ' ').split()
        if len(parts) >= 2:
            species = f"{parts[0]} {parts[1]}"
        else:
            species = parts[0] if parts else seq_id
            
        if species not in species_groups:
            species_groups[species] = []
            species_colors[species] = colors[color_index % len(colors)]
            color_index += 1
        species_groups[species].append(seq_id)
    
    return species_groups, species_colors

def create_dendrogram(distance_matrix: np.ndarray, sequence_ids: List[str]) -> bytes:
    """Create improved phylogenetic dendrogram"""
    # Convert distance matrix to condensed form
    condensed_distances = []
    n = len(distance_matrix)
    for i in range(n):
        for j in range(i+1, n):
            condensed_distances.append(distance_matrix[i, j])
    
    # Perform hierarchical clustering using UPGMA (average linkage)
    linkage_matrix = linkage(condensed_distances, method='average')
    
    # Create larger figure for better readability
    plt.figure(figsize=(16, max(12, len(sequence_ids) * 0.8)))
    
    # Extract species groups and colors
    species_groups, species_colors = extract_species_groups(sequence_ids)
    
    # Create dendrogram with custom colors
    dend = dendrogram(
        linkage_matrix, 
        labels=sequence_ids, 
        orientation='right',
        leaf_font_size=10,
        color_threshold=0.7*max(linkage_matrix[:,2]),
        above_threshold_color='gray'
    )
    
    # Customize leaf labels with species colors
    ax = plt.gca()
    ylbls = ax.get_ymajorticklabels()
    
    for lbl in ylbls:
        val = lbl.get_text()
        # Find species for this sequence
        for species, sequences in species_groups.items():
            if val in sequences:
                lbl.set_color(species_colors[species])
                lbl.set_fontweight('bold')
                break
    
    # Add title and labels
    plt.title('Phylogenetic Dendrogram (UPGMA Method)\nSpecies Groups Color-Coded', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Genetic Distance (Jukes-Cantor)', fontsize=14, fontweight='bold')
    plt.ylabel('Species/Sequences', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Create legend for species groups
    legend_elements = []
    for species, color in species_colors.items():
        legend_elements.append(patches.Patch(color=color, label=f"{species} (n={len(species_groups[species])})"))
    
    # Add legend if there are multiple species
    if len(species_groups) > 1:
        plt.legend(handles=legend_elements, 
                  bbox_to_anchor=(1.05, 1), 
                  loc='upper left',
                  fontsize=10,
                  title="Species Groups",
                  title_fontsize=12,
                  frameon=True,
                  fancybox=True,
                  shadow=True)
    
    # Add distance markers
    max_distance = max(linkage_matrix[:,2])
    distance_markers = np.linspace(0, max_distance, 6)
    for marker in distance_markers[1:-1]:
        plt.axvline(x=marker, color='red', linestyle=':', alpha=0.5, linewidth=0.8)
        plt.text(marker, plt.ylim()[1] * 0.95, f'{marker:.3f}', 
                rotation=90, fontsize=8, ha='right', va='top', color='red')
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    buffer.seek(0)
    plt.close()
    
    return buffer.getvalue()

def create_heatmap(distance_matrix: np.ndarray, sequence_ids: List[str]) -> bytes:
    """Create improved distance matrix heatmap"""
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        distance_matrix, 
        xticklabels=sequence_ids, 
        yticklabels=sequence_ids,
        annot=True, 
        fmt='.3f', 
        cmap='viridis',
        square=True,
        cbar_kws={'label': 'Genetic Distance'}
    )
    
    plt.title('Genetic Distance Matrix Heatmap\n(Jukes-Cantor Model)', fontsize=16, fontweight='bold')
    plt.xlabel('Sequences', fontsize=12)
    plt.ylabel('Sequences', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    plt.close()
    
    return buffer.getvalue()

def create_pca(sequences: List[ParsedSequence]) -> Tuple[bytes, Dict, Dict]:
    """Create improved PCA visualization with sequence-based clustering"""
    # Encode sequences numerically
    encoding = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    max_len = max(len(seq.sequence) for seq in sequences)
    
    matrix = np.full((len(sequences), max_len), -1, dtype=float)
    
    for i, seq in enumerate(sequences):
        for j, nucleotide in enumerate(seq.sequence):
            if nucleotide.upper() in encoding:
                matrix[i, j] = encoding[nucleotide.upper()]
    
    # Handle missing data
    for col in range(matrix.shape[1]):
        mask = matrix[:, col] != -1
        if np.any(mask):
            mean_val = np.mean(matrix[mask, col])
            matrix[matrix[:, col] == -1, col] = mean_val
    
    # Remove constant columns
    var_mask = np.var(matrix, axis=0) > 0
    matrix = matrix[:, var_mask]
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(matrix)
    
    # Perform PCA
    pca = PCA(n_components=min(len(sequences)-1, matrix.shape[1], 10))
    pca_result = pca.fit_transform(scaled_data)
    
    # Perform clustering
    if len(sequences) > 3:
        max_clusters = min(8, len(sequences)//2)
        if max_clusters >= 2:
            # Find optimal number of clusters using elbow method
            inertias = []
            K_range = range(2, max_clusters + 1)
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(pca_result[:, :min(3, pca_result.shape[1])])
                inertias.append(kmeans.inertia_)
            
            # Use elbow method to find optimal clusters
            if len(inertias) > 1:
                diffs = np.diff(inertias)
                if len(diffs) > 1:
                    diffs2 = np.diff(diffs)
                    n_clusters = K_range[np.argmax(diffs2) + 1] if len(diffs2) > 0 else K_range[0]
                else:
                    n_clusters = K_range[0]
            else:
                n_clusters = K_range[0]
        else:
            n_clusters = 2
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(pca_result[:, :min(3, pca_result.shape[1])])
        
        if len(set(clusters)) > 1:
            silhouette_avg = silhouette_score(pca_result[:, :2], clusters)
        else:
            silhouette_avg = 0.0
    else:
        clusters = np.zeros(len(sequences))
        silhouette_avg = 0.0
        n_clusters = 1
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    unique_clusters = np.unique(clusters)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
    
    # Plot by sequence-based clusters
    for i, cluster_id in enumerate(unique_clusters):
        mask = clusters == cluster_id
        cluster_sequences = [seq.id for j, seq in enumerate(sequences) if mask[j]]
        
        plt.scatter(
            pca_result[mask, 0], 
            pca_result[mask, 1], 
            c=[colors[i]], 
            label=f'Cluster {int(cluster_id)+1} (n={sum(mask)})', 
            alpha=0.7, 
            s=100,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Add sequence labels
    for i, seq in enumerate(sequences):
        plt.annotate(
            seq.id, 
            (pca_result[i, 0], pca_result[i, 1]),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.title('Principal Component Analysis\nSequence-Based Genetic Clustering', 
              fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to bytes
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    plt.close()
    
    # Prepare return data
    variance_explained = {
        'pc1': float(pca.explained_variance_ratio_[0]),
        'pc2': float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else 0.0,
        'total_variance_first_two': float(np.sum(pca.explained_variance_ratio_[:2]))
    }
    
    # Create cluster summary with sequence information
    cluster_summary = {}
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        cluster_sequences = [sequences[j].id for j in range(len(sequences)) if mask[j]]
        cluster_summary[f'cluster_{int(cluster_id)+1}'] = {
            'sequence_ids': cluster_sequences,
            'size': int(sum(mask))
        }
    
    clustering_results = {
        'n_clusters': int(n_clusters),
        'silhouette_score': float(silhouette_avg),
        'cluster_assignments': clusters.tolist(),
        'cluster_summary': cluster_summary,
        'clustering_method': 'sequence_similarity'
    }
    
    return buffer.getvalue(), variance_explained, clustering_results
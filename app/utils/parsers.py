from typing import List, Dict, Any
from datetime import datetime
from app.models.sequences import ParsedSequence, UploadedData
from app.core.exceptions import FileValidationError

def parse_fasta_content(content: str, filename: str = "demo.fasta") -> UploadedData:
    """Parse FASTA content and extract metadata from headers"""
    if not content.strip():
        raise FileValidationError("File is empty")
    
    if not content.strip().startswith('>'):
        raise FileValidationError("Invalid FASTA format - must start with '>'")
    
    sequences = []
    total_length = 0
    
    lines = content.strip().split('\n')
    current_seq = ""
    current_header = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('>'):
            if current_header and current_seq:
                # Process previous sequence
                seq_data = parse_sequence_header(current_header, current_seq)
                sequences.append(seq_data)
                total_length += len(current_seq)
            
            current_header = line[1:].strip()  # Remove '>'
            current_seq = ""
        else:
            # Validate nucleotide sequence
            valid_chars = set('ATGCNatgcn-')
            if not all(c in valid_chars for c in line):
                raise FileValidationError(f"Invalid characters in sequence. Only A, T, G, C, N, and - are allowed.")
            current_seq += line.upper()
    
    # Process last sequence
    if current_header and current_seq:
        seq_data = parse_sequence_header(current_header, current_seq)
        sequences.append(seq_data)
        total_length += len(current_seq)
    
    if len(sequences) == 0:
        raise FileValidationError("No valid sequences found in file")
    
    return UploadedData(
        sequences=sequences,
        fileName=filename,
        totalLength=total_length,
        isDemo=False,
        uploadedAt=datetime.now()
    )

def parse_sequence_header(header: str, sequence: str) -> ParsedSequence:
    """Parse sequence header to extract ID and metadata"""
    if not header:
        raise FileValidationError("Empty sequence header found")
    
    if not sequence:
        raise FileValidationError("Empty sequence found")
    
    # Split header by common delimiters
    parts = header.split('|')
    seq_id = parts[0].strip()
    
    if not seq_id:
        seq_id = f"seq_{hash(header) % 10000}"
    
    description = header
    
    # Extract metadata from header parts
    metadata = {}
    if len(parts) > 1:
        for part in parts[1:]:
            part = part.strip()
            if ':' in part:
                key, value = part.split(':', 1)
                metadata[key.strip()] = value.strip()
            elif '=' in part:
                key, value = part.split('=', 1)
                metadata[key.strip()] = value.strip()
    
    # Try to extract species information
    species_name = extract_species_from_header(header)
    if species_name:
        metadata['species'] = species_name
    
    return ParsedSequence(
        id=seq_id,
        description=description,
        sequence=sequence,
        metadata=metadata
    )

def extract_species_from_header(header: str) -> str:
    """Extract species name from sequence header"""
    if not header:
        return "Unknown"
    
    # Remove '>' if present
    if header.startswith('>'):
        header = header[1:]
    
    # Common patterns for species names in headers
    # Pattern 1: "Genus species" at the beginning
    words = header.replace('_', ' ').split()
    
    if len(words) >= 2:
        # Check if first two words look like genus and species
        genus = words[0].strip()
        species = words[1].strip()
        
        # Basic validation - genus should be capitalized, species lowercase
        if genus.istitle() and species.islower():
            return f"{genus} {species}"
        elif len(genus) > 2 and len(species) > 2:
            # Less strict validation
            return f"{genus.title()} {species.lower()}"
    
    # Pattern 2: Look for common species name patterns
    header_lower = header.lower()
    common_genus = ['homo', 'mus', 'drosophila', 'escherichia', 'saccharomyces', 
                   'arabidopsis', 'caenorhabditis', 'danio', 'xenopus', 'rattus',
                   'tetrahymena', 'paramecium', 'plasmodium', 'trypanosoma']
    
    for genus in common_genus:
        if genus in header_lower:
            # Find the genus and try to get the species
            idx = header_lower.find(genus)
            remaining = header[idx:].split()
            if len(remaining) >= 2:
                return f"{remaining[0].title()} {remaining[1].lower()}"
    
    # Pattern 3: If no clear pattern, use first word as species
    if words:
        return words[0].title()
    
    return "Unknown"

def create_demo_fasta() -> str:
    """Create demo FASTA content with realistic species data"""
    return """>Tetrahymena thermophila strain SB210
ATGCGATCGATCGATCGTAGCTAGCTAGCTAGCATGCATGCATGCTAGCTAGCTAGCATCGATCG
ATCGATCGTAGCTAGCTAGCTAGCATGCATGCATGCTAGCTAGCTAGCATCGATCGATCGATCGT
AGCTAGCTAGCTAGCATGCATGCATGCTAGCTAGCTAGC
>Tetrahymena thermophila strain CU428
ATGCGATCGATCGATCGTAGCTAGCTAGCTAGCATGCATGCATGCTAGCTAGCTAACATCGATCG
ATCGATCGTAGCTAGCTAGCTAGCATGCATGCATGCTAGCTAGCTAACATCGATCGATCGATCGT
AGCTAGCTAGCTAGCATGCATGCATGCTAGCTAGCTAAC
>Tetrahymena pyriformis strain GL-C
ATGCGATCGATCGATCGTAGCTAGCTAGCTACCATGCATGCATGCTAGCTAGCTGGCATCGATCG
ATCGATCGTAGCTAGCTAGCTACCATGCATGCATGCTAGCTAGCTGGCATCGATCGATCGATCGT
AGCTAGCTAGCTACCATGCATGCATGCTAGCTAGCTGGC
>Tetrahymena pyriformis strain GL
ATGCGATCGATCGATCGTAGCTAGCTAGCTACCATGCATGCATGCTAGCTAGCTGGCATCGATCG
ATCGATCGTAGCTAGCTAGCTACCATGCATGCATGCTAGCTAGCTGGCATCGATCGATCGATCGT
AGCTAGCTAGCTACCATGCATGCATGCTAGCTAGCTGGC
>Paramecium caudatum isolate d4-2
ATGCGATCGATCGATCGTAGCTAGCTAGCTAGCATGCATGCATGCTAGCTAGCTAGCATCGATCG
ATCGATCGTAGCTAGCTAGCTAGCATGCATGCATGCTAGCTAGCTAGCATCGATCGATCGATCGT
AGCTAGCTAGCTAGCATGCATGCATGCTAGCTAGCTAGC
>Paramecium caudatum strain 51
ATGCGATCGATCGATCGTAGCTAGCTAGCTAGCATGCATGCATGCTAGCTAGCTAGCATCGATCG
ATCGATCGTAGCTAGCTAGCTAGCATGCATGCATGCTAGCTAGCTAGCATCGATCGATCGATCGT
AGCTAGCTAGCTAGCATGCATGCATGCTAGCTAGCTAGC
>Paramecium bursaria syngen 1
ATGCGATCGATCGATCGTAGCTAGCTAGCTACCATGCATGCATGCTAGCTAGCTGGCATCGATCG
ATCGATCGTAGCTAGCTAGCTACCATGCATGCATGCTAGCTAGCTGGCATCGATCGATCGATCGT
AGCTAGCTAGCTACCATGCATGCATGCTAGCTAGCTGGC
>Paramecium bursaria Pb1
ATGCGATCGATCGATCGTAGCTAGCTAGCTACCATGCATGCATGCTAGCTAGCTGGCATCGATCG
ATCGATCGTAGCTAGCTAGCTACCATGCATGCATGCTAGCTAGCTGGCATCGATCGATCGATCGT
AGCTAGCTAGCTACCATGCATGCATGCTAGCTAGCTGGC"""
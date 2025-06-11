from fastapi import UploadFile
from typing import List
from app.core.exceptions import FileValidationError, InsufficientDataError
from app.models.sequences import ParsedSequence

def validate_fasta_file(file: UploadFile) -> None:
    """Validate uploaded FASTA file"""
    if not file.filename:
        raise FileValidationError("No file provided")
    
    # Check file extension
    allowed_extensions = ('.fasta', '.fa', '.fas', '.fna', '.txt')
    if not file.filename.lower().endswith(allowed_extensions):
        raise FileValidationError(
            f"File must be in FASTA format. Allowed extensions: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (limit to 10MB)
    if hasattr(file, 'size') and file.size:
        max_size = 10 * 1024 * 1024  # 10MB
        if file.size > max_size:
            raise FileValidationError("File size must be less than 10MB")

def validate_file_content(content: str) -> None:
    """Validate FASTA file content"""
    if not content.strip():
        raise FileValidationError("File is empty")
    
    if not content.strip().startswith('>'):
        raise FileValidationError("Invalid FASTA format - file must start with '>'")
    
    # Count sequences
    sequence_count = content.count('>')
    if sequence_count == 0:
        raise FileValidationError("No sequences found in file")
    
    # Basic format validation
    lines = content.strip().split('\n')
    in_sequence = False
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('>'):
            # Header line
            if len(line) == 1:
                raise FileValidationError(f"Empty header found at line {i+1}")
            in_sequence = True
        else:
            # Sequence line
            if not in_sequence:
                raise FileValidationError(f"Sequence data without header at line {i+1}")
            
            # Check for valid nucleotide characters
            valid_chars = set('ATGCNatgcn-')
            invalid_chars = set(line) - valid_chars
            if invalid_chars:
                raise FileValidationError(
                    f"Invalid characters '{', '.join(invalid_chars)}' found at line {i+1}. "
                    f"Only A, T, G, C, N, and - are allowed."
                )

def validate_sequences_for_analysis(sequences: List[ParsedSequence], min_sequences: int = 2) -> None:
    """Validate sequences for analysis"""
    if len(sequences) < min_sequences:
        raise InsufficientDataError(f"At least {min_sequences} sequences required for analysis")
    
    # Check if all sequences are empty
    if all(len(seq.sequence) == 0 for seq in sequences):
        raise InsufficientDataError("All sequences are empty")
    
    # Check for minimum sequence length
    min_length = 10
    valid_sequences = [seq for seq in sequences if len(seq.sequence) >= min_length]
    if len(valid_sequences) < min_sequences:
        raise InsufficientDataError(
            f"At least {min_sequences} sequences with minimum length of {min_length} nucleotides required"
        )
    
    # Check for duplicate sequence IDs
    seq_ids = [seq.id for seq in sequences]
    if len(seq_ids) != len(set(seq_ids)):
        raise FileValidationError("Duplicate sequence IDs found")

def validate_analysis_type(analysis_type: str) -> None:
    """Validate analysis type"""
    valid_types = ['clustal', 'jukes', 'visualization', 'population']
    if analysis_type not in valid_types:
        raise FileValidationError(f"Invalid analysis type. Must be one of: {', '.join(valid_types)}")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    import re
    
    # Remove or replace unsafe characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Limit length
    if len(filename) > 100:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:95] + ('.' + ext if ext else '')
    
    return filename

def validate_sequence_format(sequence: str) -> bool:
    """Validate individual sequence format"""
    if not sequence:
        return False
    
    # Check for valid nucleotide characters only
    valid_chars = set('ATGCNatgcn-')
    return all(c in valid_chars for c in sequence)

def calculate_sequence_stats(sequences: List[ParsedSequence]) -> dict:
    """Calculate basic statistics for sequences"""
    if not sequences:
        return {}
    
    lengths = [len(seq.sequence) for seq in sequences]
    
    return {
        'count': len(sequences),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_length': sum(lengths) / len(lengths),
        'total_length': sum(lengths)
    }
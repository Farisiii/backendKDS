
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class ParsedSequence(BaseModel):
    id: str
    description: str
    sequence: str
    metadata: Optional[Dict[str, str]] = None

class UploadedData(BaseModel):
    sequences: List[ParsedSequence]
    fileName: str
    totalLength: int
    isDemo: bool
    uploadedAt: datetime
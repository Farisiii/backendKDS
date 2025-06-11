from enum import Enum
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class AnalysisType(str, Enum):
    clustal = "clustal"
    jukes = "jukes"
    visualization = "visualization"
    population = "population"

class AnalysisStatus(str, Enum):
    not_started = "not_started"
    running = "running"
    completed = "completed"
    error = "error"

class BaseAnalysisResult(BaseModel):
    analysisType: AnalysisType
    processingTime: float
    sequenceCount: int
    timestamp: datetime
from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from models.base import BaseAnalysisResult, AnalysisType, AnalysisStatus

class ClustalResult(BaseAnalysisResult):
    analysisType: AnalysisType = AnalysisType.clustal
    alignmentLength: int
    averageConservation: float
    gapPercentage: float
    qualityScore: float
    conservedPositions: int
    variablePositions: int
    identityMatrix: List[List[float]]

class JukesResult(BaseAnalysisResult):
    analysisType: AnalysisType = AnalysisType.jukes
    distanceMatrix: List[List[float]]
    sequenceIds: List[str]
    minDistance: float
    maxDistance: float
    averageDistance: float
    modelParameters: Dict[str, float]
    substitutionRates: Dict[str, float]

class VisualizationResult(BaseAnalysisResult):
    analysisType: AnalysisType = AnalysisType.visualization
    dendrogramUrl: str
    heatmapUrl: str
    pcaUrl: str
    pcaVarianceExplained: Dict[str, float]
    clusteringResults: Dict[str, Any]

class PopulationResult(BaseAnalysisResult):
    analysisType: AnalysisType = AnalysisType.population
    amongPopulationsVariance: float
    amongIndividualsVariance: float
    withinIndividualsVariance: float
    fst: float
    fis: float
    fit: float
    migrationRate: float
    nucleotideDiversity: float
    haplotypeDiversity: float
    tajimasD: float
    fusFs: float

class AnalysisResult(BaseModel):
    type: AnalysisType
    status: AnalysisStatus
    startedAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None
    error: Optional[str] = None
    data: Optional[Union[ClustalResult, JukesResult, VisualizationResult, PopulationResult]] = None
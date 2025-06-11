from fastapi import APIRouter, HTTPException, BackgroundTasks
from datetime import datetime
from app.models.base import AnalysisType, AnalysisStatus
from app.models.analysis import AnalysisResult
from app.services.analysis_service import (
    run_clustal_analysis, 
    run_jukes_analysis, 
    run_visualization_analysis, 
    run_population_analysis
)
from storage.memory_storage import get_storage

router = APIRouter()

@router.post("/{data_id}/{analysis_type}")
async def start_analysis(data_id: str, analysis_type: AnalysisType, background_tasks: BackgroundTasks):
    """Start analysis for uploaded data"""
    storage = get_storage()
    
    # Check if data exists
    data = storage.get_uploaded_data(data_id)
    if not data:
        raise HTTPException(status_code=404, detail="Data not found")
    
    # Validate data for analysis
    if len(data.sequences) < 2:
        raise HTTPException(status_code=400, detail="At least 2 sequences required for analysis")
    
    analysis_id = f"{data_id}_{analysis_type.value}"
    
    # Check if analysis is already running
    existing_analysis = storage.get_analysis(analysis_id)
    if existing_analysis and existing_analysis.status == AnalysisStatus.running:
        return {"analysisId": analysis_id, "status": "already_running"}
    
    # Set initial status
    initial_result = AnalysisResult(
        type=analysis_type,
        status=AnalysisStatus.running,
        startedAt=datetime.now()
    )
    storage.store_analysis(analysis_id, initial_result)
    
    # Start background analysis
    if analysis_type == AnalysisType.clustal:
        background_tasks.add_task(run_clustal_analysis, analysis_id, data_id)
    elif analysis_type == AnalysisType.jukes:
        background_tasks.add_task(run_jukes_analysis, analysis_id, data_id)
    elif analysis_type == AnalysisType.visualization:
        background_tasks.add_task(run_visualization_analysis, analysis_id, data_id)
    elif analysis_type == AnalysisType.population:
        background_tasks.add_task(run_population_analysis, analysis_id, data_id)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported analysis type: {analysis_type}")
    
    return {"analysisId": analysis_id, "status": "started"}

@router.get("/{analysis_id}")
async def get_analysis_result(analysis_id: str):
    """Get analysis result by ID"""
    storage = get_storage()
    
    result = storage.get_analysis(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return result

@router.get("/{data_id}/all")
async def get_all_analyses(data_id: str):
    """Get all analyses for a data ID"""
    storage = get_storage()
    
    # Check if data exists
    if not storage.get_uploaded_data(data_id):
        raise HTTPException(status_code=404, detail="Data not found")
    
    results = {}
    for analysis_type in AnalysisType:
        analysis_id = f"{data_id}_{analysis_type.value}"
        result = storage.get_analysis(analysis_id)
        
        if result:
            results[analysis_type.value] = result
        else:
            results[analysis_type.value] = AnalysisResult(
                type=analysis_type,
                status=AnalysisStatus.not_started
            )
    
    return results

@router.delete("/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete a specific analysis result"""
    storage = get_storage()
    
    if not storage.delete_analysis(analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return {"message": f"Analysis {analysis_id} deleted successfully"}

@router.get("/{data_id}/status")
async def get_analyses_status(data_id: str):
    """Get status of all analyses for a data ID"""
    storage = get_storage()
    
    # Check if data exists
    if not storage.get_uploaded_data(data_id):
        raise HTTPException(status_code=404, detail="Data not found")
    
    status_summary = {}
    for analysis_type in AnalysisType:
        analysis_id = f"{data_id}_{analysis_type.value}"
        result = storage.get_analysis(analysis_id)
        
        if result:
            status_summary[analysis_type.value] = result.status.value
        else:
            status_summary[analysis_type.value] = AnalysisStatus.not_started.value
    
    return status_summary
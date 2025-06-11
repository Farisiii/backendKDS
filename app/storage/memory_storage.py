from typing import Dict, List, Optional
from datetime import datetime
from models.analysis import AnalysisResult
from models.sequences import UploadedData

class MemoryStorage:
    """In-memory storage for analysis results and uploaded data"""
    
    def __init__(self):
        self._analysis_storage: Dict[str, AnalysisResult] = {}
        self._uploaded_data_storage: Dict[str, UploadedData] = {}
    
    # Analysis storage methods
    def store_analysis(self, analysis_id: str, result: AnalysisResult) -> None:
        """Store analysis result"""
        self._analysis_storage[analysis_id] = result
    
    def get_analysis(self, analysis_id: str) -> Optional[AnalysisResult]:
        """Get analysis result by ID"""
        return self._analysis_storage.get(analysis_id)
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis result"""
        if analysis_id in self._analysis_storage:
            del self._analysis_storage[analysis_id]
            return True
        return False
    
    def get_analyses_by_data_id(self, data_id: str) -> Dict[str, AnalysisResult]:
        """Get all analyses for a specific data ID"""
        results = {}
        for analysis_id, result in self._analysis_storage.items():
            if analysis_id.startswith(data_id):
                analysis_type = analysis_id.replace(f"{data_id}_", "")
                results[analysis_type] = result
        return results
    
    def delete_analyses_by_data_id(self, data_id: str) -> int:
        """Delete all analyses for a specific data ID"""
        keys_to_remove = [key for key in self._analysis_storage.keys() if key.startswith(data_id)]
        for key in keys_to_remove:
            del self._analysis_storage[key]
        return len(keys_to_remove)
    
    # Uploaded data storage methods
    def store_uploaded_data(self, data_id: str, data: UploadedData) -> None:
        """Store uploaded data"""
        self._uploaded_data_storage[data_id] = data
    
    def get_uploaded_data(self, data_id: str) -> Optional[UploadedData]:
        """Get uploaded data by ID"""
        return self._uploaded_data_storage.get(data_id)
    
    def delete_uploaded_data(self, data_id: str) -> bool:
        """Delete uploaded data"""
        if data_id in self._uploaded_data_storage:
            del self._uploaded_data_storage[data_id]
            return True
        return False
    
    def list_uploaded_data(self) -> List[str]:
        """List all uploaded data IDs"""
        return list(self._uploaded_data_storage.keys())
    
    # Storage statistics
    def get_storage_stats(self) -> Dict[str, int]:
        """Get storage statistics"""
        return {
            "uploaded_data_count": len(self._uploaded_data_storage),
            "analysis_count": len(self._analysis_storage)
        }
    
    def clear_all(self) -> None:
        """Clear all stored data"""
        self._analysis_storage.clear()
        self._uploaded_data_storage.clear()

# Global storage instance
storage = MemoryStorage()

def get_storage() -> MemoryStorage:
    """Get the global storage instance"""
    return storage
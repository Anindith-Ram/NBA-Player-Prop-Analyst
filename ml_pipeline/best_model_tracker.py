"""
BEST MODEL TRACKER
==================
Tracks and manages the best model per prop type based on calibrated accuracy.

This ensures we always use the best-performing model globally, not just the most recent one.
"""
import os
import json
from typing import Dict, Optional
from datetime import datetime

BEST_MODELS_METADATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'ml_models',
    'best_models_metadata.json'
)


class BestModelTracker:
    """Tracks the best model per prop type based on calibrated accuracy."""
    
    def __init__(self, metadata_path: str = None):
        """
        Initialize the best model tracker.
        
        Args:
            metadata_path: Path to JSON file storing best model metadata
        """
        if metadata_path is None:
            metadata_path = BEST_MODELS_METADATA_PATH
        
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load best model metadata from JSON file."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"   ⚠️ Error loading best models metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save best model metadata to JSON file."""
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_best_accuracy(self, prop_type: str) -> Optional[float]:
        """
        Get the best calibrated accuracy for a prop type.
        
        Args:
            prop_type: One of 'PTS', 'REB', 'AST', '3PM'
            
        Returns:
            Best calibrated accuracy, or None if no best model exists
        """
        prop_key = prop_type.upper()
        if prop_key in self.metadata:
            return self.metadata[prop_key].get('calibrated_accuracy')
        return None
    
    def is_better(self, prop_type: str, calibrated_accuracy: float) -> bool:
        """
        Check if a new model is better than the current best.
        
        Args:
            prop_type: One of 'PTS', 'REB', 'AST', '3PM'
            calibrated_accuracy: Calibrated accuracy of the new model
            
        Returns:
            True if new model is better, False otherwise
        """
        best_accuracy = self.get_best_accuracy(prop_type)
        
        if best_accuracy is None:
            # No existing best model, so this is better
            return True
        
        # New model is better if it has higher calibrated accuracy
        return calibrated_accuracy > best_accuracy
    
    def update_best_model(
        self,
        prop_type: str,
        calibrated_accuracy: float,
        model_path: str,
        calibrator_path: str,
        metrics: Dict,
        training_info: Dict = None
    ):
        """
        Update the best model for a prop type.
        
        Args:
            prop_type: One of 'PTS', 'REB', 'AST', '3PM'
            calibrated_accuracy: Calibrated accuracy of this model
            model_path: Path to the model file
            calibrator_path: Path to the calibrator file
            metrics: Dictionary of model metrics
            training_info: Optional training information (params, etc.)
        """
        prop_key = prop_type.upper()
        
        # Check if this is better than existing
        is_better = self.is_better(prop_type, calibrated_accuracy)
        
        if is_better:
            self.metadata[prop_key] = {
                'calibrated_accuracy': calibrated_accuracy,
                'model_path': model_path,
                'calibrator_path': calibrator_path,
                'metrics': metrics,
                'training_info': training_info or {},
                'updated_at': datetime.now().isoformat(),
                'is_current_best': True
            }
            self._save_metadata()
            return True
        else:
            # Not better, but store info about this training run
            if 'training_history' not in self.metadata.get(prop_key, {}):
                if prop_key not in self.metadata:
                    self.metadata[prop_key] = {}
                self.metadata[prop_key]['training_history'] = []
            
            self.metadata[prop_key]['training_history'].append({
                'calibrated_accuracy': calibrated_accuracy,
                'trained_at': datetime.now().isoformat(),
                'metrics': metrics
            })
            # Keep only last 10 training runs
            if len(self.metadata[prop_key]['training_history']) > 10:
                self.metadata[prop_key]['training_history'] = \
                    self.metadata[prop_key]['training_history'][-10:]
            
            self._save_metadata()
            return False
    
    def get_best_model_paths(self, prop_type: str) -> Optional[Dict[str, str]]:
        """
        Get paths to the best model and calibrator for a prop type.
        
        Args:
            prop_type: One of 'PTS', 'REB', 'AST', '3PM'
            
        Returns:
            Dictionary with 'model_path' and 'calibrator_path', or None
        """
        prop_key = prop_type.upper()
        if prop_key in self.metadata:
            return {
                'model_path': self.metadata[prop_key]['model_path'],
                'calibrator_path': self.metadata[prop_key]['calibrator_path']
            }
        return None
    
    def get_summary(self) -> Dict:
        """Get summary of all best models."""
        summary = {}
        for prop_type in ['PTS', 'REB', 'AST', '3PM']:
            if prop_type in self.metadata:
                summary[prop_type] = {
                    'calibrated_accuracy': self.metadata[prop_type].get('calibrated_accuracy'),
                    'updated_at': self.metadata[prop_type].get('updated_at'),
                    'model_path': self.metadata[prop_type].get('model_path')
                }
            else:
                summary[prop_type] = None
        return summary
    
    def print_summary(self):
        """Print a summary of all best models."""
        print("\n" + "="*60)
        print("BEST MODELS SUMMARY (by Calibrated Accuracy)")
        print("="*60)
        
        for prop_type in ['PTS', 'REB', 'AST', '3PM']:
            if prop_type in self.metadata:
                info = self.metadata[prop_type]
                acc = info.get('calibrated_accuracy', 0)
                updated = info.get('updated_at', 'Unknown')
                print(f"\n{prop_type}:")
                print(f"   Calibrated Accuracy: {acc:.1%}")
                print(f"   Updated: {updated}")
                print(f"   Model: {info.get('model_path', 'N/A')}")
            else:
                print(f"\n{prop_type}: No best model saved yet")
        
        print("="*60)


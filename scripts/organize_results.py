import os
import shutil
from pathlib import Path
import glob

def organize_results(source_dir):
    """
    Organize results by copying model checkpoints and specific json files to a structured format.
    
    Args:
        source_dir (str): Path to the source results directory
    """
    source_path = Path(source_dir)
    
    # Create new results structure
    results_path = source_path / 'results'
    models_dir = results_path / 'models'
    results_dir = results_path / 'results'
    
    # Create base directories
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define patterns for JSON files we want to copy
    json_patterns = [
        "*_metrics.json",
        "*_predictions.json",
        "prediction.json",
        "*_report.json",
        "args.json"
    ]
    
    # Get all task directories (immediate subdirectories of source_dir)
    task_dirs = [d for d in source_path.iterdir() if d.is_dir() and d.name != 'results']
    
    for task_dir in task_dirs:
        task_name = task_dir.name
        print(f"Processing task: {task_name}")
        
        # Create task-specific directories
        task_models_dir = models_dir / task_name
        task_results_dir = results_dir / task_name
        task_models_dir.mkdir(parents=True, exist_ok=True)
        task_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all experiment directories recursively
        experiment_dirs = [d for d in task_dir.rglob("*") if d.is_dir()]
        
        for exp_dir in experiment_dirs:
            # Copy best model checkpoints
            best_models = list(exp_dir.glob("*_best.pt"))
            for model in best_models:
                print(f"Copying model: {model.name}")
                shutil.copy2(model, task_models_dir / model.name)
            
            # Copy only specific json files
            for pattern in json_patterns:
                json_files = list(exp_dir.glob(pattern))
                for json_file in json_files:
                    print(f"Copying results: {json_file.name}")
                    shutil.copy2(json_file, task_results_dir / json_file.name)

if __name__ == "__main__":
    # Source directory containing the results
    source_dir = "/work/10608/aadharsh_aadhithya/vista/RepresentationLearning/results_20250725_111527/run_1"
    organize_results(source_dir)
    print("Results organization completed!") 
"""Model definitions, training pipelines, inference, and evaluation for CADICA."""

from src.models.yolo_trainer import train_yolo, load_yolo_model
from src.models.yolo_infer import predict_canonical_split
from src.models.eval_runner import run_pipeline_evaluation
from src.models.visualization import generate_error_grids, render_prediction_grid
from src.models.experiment_runner import (
    run_single_experiment,
    load_leaderboard,
    save_leaderboard_row,
    generate_cv_summary,
    ensure_yolo_dataset,
)

__all__ = [
    "train_yolo",
    "load_yolo_model",
    "predict_canonical_split",
    "run_pipeline_evaluation",
    "generate_error_grids",
    "render_prediction_grid",
    "run_single_experiment",
    "load_leaderboard",
    "save_leaderboard_row",
    "generate_cv_summary",
    "ensure_yolo_dataset",
]

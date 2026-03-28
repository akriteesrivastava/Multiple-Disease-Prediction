from __future__ import annotations

from model_utils import DISEASE_CONFIG, ensure_models_available, get_training_summary


def _format_metric_row(label: str, value: float) -> str:
	return f"{label:<12}: {value:.4f}"


def _print_confusion_matrix(matrix: list[list[int]]) -> None:
	print("Confusion Matrix")
	print("  Predicted Negative  Predicted Positive")
	print(f"Actual Negative {matrix[0][0]:>8} {matrix[0][1]:>19}")
	print(f"Actual Positive {matrix[1][0]:>8} {matrix[1][1]:>19}")


def _print_disease_summary(disease_name: str) -> None:
	summary = get_training_summary(disease_name)
	ensemble_metrics = summary["ensemble_metrics"]

	print(f"\n{'=' * 72}")
	print(disease_name)
	print(f"{'=' * 72}")
	print(f"Train/Test Split : {summary['split_ratio']}")
	print(f"Cross Validation : {summary['cross_validation']}")
	print("\nEnsemble Metrics")
	print(_format_metric_row("Accuracy", ensemble_metrics["accuracy"]))
	print(_format_metric_row("Precision", ensemble_metrics["precision"]))
	print(_format_metric_row("Recall", ensemble_metrics["recall"]))
	print(_format_metric_row("F1 Score", ensemble_metrics["f1"]))
	print(_format_metric_row("ROC AUC", ensemble_metrics["roc_auc"]))
	print()
	_print_confusion_matrix(summary["confusion_matrix"])

	print("\nModel Comparison")
	for result in summary["model_results"]:
		print(
			f"- {result['model']}: "
			f"cv_acc={result['cv_accuracy']:.4f}, "
			f"cv_f1={result['cv_f1']:.4f}, "
			f"cv_roc_auc={result['cv_roc_auc']:.4f}, "
			f"test_acc={result['test_accuracy']:.4f}, "
			f"test_f1={result['test_f1']:.4f}, "
			f"test_roc_auc={result['test_roc_auc']:.4f}"
		)

	print("\nTop Feature Importance")
	for feature_name, importance in summary["top_feature_importance"].items():
		print(f"- {feature_name}: {importance:.4f}")


def main() -> None:
	print("Preparing trained models and evaluation summary...")
	ensure_models_available()

	for disease_name in DISEASE_CONFIG:
		_print_disease_summary(disease_name)


if __name__ == "__main__":
	main()

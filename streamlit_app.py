from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

from model_utils import (
	DISEASE_CONFIG,
	ensure_models_available,
	get_effective_feature_frame,
	get_feature_summary,
	get_training_summary,
	predict_disease,
)


CATEGORICAL_OPTIONS = {
	"Sex": ["M", "F"],
	"ChestPainType": ["TA", "ATA", "NAP", "ASY"],
	"RestingECG": ["Normal", "ST", "LVH"],
	"ExerciseAngina": ["Y", "N"],
	"ST_Slope": ["Up", "Flat", "Down"],
}


def build_input_frame(feature_defaults: Dict[str, Any]) -> pd.DataFrame:
	user_values = {}

	for feature_name, default_value in feature_defaults.items():
		label = feature_name.replace("_", " ").title()

		if feature_name in CATEGORICAL_OPTIONS:
			options = CATEGORICAL_OPTIONS[feature_name]
			default_index = options.index(default_value) if default_value in options else 0
			user_values[feature_name] = st.selectbox(label, options, index=default_index)
		else:
			user_values[feature_name] = st.number_input(
				label,
				value=float(default_value),
				step=0.1,
				format="%.2f",
			)

	return pd.DataFrame([user_values])


def render_sidebar() -> str:
	st.sidebar.title("Prediction Options")
	st.sidebar.caption("Choose a disease model and enter patient feature values.")
	return st.sidebar.selectbox("Disease Model", list(DISEASE_CONFIG.keys()))


def main() -> None:
	st.set_page_config(page_title="Multiple Disease Prediction", page_icon="+", layout="wide")
	st.title("ML Based Multiple Disease Prediction")
	st.write(
		"This educational app trains machine learning models from the CSV files in your Dataset folder and uses them "
		"to estimate disease risk. It is intended for learning and demonstration, not medical diagnosis."
	)

	with st.spinner("Preparing trained models..."):
		ensure_models_available()

	selected_disease = render_sidebar()
	disease_details = DISEASE_CONFIG[selected_disease]
	training_summary = get_training_summary(selected_disease)
	feature_summary = get_feature_summary(selected_disease)

	left_column, right_column = st.columns([1.3, 1])

	with left_column:
		st.subheader(f"{selected_disease} Inputs")
		st.caption(disease_details["description"])
		input_frame = build_input_frame(disease_details["input_defaults"])
		effective_input_frame = get_effective_feature_frame(selected_disease, input_frame)

		if st.button(f"Predict {selected_disease}", type="primary"):
			prediction_label, probability = predict_disease(selected_disease, input_frame)

			if probability >= 0.5:
				st.error(
					f"Prediction: {prediction_label} | Estimated probability: {probability:.1%}"
				)
			else:
				st.success(
					f"Prediction: {prediction_label} | Estimated probability: {probability:.1%}"
				)

	with right_column:
		st.subheader("Model Information")
		st.write(f"Model type: {disease_details['model_name']}")
		st.write(f"Training source: {disease_details['dataset_note']}")
		st.write(f"Raw input features: {feature_summary['raw_feature_count']}")
		st.write(f"Effective model features: {feature_summary['effective_feature_count']}")
		st.write(f"Train/Test split: {training_summary['split_ratio']}")
		st.write(f"Cross-validation: {training_summary['cross_validation']}")
		st.write(f"Ensemble test accuracy: {training_summary['ensemble_metrics']['accuracy']:.2%}")
		st.write(f"Ensemble ROC-AUC: {training_summary['ensemble_metrics']['roc_auc']:.2%}")
		st.write(f"Ensemble recall: {training_summary['ensemble_metrics']['recall']:.2%}")
		st.subheader("Raw Input Frame")
		st.dataframe(input_frame, use_container_width=True)
		st.subheader("Effective Model Feature Frame")
		st.dataframe(effective_input_frame, use_container_width=True)

		st.subheader("Model Comparison")
		st.dataframe(pd.DataFrame(training_summary["model_results"]), use_container_width=True)

		st.subheader("Confusion Matrix")
		confusion_matrix = training_summary["confusion_matrix"]
		st.dataframe(
			pd.DataFrame(
				confusion_matrix,
				index=["Actual Negative", "Actual Positive"],
				columns=["Predicted Negative", "Predicted Positive"],
			),
			use_container_width=True,
		)

		st.subheader("Top Feature Importance")
		st.dataframe(
			pd.DataFrame(
				[
					{"Feature": feature_name, "Importance": value}
					for feature_name, value in training_summary["top_feature_importance"].items()
				]
			),
			use_container_width=True,
		)

		st.info(
			"Results are generated from your project datasets. Use domain review and stronger validation "
			"before treating this as a production healthcare application."
		)


if __name__ == "__main__":
	main()
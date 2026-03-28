from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
MODELS_DIR = BASE_DIR / "models"
TRAIN_TEST_SPLIT_RATIO = 0.30
CV_FOLDS = 5
SCORING = {
	"accuracy": "accuracy",
	"precision": "precision",
	"recall": "recall",
	"f1": "f1",
	"roc_auc": "roc_auc",
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
	return numerator.divide(denominator.replace(0, np.nan))


def _add_diabetes_engineered_features(dataset: pd.DataFrame) -> pd.DataFrame:
	feature_frame = dataset.copy()

	feature_frame["Glucose_BMI_Interaction"] = feature_frame["Glucose"] * feature_frame["BMI"]
	feature_frame["Age_BMI_Interaction"] = feature_frame["Age"] * feature_frame["BMI"]
	feature_frame["Pregnancies_Age_Ratio"] = _safe_ratio(
		feature_frame["Pregnancies"],
		feature_frame["Age"],
	)
	feature_frame["Insulin_Glucose_Ratio"] = _safe_ratio(
		feature_frame["Insulin"],
		feature_frame["Glucose"],
	)
	feature_frame["SkinThickness_BMI_Ratio"] = _safe_ratio(
		feature_frame["SkinThickness"],
		feature_frame["BMI"],
	)
	feature_frame["BloodPressure_Age_Ratio"] = _safe_ratio(
		feature_frame["BloodPressure"],
		feature_frame["Age"],
	)

	feature_frame["Has_High_Glucose"] = (feature_frame["Glucose"] >= 140).astype(str)
	feature_frame["Has_Obesity"] = (feature_frame["BMI"] >= 30).astype(str)
	feature_frame["Has_Hypertension"] = (feature_frame["BloodPressure"] >= 90).astype(str)
	feature_frame["Has_Diabetes_Pedigree_Risk"] = (
		feature_frame["DiabetesPedigreeFunction"] >= 0.5
	).astype(str)

	feature_frame["BMI_Class"] = pd.cut(
		feature_frame["BMI"],
		bins=[-np.inf, 18.5, 25.0, 30.0, np.inf],
		labels=["Underweight", "Normal", "Overweight", "Obese"],
	)
	feature_frame["Age_Group"] = pd.cut(
		feature_frame["Age"],
		bins=[-np.inf, 30, 45, 60, np.inf],
		labels=["Young", "Adult", "Middle", "Senior"],
	)
	feature_frame["Glucose_Group"] = pd.cut(
		feature_frame["Glucose"],
		bins=[-np.inf, 100, 126, 160, np.inf],
		labels=["Normal", "Elevated", "Prediabetic", "High"],
	)

	return feature_frame


def prepare_feature_frame_for_disease(disease_name: str, feature_frame: pd.DataFrame) -> pd.DataFrame:
	prepared_frame = feature_frame.copy()

	if disease_name == "Female Diabetes Risk":
		invalid_zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
		available_columns = [column for column in invalid_zero_columns if column in prepared_frame.columns]
		prepared_frame[available_columns] = prepared_frame[available_columns].replace(0, float("nan"))
		prepared_frame = _add_diabetes_engineered_features(prepared_frame)

	return prepared_frame


def _diabetes_dataset() -> Tuple[pd.DataFrame, pd.Series]:
	dataset = pd.read_csv(DATASET_DIR / "diabetes.csv")
	features = prepare_feature_frame_for_disease("Diabetes Risk", dataset.drop(columns=["Outcome"]))
	target = dataset["Outcome"]
	return features, target


def _breast_cancer_dataset() -> Tuple[pd.DataFrame, pd.Series]:
	dataset = pd.read_csv(DATASET_DIR / "breast_cancer_wisconsin.csv")
	dataset = dataset.drop(columns=["Unnamed: 32"], errors="ignore")
	target = dataset["diagnosis"].map({"B": 0, "M": 1})
	features = dataset.drop(columns=["diagnosis", "id"], errors="ignore")
	return features, target


def _heart_disease_dataset() -> Tuple[pd.DataFrame, pd.Series]:
	dataset = pd.read_csv(DATASET_DIR / "heart.csv")
	features = dataset.drop(columns=["HeartDisease"])
	target = dataset["HeartDisease"]
	return features, target


def _diabetes_default_inputs() -> Dict[str, Any]:
	return {
		"Pregnancies": 2.0,
		"Glucose": 120.0,
		"BloodPressure": 70.0,
		"SkinThickness": 20.0,
		"Insulin": 80.0,
		"BMI": 28.5,
		"DiabetesPedigreeFunction": 0.47,
		"Age": 33.0,
	}


def _breast_cancer_default_inputs() -> Dict[str, Any]:
	return {
		"radius_mean": 14.0,
		"texture_mean": 19.0,
		"perimeter_mean": 92.0,
		"area_mean": 650.0,
		"smoothness_mean": 0.10,
		"compactness_mean": 0.10,
		"concavity_mean": 0.09,
		"concave points_mean": 0.05,
		"symmetry_mean": 0.18,
		"fractal_dimension_mean": 0.06,
		"radius_se": 0.40,
		"texture_se": 1.20,
		"perimeter_se": 2.80,
		"area_se": 40.0,
		"smoothness_se": 0.007,
		"compactness_se": 0.02,
		"concavity_se": 0.03,
		"concave points_se": 0.01,
		"symmetry_se": 0.02,
		"fractal_dimension_se": 0.003,
		"radius_worst": 16.0,
		"texture_worst": 25.0,
		"perimeter_worst": 107.0,
		"area_worst": 880.0,
		"smoothness_worst": 0.14,
		"compactness_worst": 0.25,
		"concavity_worst": 0.27,
		"concave points_worst": 0.11,
		"symmetry_worst": 0.29,
		"fractal_dimension_worst": 0.08,
	}


def _heart_default_inputs() -> Dict[str, Any]:
	return {
		"Age": 52.0,
		"Sex": "M",
		"ChestPainType": "ATA",
		"RestingBP": 130.0,
		"Cholesterol": 220.0,
		"FastingBS": 0.0,
		"RestingECG": "Normal",
		"MaxHR": 150.0,
		"ExerciseAngina": "N",
		"Oldpeak": 1.0,
		"ST_Slope": "Flat",
	}


def _build_preprocessor(features: pd.DataFrame, disease_name: str) -> ColumnTransformer:
	categorical_columns = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
	numeric_columns = [column for column in features.columns if column not in categorical_columns]
	numeric_imputer_strategy = "mean" if disease_name == "Female Diabetes Risk" else "median"

	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy=numeric_imputer_strategy)),
			("scaler", StandardScaler()),
		]
	)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
		]
	)

	transformers = []
	if numeric_columns:
		transformers.append(("numeric", numeric_pipeline, numeric_columns))
	if categorical_columns:
		transformers.append(("categorical", categorical_pipeline, categorical_columns))

	return ColumnTransformer(transformers=transformers)


def _build_base_estimators() -> Dict[str, Any]:
	return {
		"Random Forest": RandomForestClassifier(
			n_estimators=300,
			max_depth=10,
			min_samples_leaf=2,
			class_weight="balanced",
			random_state=42,
			n_jobs=-1,
		),
		"SVM": CalibratedClassifierCV(
			estimator=SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", random_state=42),
			method="sigmoid",
			cv=3,
		),
		"KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
	}


def _build_model_pipeline(features: pd.DataFrame, estimator: Any) -> Pipeline:
	raise NotImplementedError


def _build_model_pipeline_for_disease(disease_name: str, features: pd.DataFrame, estimator: Any) -> Pipeline:
	return Pipeline(steps=[("preprocessor", _build_preprocessor(features, disease_name)), ("model", estimator)])


def _build_stacking_pipeline(disease_name: str, features: pd.DataFrame) -> Pipeline:
	base_estimators = [(name, estimator) for name, estimator in _build_base_estimators().items()]
	stacking_model = StackingClassifier(
		estimators=base_estimators,
		final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
		stack_method="predict_proba",
		cv=CV_FOLDS,
	)
	return _build_model_pipeline_for_disease(disease_name, features, stacking_model)


def _collect_metrics(y_true: pd.Series, y_probabilities: pd.Series) -> Dict[str, Any]:
	predictions = (y_probabilities >= 0.5).astype(int)
	return {
		"accuracy": round(float(accuracy_score(y_true, predictions)), 4),
		"precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
		"recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
		"f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
		"roc_auc": round(float(roc_auc_score(y_true, y_probabilities)), 4),
		"confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
	}


def _summarize_cv_results(model_name: str, cv_results: Dict[str, Any], test_metrics: Dict[str, Any]) -> Dict[str, Any]:
	return {
		"model": model_name,
		"cv_accuracy": round(float(cv_results["test_accuracy"].mean()), 4),
		"cv_precision": round(float(cv_results["test_precision"].mean()), 4),
		"cv_recall": round(float(cv_results["test_recall"].mean()), 4),
		"cv_f1": round(float(cv_results["test_f1"].mean()), 4),
		"cv_roc_auc": round(float(cv_results["test_roc_auc"].mean()), 4),
		"test_accuracy": test_metrics["accuracy"],
		"test_precision": test_metrics["precision"],
		"test_recall": test_metrics["recall"],
		"test_f1": test_metrics["f1"],
		"test_roc_auc": test_metrics["roc_auc"],
	}


def _get_feature_importance(model_pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
	importance = permutation_importance(
		model_pipeline,
		x_test,
		y_test,
		n_repeats=8,
		random_state=42,
		scoring="roc_auc",
	)
	summary = pd.Series(importance.importances_mean, index=x_test.columns).sort_values(ascending=False)
	return {name: round(float(value), 4) for name, value in summary.head(10).items()}


def _train_disease_bundle(disease_name: str, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
	x_train, x_test, y_train, y_test = train_test_split(
		features,
		target,
		test_size=TRAIN_TEST_SPLIT_RATIO,
		random_state=42,
		stratify=target,
	)

	cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
	model_results = []

	for model_name, estimator in _build_base_estimators().items():
		pipeline = _build_model_pipeline_for_disease(disease_name, features, estimator)
		cv_results = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=SCORING)
		pipeline.fit(x_train, y_train)
		test_probabilities = pipeline.predict_proba(x_test)[:, 1]
		test_metrics = _collect_metrics(y_test, test_probabilities)
		model_results.append(_summarize_cv_results(model_name, cv_results, test_metrics))

	stacking_pipeline = _build_stacking_pipeline(disease_name, features)
	stacking_cv_results = cross_validate(stacking_pipeline, x_train, y_train, cv=cv, scoring=SCORING)
	stacking_pipeline.fit(x_train, y_train)
	ensemble_probabilities = stacking_pipeline.predict_proba(x_test)[:, 1]
	ensemble_metrics = _collect_metrics(y_test, ensemble_probabilities)
	model_results.append(_summarize_cv_results("Stacked Ensemble", stacking_cv_results, ensemble_metrics))

	return {
		"model": stacking_pipeline,
		"training_summary": {
			"split_ratio": "70:30",
			"cross_validation": f"Stratified {CV_FOLDS}-fold",
			"model_results": model_results,
			"ensemble_metrics": {
				"accuracy": ensemble_metrics["accuracy"],
				"precision": ensemble_metrics["precision"],
				"recall": ensemble_metrics["recall"],
				"f1": ensemble_metrics["f1"],
				"roc_auc": ensemble_metrics["roc_auc"],
			},
			"confusion_matrix": ensemble_metrics["confusion_matrix"],
			"top_feature_importance": _get_feature_importance(stacking_pipeline, x_test, y_test),
		},
	}


DISEASE_CONFIG: Dict[str, Dict[str, Any]] = {
	"Female Diabetes Risk": {
		"builder": _diabetes_dataset,
		"model_file": MODELS_DIR / "female_diabetes_risk_model_v6.joblib",
		"description": "Predicts diabetes risk for the female-focused diabetes dataset using all raw parameters plus engineered interaction, ratio, and risk-bucket features.",
		"dataset_note": "Kaggle Pima Indians Diabetes Database CSV with mean imputation for zero-as-missing clinical values.",
		"model_name": "Stacked ensemble of Random Forest, calibrated SVM, and KNN",
		"target_labels": {0: "Lower diabetes risk", 1: "Higher diabetes risk"},
		"input_defaults": _diabetes_default_inputs(),
	},
	"Breast Cancer": {
		"builder": _breast_cancer_dataset,
		"model_file": MODELS_DIR / "breast_cancer_model_v4.joblib",
		"description": "Classifies breast cancer records using the breast_cancer_wisconsin.csv file from your Dataset folder.",
		"dataset_note": "Hugging Face scikit-learn breast-cancer-wisconsin CSV.",
		"model_name": "Stacked ensemble of Random Forest, calibrated SVM, and KNN",
		"target_labels": {0: "Benign pattern detected", 1: "Malignant pattern detected"},
		"input_defaults": _breast_cancer_default_inputs(),
	},
	"Heart Disease": {
		"builder": _heart_disease_dataset,
		"model_file": MODELS_DIR / "heart_disease_model_v4.joblib",
		"description": "Estimates heart disease risk using the heart.csv file from your Dataset folder.",
		"dataset_note": "Kaggle heart.csv with categorical and numeric clinical features.",
		"model_name": "Stacked ensemble of Random Forest, calibrated SVM, and KNN",
		"target_labels": {0: "Lower heart disease risk", 1: "Higher heart disease risk"},
		"input_defaults": _heart_default_inputs(),
	},
}


def train_and_save_models() -> None:
	MODELS_DIR.mkdir(exist_ok=True)

	for disease_name, config in DISEASE_CONFIG.items():
		features, target = config["builder"]()
		bundle = _train_disease_bundle(disease_name, features, target)
		joblib.dump(bundle, config["model_file"])


def ensure_models_available() -> None:
	required_files = ["diabetes.csv", "heart.csv", "breast_cancer_wisconsin.csv"]
	missing_datasets = [name for name in required_files if not (DATASET_DIR / name).exists()]
	if missing_datasets:
		missing_list = ", ".join(missing_datasets)
		raise FileNotFoundError(f"Missing dataset files in {DATASET_DIR}: {missing_list}")

	missing_model = any(not config["model_file"].exists() for config in DISEASE_CONFIG.values())
	if missing_model:
		train_and_save_models()


def get_effective_feature_frame(disease_name: str, input_frame: pd.DataFrame) -> pd.DataFrame:
	return prepare_feature_frame_for_disease(disease_name, input_frame)


def predict_disease(disease_name: str, input_frame: pd.DataFrame) -> Tuple[str, float]:
	config = DISEASE_CONFIG[disease_name]
	bundle = joblib.load(config["model_file"])
	prepared_input = get_effective_feature_frame(disease_name, input_frame)
	probability = float(bundle["model"].predict_proba(prepared_input)[0][1])
	prediction = int(probability >= 0.5)
	label = config["target_labels"][prediction]
	return label, probability


def get_training_summary(disease_name: str) -> Dict[str, Any]:
	config = DISEASE_CONFIG[disease_name]
	bundle = joblib.load(config["model_file"])
	return bundle["training_summary"]


def get_feature_summary(disease_name: str) -> Dict[str, Any]:
	raw_features = list(DISEASE_CONFIG[disease_name]["input_defaults"].keys())
	effective_features = list(get_effective_feature_frame(disease_name, pd.DataFrame([DISEASE_CONFIG[disease_name]["input_defaults"]])).columns)
	return {
		"raw_feature_count": len(raw_features),
		"effective_feature_count": len(effective_features),
		"raw_features": raw_features,
		"effective_features": effective_features,
	}
# ...existing code...
import json
from pathlib import Path

import joblib
import pandas as pd

from explainability_agent import ExplainabilityAgent

MODEL_ID = "trainer_random_forest"
ARTIFACT_ROOT = Path("artifacts")
SAMPLE_SIZE = 200  # quick verification size for local explanations
BACKGROUND_SIZE = 100  # SHAP background subset to keep runtime manageable
GLOBAL_SAMPLE_SIZE = 512  # cap for global SHAP computation

model_path = ARTIFACT_ROOT / "modeling" / "models" / f"{MODEL_ID}.joblib"
metadata_path = ARTIFACT_ROOT / "modeling" / "models" / f"{MODEL_ID}_metadata.json"
prob_path = ARTIFACT_ROOT / "modeling" / "models" / f"{MODEL_ID}_probabilities.csv"
feature_path = ARTIFACT_ROOT / "predict_online_gaming_enhanced.csv"


def main() -> None:
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())

    feature_frame = pd.read_csv(feature_path).set_index("PlayerID")
    feature_frame = feature_frame[metadata["feature_names"]]

    prob_df = pd.read_csv(prob_path).set_index("PlayerID")
    predictions = prob_df["predicted_label"]

    probability_columns = {
        f"prob_{label}": label for label in metadata["class_labels"]
    }
    aligned_probabilities = prob_df.rename(columns=probability_columns)

    agent = ExplainabilityAgent(llm_enabled=False)

    background_frame = feature_frame
    if len(background_frame) > BACKGROUND_SIZE:
        background_frame = background_frame.sample(BACKGROUND_SIZE, random_state=42)

    agent.register_model(
        model=model,
        feature_names=metadata["feature_names"],
        class_labels=metadata["class_labels"],
        background_frame=background_frame,
        background_sample_size=BACKGROUND_SIZE,
    )

    global_summary = agent.compute_global_importance(
        feature_frame,
        sample_size=min(GLOBAL_SAMPLE_SIZE, len(feature_frame)),
    )

    inference_index = predictions.index
    if len(inference_index) > SAMPLE_SIZE:
        inference_index = predictions.sample(SAMPLE_SIZE, random_state=42).index

    local_df = agent.explain_batch(
        feature_frame=feature_frame.loc[inference_index],
        predictions=predictions.loc[inference_index],
        probability_frame=aligned_probabilities.loc[inference_index],
        batch_id=MODEL_ID,
    )
    guardrail = agent.build_guardrail_summary(local_df)

    print("Global summary saved to:", (agent.artifact_dir / "global_importance.json").resolve())
    print(
        "Local explanations saved to:",
        (agent.artifact_dir / f"local_explanations_{MODEL_ID}.jsonl").resolve(),
    )
    print("Guardrail summary:", guardrail)


if __name__ == "__main__":
    main()
# ...existing code...
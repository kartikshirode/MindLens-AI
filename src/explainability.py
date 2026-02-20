import numpy as np
import shap
from lime.lime_text import LimeTextExplainer

# Vocabulary of words and word-stems relevant to mental health.
# We use this to check whether the model's influential features
# actually correspond to clinically meaningful language.
mental_health_vocabulary = {
    "sad", "hopeless", "alone", "suicide", "tired", "worthless",
    "depressed", "anxious", "empty", "pain", "die", "help",
    "crying", "lost", "hate", "suffer", "miserable", "numb",
    "angry", "scared", "hurt", "broken", "overwhelmed", "desperate",
    "lonely", "useless", "failure", "guilt", "ashamed", "isolat",
    "depression", "anxiety", "stress", "trauma", "therapy", "therapist",
    "medication", "mental", "health", "disorder", "diagnosis", "symptom",
    "insomnia", "sleep", "fatigue", "exhausted", "restless",
    "suicidal", "ideation", "self", "harm", "cutting", "overdose",
    "panic", "attack", "phobia", "ptsd", "bipolar", "mania",
    "mood", "emotion", "feeling", "feel", "felt",
    "cry", "cried", "tear", "sobbing",
    "worry", "worrie", "fear", "dread", "terrif",
    "isol", "withdraw", "detach", "disconnect",
    "reject", "abandon", "neglect", "abuse",
    "burden", "exhaust", "drain", "burn",
    "disappoint", "regret", "shame", "embarrass",
    "confus", "frustrat", "irritat", "agitat",
    "helpless", "powerless", "trapped", "stuck",
    "meaningless", "pointless", "purpose",
    "death", "dead", "kill", "end", "life", "live", "alive",
    "drug", "alcohol", "addict", "substance",
    "counselor", "psychiatr", "psycholog", "doctor",
    "diagnos", "treat", "recover", "cope", "coping",
    "support", "talk", "listen", "understand",
    "friend", "family", "relationship", "partner",
    "work", "job", "school", "college",
    "money", "financial", "debt",
    "weight", "eat", "food", "appetite", "binge", "purg",
    "think", "thought", "mind", "brain", "head",
    "night", "day", "morning", "wake", "bed",
    "want", "need", "wish", "hope", "try",
    "can", "anymore", "nothing", "everything", "always", "never",
    "know", "better", "worse", "bad", "good", "normal",
    "people", "person", "someone", "anyone", "nobody", "everybody",
}


def explain_with_lime(classifier, vectorizer, text, num_features=10, class_names=None):
    """Build a LIME explanation for one text sample."""
    if class_names is None:
        class_names = ["No Risk", "Risk"]

    explainer = LimeTextExplainer(class_names=class_names, random_state=42)

    def _predict(texts):
        return classifier.predict_proba(vectorizer.transform(texts))

    return explainer.explain_instance(text, _predict, num_features=num_features, num_samples=2000)


def explain_with_shap(classifier, x_train, x_test, feature_names):
    """Compute SHAP values via LinearExplainer."""
    explainer = shap.LinearExplainer(classifier, x_train, feature_perturbation="interventional")
    values = explainer.shap_values(x_test)
    return values, explainer


def shap_summary(shap_values, x_test, feature_names, max_display=20):
    """Render the SHAP beeswarm summary."""
    shap.summary_plot(shap_values, x_test, feature_names=feature_names,
                      max_display=max_display, show=True)


def shap_single_force_plot(explainer, shap_values, sample_index, feature_names):
    """Render a SHAP force plot for one observation."""
    return shap.force_plot(explainer.expected_value, shap_values[sample_index],
                           feature_names=feature_names, matplotlib=True)


def interpretability_score(shap_values, feature_names, vocabulary=None, k=10):
    """
    For every sample, count how many of the top-k SHAP features appear
    in the mental-health vocabulary. Score = overlap / k.
    """
    if vocabulary is None:
        vocabulary = mental_health_vocabulary

    feature_names = np.array(feature_names)
    per_sample = []

    for row in range(shap_values.shape[0]):
        top_k_idx = np.argsort(np.abs(shap_values[row]))[-k:]
        top_words = set(feature_names[top_k_idx])
        overlap = sum(1 for w in top_words if any(term in w for term in vocabulary))
        per_sample.append(overlap / k)

    scores = np.array(per_sample)
    return {
        "per_sample_scores": scores,
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
    }

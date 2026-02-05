def classify_with_rules(features):
    """
    Rule-based voice classification:
    Returns classification, confidence, explanation
    """

    pitch_std = features["pitch_std"]
    mfcc_std = features["mfcc_std"]
    energy = features["energy"]

    score = 0
    explanations = []

    # Rule 1: Pitch variation
    if pitch_std < 15:
        score += 1
        explanations.append("Pitch variation is unnaturally low")

    # Rule 2: MFCC smoothness
    if mfcc_std < 20:
        score += 1
        explanations.append("Voice texture is too smooth")

    # Rule 3: Energy consistency
    if energy < 0.02:
        score += 1
        explanations.append("Audio energy is unusually consistent")

    # Decision logic
    if score >= 2:
        classification = "AI-generated"
        confidence = 70 + (score * 10)
    else:
        classification = "Human"
        confidence = 60 + ((3 - score) * 10)

    explanation = "; ".join(explanations) if explanations else "Natural voice patterns detected"

    return classification, confidence, explanation

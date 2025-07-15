import pandas as pd

def predict_with_confidence(model, vectorizer, text):
    vect = vectorizer.transform([text])
    pred = model.predict(vect)[0]
    proba = model.predict_proba(vect).max()
    return pred, proba

def batch_predict(model, vectorizer, texts):
    vect = vectorizer.transform(texts)
    preds = model.predict(vect)
    probas = model.predict_proba(vect).max(axis=1)
    return pd.DataFrame({
        "Resume": texts,
        "Predicted Category": preds,
        "Confidence": probas
    })
"""Trains SMS Shield, compares two models and saves the better one.

    python train.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

import corpus

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

DEMO = ["can you send me the ai notes before 6 pm",
        "congratulations you have won a free laptop claim now at http://offer4u.xyz/go",
        "your exam fee of rs 5000 is due on friday pay at the office counter",
        "sir your kyc is pending kindly share otp to continue"]


def make_models():
    """Both models share the same TF-IDF front end: word uni- and bi-grams
    that appear in at least two messages."""
    return {
        "Naive Bayes": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2), MultinomialNB()),
        "Logistic Regression": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2),
            LogisticRegression(max_iter=1000)),
    }


def main():
    rows = corpus.build()
    X = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    print("corpus: %d  spam: %d  genuine: %d" % (len(X), y.sum(), (1 - y).sum()))

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)
    models = make_models()
    results = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]
        cv = cross_val_score(model, X, y, cv=5, scoring="f1")
        fpr, tpr, _ = roc_curve(yte, proba)
        results[name] = dict(
            accuracy=round(accuracy_score(yte, pred), 4),
            precision=round(precision_score(yte, pred), 4),
            recall=round(recall_score(yte, pred), 4),
            f1=round(f1_score(yte, pred), 4),
            cv_f1_mean=round(cv.mean(), 4),
            auc=round(auc(fpr, tpr), 4),
            confusion=confusion_matrix(yte, pred).tolist())
        results[name]["_roc"] = (fpr.tolist(), tpr.tolist())
        print("\n== %s ==" % name)
        print(classification_report(yte, pred, target_names=["genuine", "spam"],
                                    digits=4))

    best_name = max(results, key=lambda k: results[k]["f1"])
    best = models[best_name]
    print("best model:", best_name)

    # --- ROC curve -------------------------------------------------------
    plt.figure(figsize=(5.2, 4))
    for name in results:
        fpr, tpr = results[name]["_roc"]
        plt.plot(fpr, tpr, lw=2, label="%s (AUC %.3f)" % (name, results[name]["auc"]))
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve - SMS Shield")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "roc.png"), dpi=160)
    plt.close()

    # --- confusion matrix ------------------------------------------------
    cm = np.array(results[best_name]["confusion"])
    plt.figure(figsize=(4.2, 3.8))
    plt.imshow(cm, cmap="Purples")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["genuine", "spam"])
    plt.yticks([0, 1], ["genuine", "spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix - " + best_name)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "confusion.png"), dpi=160)
    plt.close()

    # --- which words push a message towards spam -------------------------
    lr = models["Logistic Regression"]
    vec = lr.named_steps["tfidfvectorizer"]
    clf = lr.named_steps["logisticregression"]
    names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    top = np.argsort(coefs)[::-1][:12]
    plt.figure(figsize=(5.6, 4))
    plt.barh([names[i] for i in top][::-1], coefs[top][::-1], color="#7b4bbd")
    plt.xlabel("Weight towards spam")
    plt.title("Words that signal spam")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "words.png"), dpi=160)
    plt.close()

    joblib.dump(best, os.path.join(HERE, "model.joblib"))
    for r in results.values():
        r.pop("_roc", None)
    json.dump({"best": best_name, "results": results,
               "top_spam_words": [names[i] for i in top[:8]]},
              open(os.path.join(HERE, "metrics.json"), "w"), indent=2)

    print("\nDEMO")
    for message in DEMO:
        p = best.predict_proba([message])[0][1]
        print("%-70s %-6s %5.1f%%" % (message[:70], "SPAM" if p >= .5 else "ham",
                                      p * 100))


if __name__ == "__main__":
    main()

"""Scores messages from the command line.

    python predict.py "your kyc is pending kindly share otp"
    python predict.py            # scores a short built-in sample
"""
import os
import sys

import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.joblib")

SAMPLE = [
    "can you send me the ai notes before 6 pm",
    "congratulations you have won a free laptop claim now at http://offer4u.xyz/go",
    "your exam fee of rs 5000 is due on friday pay at the office counter",
    "sir your kyc is pending kindly share otp to continue",
]


def load():
    if not os.path.exists(MODEL):
        sys.exit("model.joblib not found - run 'python train.py' first")
    return joblib.load(MODEL)


def score(model, message):
    return round(float(model.predict_proba([message])[0][1]) * 100, 1)


def main():
    model = load()
    messages = sys.argv[1:] or SAMPLE
    for message in messages:
        risk = score(model, message)
        verdict = "SPAM" if risk >= 50 else "genuine"
        print("%-8s %5.1f%%  %s" % (verdict, risk, message))


if __name__ == "__main__":
    main()

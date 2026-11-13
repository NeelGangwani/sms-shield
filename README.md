# SMS Shield

A small machine-learning tool that reads a text message and estimates how
likely it is to be a scam. Built as the FAI mini project (Practical 10) by
**Neel Gangvani** — B.Tech Cyber Security, Karnavati University.

Live portfolio: <https://sites.google.com/view/sms-shield-ai-mini-project>

## What it does

A message goes in as plain text. A TF-IDF vectoriser turns it into word
uni-grams and bi-grams, and a logistic-regression classifier returns the
probability that the wording belongs to the spam class.

```
$ python predict.py "your account will be blocked today verify kyc immediately"
SPAM      96.8%  your account will be blocked today verify kyc immediately
```

## Results

Trained on 1,998 messages, tested on the 500 it had never seen:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 0.9420 | 0.9308 | 0.9565 | 0.9435 | 0.9947 |
| **Logistic Regression** | **0.9440** | **0.9447** | **0.9447** | **0.9447** | **0.9952** |

Five-fold cross-validated F1 for the chosen model is 0.9554, so the test
score is not a lucky split. Confusion matrix: 233 genuine and 239 spam
correct, 14 false alarms, 14 scams missed.

The words that push a message hardest towards spam are `http`, `claim`,
`rs`, `kyc`, `win` and `offer4u xyz` — the model learnt the vocabulary of
a scam rather than any single keyword.

## Honest limitations

- **The corpus is generated, not collected.** `corpus.py` builds messages
  from documented spam-SMS wording patterns and ordinary college chatter.
  No real person's inbox was used, and public SMS-spam datasets were not
  reachable from the build environment. Real traffic is messier, so these
  scores should be read as a controlled experiment, not a field result.
- **Polite scams get through.** `sir your kyc is pending kindly share otp
  to continue` scores 42.3% and passes as genuine. It carries no link, no
  rupee amount and no shouting, which is exactly what the model leans on.
- English and transliterated wording only; no Gujarati or Hindi script.
- Wording, not metadata. A real filter would also weigh the sender ID,
  sending rate and any shortened links.

About 22% of the corpus is a deliberately ambiguous band where the same
sentence appears under both labels. Without it the model scored a
suspicious 100%; the overlap is what makes the 94% figure meaningful.

## Running it

```bash
pip install -r requirements.txt
python train.py        # builds the corpus, trains both models, writes figures/
python predict.py "paste a message here"
python app.py          # web interface on http://127.0.0.1:8082
```

## Files

| File | Purpose |
|---|---|
| `corpus.py` | Generates the labelled message corpus, including the ambiguous band |
| `train.py` | Trains and compares both models, saves `model.joblib`, `metrics.json`, `figures/` |
| `predict.py` | Command-line scoring |
| `app.py` | Flask web interface |
| `metrics.json` | The scores above, as written by the last training run |

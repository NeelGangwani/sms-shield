"""Minimal web interface for SMS Shield."""
import os

import joblib
from flask import Flask, render_template_string, request

HERE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(HERE, "model.joblib"))
app = Flask(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>SMS Shield</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial;background:#160f24;color:#efe9f7;margin:0;padding:40px}
 .card{max-width:720px;margin:0 auto;background:#241a38;border-radius:14px;padding:28px 32px;
       box-shadow:0 8px 30px rgba(0,0,0,.4)}
 h1{margin:0 0 4px;font-size:26px} .sub{color:#a795c4;margin-bottom:22px;font-size:14px}
 textarea{width:100%;height:90px;padding:12px 14px;border-radius:8px;border:1px solid #3d2f59;
       background:#1a1229;color:#efe9f7;font-size:15px;box-sizing:border-box;resize:vertical}
 button{margin-top:12px;padding:11px 22px;border:0;border-radius:8px;background:#7b4bbd;
       color:#fff;font-size:15px;cursor:pointer}
 .verdict{margin-top:24px;padding:18px;border-radius:10px;font-size:19px;font-weight:600}
 .bad{background:#3a1420;border:1px solid #7d2440;color:#ff9db0}
 .good{background:#10281c;border:1px solid #1f6b43;color:#79e0a8}
 .note{margin-top:16px;font-size:13px;color:#a795c4;line-height:1.5}
</style></head><body><div class="card">
<h1>SMS Shield</h1>
<div class="sub">Machine-learning check for scam text messages &middot; FAI Mini Project</div>
<form method="post">
  <textarea name="msg" placeholder="Paste a text message, e.g. your account will be blocked today verify kyc"
            autofocus>{{ msg or '' }}</textarea>
  <button type="submit">Check this message</button>
</form>
{% if msg %}
  <div class="verdict {{ 'bad' if risk >= 50 else 'good' }}">
    {{ 'Likely SPAM' if risk >= 50 else 'Looks genuine' }} &mdash; risk score {{ risk }}%
  </div>
  <div class="note">The score is the model's probability that the wording belongs to the
  spam class. Scores between roughly 40% and 60% are the ambiguous band, where polite
  scams and real notices read almost the same.</div>
{% endif %}
</div></body></html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    msg = request.form.get("msg", "").strip()
    risk = None
    if msg:
        risk = round(float(model.predict_proba([msg])[0][1]) * 100, 1)
    return render_template_string(PAGE, msg=msg, risk=risk)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8082)

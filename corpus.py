"""Builds the labelled SMS corpus SMS Shield learns from.

No real person's messages are used. The sentences are generated from
documented spam-SMS wording patterns plus ordinary college chatter, and a
deliberately ambiguous band is added so the two classes overlap the way
they do in a real inbox.
"""
import random

HAM_T = [
    "can you send me the {thing} before {time}",
    "i am reaching college by {time}, wait near the {place}",
    "mom asked you to bring {thing} on the way home",
    "the {subject} lab submission is due on {day}",
    "are you coming for {sport} practice today",
    "i forgot my {thing} in the {place}, please check",
    "lecture is cancelled, we are meeting in the {place}",
    "happy birthday bro, have a great {day}",
    "please share the {subject} notes when you are free",
    "call me after {time}, i am in a lecture right now",
    "the bus to {place} leaves at {time} tomorrow",
    "did you submit the {subject} assignment yet",
]

SPAM_T = [
    "congratulations you have won a free {prize} claim now at {link}",
    "your account will be blocked today verify kyc immediately {link}",
    "urgent your {bank} debit card is suspended update details {link}",
    "you are selected for rs {amt} cash prize reply yes to claim",
    "limited offer get {pct} percent discount only today click {link}",
    "your parcel is held at customs pay rs {amt} now {link}",
    "earn rs {amt} daily from home no investment whatsapp {num}",
    "loan approved rs {amt} instant transfer apply here {link}",
    "your {bank} netbanking password expires today reset {link}",
    "final reminder claim your free {prize} before midnight {link}",
    "you have unclaimed refund of rs {amt} submit bank details {link}",
    "win a brand new {prize} just answer one question {link}",
]

# The same sentences appear as genuine notices AND as scams, so the model
# cannot separate the classes perfectly - this is what stops the accuracy
# from coming out at an unrealistic 100%.
AMBIGUOUS = [
    "urgent your payment of rs {amt} is pending please confirm today",
    "congratulations you have been selected kindly reply to confirm",
    "your account details need verification before {day}",
    "claim your refund of rs {amt} by sharing the required details",
    "final reminder your {bank} form is incomplete click {link}",
    "dear student your fee receipt is ready verify at {link}",
    "you have won the prize please share your account number",
    "important notice your access will be blocked kindly update",
]

FILLERS = dict(
    thing=["charger", "notebook", "umbrella", "id card", "pen drive", "water bottle"],
    time=["9 am", "4 pm", "6:30", "noon", "evening"],
    place=["library", "canteen", "lab", "hostel gate", "parking", "auditorium"],
    subject=["ai", "python", "maths", "networks", "web design"],
    day=["monday", "friday", "sunday", "weekend"],
    sport=["badminton", "football", "cricket", "basketball"],
    prize=["smartphone", "laptop", "gift voucher", "smart watch", "scooter"],
    link=["http://bit-win.tk/claim", "http://secure-verify.ga/kyc", "http://offer4u.xyz/go"],
    bank=["bank", "netbank", "card"],
    amt=["5000", "25000", "1,50,000", "899"],
    pct=["70", "90", "80"],
    num=["90xxxx1234", "80xxxx9876"],
)

AMBIGUOUS_SHARE = 0.22   # size of the overlapping band, as a share of n
AMBIGUOUS_HAM_ODDS = 0.45


def fill(template, rng=random):
    out = template
    for key, values in FILLERS.items():
        out = out.replace("{%s}" % key, rng.choice(values))
    return out


def build(n=900, seed=7):
    """Returns a shuffled list of (message, label) with label 1 = spam."""
    rng = random.Random(seed)
    rows = [(fill(rng.choice(HAM_T), rng), 0) for _ in range(n)]
    rows += [(fill(rng.choice(SPAM_T), rng), 1) for _ in range(n)]
    for _ in range(int(n * AMBIGUOUS_SHARE)):
        text = fill(rng.choice(AMBIGUOUS), rng)
        rows.append((text, 0 if rng.random() < AMBIGUOUS_HAM_ODDS else 1))
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = build()
    spam = sum(r[1] for r in rows)
    print("messages: %d  spam: %d  genuine: %d" % (len(rows), spam, len(rows) - spam))
    for text, label in rows[:6]:
        print(" ", "SPAM" if label else "ham ", text)

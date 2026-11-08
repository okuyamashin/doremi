import sys, wave, json
import numpy as np

d, out = sys.argv[1], sys.argv[2]
def load(n):
    with wave.open(f"{d}/{n}.wav") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32).reshape(-1, 2).mean(1) / 32768
    return np.convolve(x, np.ones(3) / 3, mode="same")[::3]  # 16k
SR = 16000
harm = sum(load(n) for n in ["4 Guitar", "5 Keyboard", "6 Strings", "7 Synth", "8 Brass", "1 Backing Vocals"])
bass = load("3 Bass")
N, HOP = 8192, 1600  # 0.51s window, 0.1s hop
win = np.hanning(N).astype(np.float32)
freqs = np.fft.rfftfreq(N, 1 / SR)
def chroma(x, lo, hi):
    n = (len(x) - N) // HOP
    sel = (freqs >= lo) & (freqs <= hi)
    pcs = (np.round(12 * np.log2(freqs[sel] / 440)) + 9) % 12
    pcs = pcs.astype(int)
    C = np.zeros((n, 12))
    for i in range(n):
        m = np.abs(np.fft.rfft(x[i * HOP:i * HOP + N] * win))[sel]
        m = np.sqrt(m)
        np.add.at(C[i], pcs, m)
    return C
Ch = chroma(harm, 100, 2000)
Cb = chroma(bass, 35, 260)
n = min(len(Ch), len(Cb))
Ch, Cb = Ch[:n], Cb[:n]
energy = Ch.sum(1) + Cb.sum(1)
Ch = Ch / (np.linalg.norm(Ch, axis=1, keepdims=True) + 1e-9)
Cb = Cb / (Cb.max(1, keepdims=True) + 1e-9)

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
chords = []
for r in range(12):
    for q, iv in (("", (0, 4, 7)), ("m", (0, 3, 7))):
        t = np.zeros(12); t[[(r + i) % 12 for i in iv]] = 1; t[r] = 1.3
        chords.append((r, q, t / np.linalg.norm(t)))
S = np.array([[Ch[i] @ c[2] + 0.35 * Cb[i][c[0]] for c in chords] for i in range(n)])
# 無音は「なし」
silent = energy < np.percentile(energy, 5) * 1.5
# Viterbi（同じコードに留まりやすく）
K = len(chords); STAY = 0.0; SWITCH = -0.6
V = S[0].copy(); back = np.zeros((n, K), dtype=int)
for i in range(1, n):
    best_prev = V.argmax()
    stay = V + STAY
    sw = V[best_prev] + SWITCH
    back[i] = np.where(stay >= sw, np.arange(K), best_prev)
    V = np.maximum(stay, sw) + S[i]
path = np.zeros(n, dtype=int); path[-1] = V.argmax()
for i in range(n - 1, 0, -1):
    path[i - 1] = back[i][path[i]]

segs = []
t_of = lambda i: (i * HOP + N / 2) / SR
for i in range(n):
    k = -1 if silent[i] else int(path[i])
    if segs and segs[-1]["k"] == k:
        segs[-1]["end"] = i + 1
    else:
        segs.append({"k": k, "start": i, "end": i + 1})
res = []
for s in segs:
    t0, t1 = t_of(s["start"]) - HOP / SR / 2, t_of(s["end"]) - HOP / SR / 2
    if s["k"] < 0 or t1 - t0 < 0.25:
        continue
    r, q, _ = chords[s["k"]]
    if res and res[-1]["root"] == r and res[-1]["q"] == q and t0 - (res[-1]["t"] + res[-1]["d"]) < 0.3:
        res[-1]["d"] = round(t1 - res[-1]["t"], 2); continue
    res.append({"t": round(max(0, t0), 2), "d": round(t1 - t0, 2), "root": r, "q": q, "name": NAMES[r] + q})
json.dump({"chords": res}, open(out, "w"), ensure_ascii=False)
from collections import Counter
cnt = Counter()
for c in res: cnt[c["name"]] += c["d"]
print(len(res), "chords;", [(k, round(v, 1)) for k, v in cnt.most_common(10)])
print(" ".join(f'{c["t"]}:{c["name"]}' for c in res[:40]))

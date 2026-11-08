import sys, wave, json
import numpy as np

path, out = sys.argv[1], sys.argv[2]
with wave.open(path) as w:
    sr0 = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32).reshape(-1, w.getnchannels()).mean(1) / 32768
# 48k -> 16k (簡易ローパス + 間引き)
k = np.ones(5) / 5
x = np.convolve(x, k, mode="same")[::3]
SR = sr0 // 3
HOP, W = 160, 640           # 10ms, 40ms
TMIN, TMAX = SR // 1000, SR // 70
n = (len(x) - W - TMAX) // HOP
f0 = np.zeros(n); conf = np.zeros(n); rms = np.zeros(n)
for i in range(n):
    a = i * HOP
    seg = x[a:a + W + TMAX]
    fr = seg[:W]
    rms[i] = np.sqrt(np.mean(fr * fr))
    if rms[i] < 0.01:
        continue
    # 差分関数 d(tau)
    e = np.cumsum(np.concatenate([[0], seg * seg]))
    e0 = e[W] - e[0]
    et = e[np.arange(TMAX + 1) + W] - e[np.arange(TMAX + 1)]
    F = np.fft.rfft(seg, 2 * len(seg))
    G = np.fft.rfft(fr[::-1], 2 * len(seg))
    corr = np.fft.irfft(F * G)[W - 1:W + TMAX]
    d = e0 + et - 2 * corr
    d[0] = 0
    cm = np.ones_like(d)
    cs = np.cumsum(d[1:])
    cm[1:] = d[1:] * np.arange(1, TMAX + 1) / np.maximum(cs, 1e-12)
    tau = -1
    for t in range(TMIN, TMAX):
        if cm[t] < 0.15:
            while t + 1 < TMAX and cm[t + 1] < cm[t]:
                t += 1
            tau = t
            break
    if tau < 0:
        continue
    a_, b_, c_ = cm[tau - 1], cm[tau], cm[tau + 1]
    den = a_ + c_ - 2 * b_
    sh = (a_ - c_) / (2 * den) if den else 0
    f0[i] = SR / (tau + sh)
    conf[i] = 1 - b_

midi = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1) / 440), np.nan)
# メディアンで平滑化
sm = midi.copy()
for i in range(2, n - 2):
    w5 = midi[i - 2:i + 3]
    if np.isnan(midi[i]) or np.sum(~np.isnan(w5)) < 3:
        continue
    sm[i] = np.nanmedian(w5)

notes = []
cur = None
def close(c):
    if c and (c["end"] - c["start"]) * HOP / SR >= 0.08:
        m = float(np.median(c["vals"]))
        notes.append({"t": round(c["start"] * HOP / SR, 3), "d": round((c["end"] - c["start"]) * HOP / SR, 3), "m": int(round(m)), "c": round((m - round(m)) * 100)})
for i in range(n):
    v = sm[i]
    if np.isnan(v):
        if cur and i - cur["end"] > 3:
            close(cur); cur = None
        continue
    if cur and abs(v - np.median(cur["vals"][-8:])) < 0.7 and i - cur["end"] <= 3:
        cur["vals"].append(v); cur["end"] = i + 1
    else:
        close(cur)
        cur = {"start": i, "end": i + 1, "vals": [v]}
close(cur)

# 同じ音で隙間が短いものはつなぐ
merged = []
for nt in notes:
    if merged and merged[-1]["m"] == nt["m"] and nt["t"] - (merged[-1]["t"] + merged[-1]["d"]) < 0.05:
        merged[-1]["d"] = round(nt["t"] + nt["d"] - merged[-1]["t"], 3)
    else:
        merged.append(nt)

# 調の推定（Krumhansl のメジャープロファイル）
prof = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
hist = np.zeros(12)
for nt in merged:
    hist[nt["m"] % 12] += nt["d"]
scores = [np.corrcoef(np.roll(prof, k), hist)[0, 1] for k in range(12)]
key = int(np.argmax(scores))
ms = [nt["m"] for nt in merged]
json.dump({"key": key, "notes": merged}, open(out, "w"), ensure_ascii=False)
print(len(merged), "notes; key", ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][key], "major; range", min(ms), max(ms), "median", int(np.median(ms)))
print("first notes", merged[:12])

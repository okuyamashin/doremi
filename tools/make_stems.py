import sys, wave, numpy as np
d, outdir = sys.argv[1], sys.argv[2]
groups = {"vocal": ["0 Lead Vocals"], "bvox": ["1 Backing Vocals"], "drums": ["2 Drums"],
          "other": ["3 Bass", "4 Guitar", "5 Keyboard", "6 Strings", "7 Synth", "8 Brass"]}
for name, files in groups.items():
    acc = None
    for f in files:
        with wave.open(f"{d}/{f}.wav") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        acc = x if acc is None else acc + x
    acc *= min(1.0, 32000 / np.abs(acc).max())
    with wave.open(f"{outdir}/{name}.wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000); w.writeframes(acc.astype(np.int16).tobytes())
    print(name, "ok")

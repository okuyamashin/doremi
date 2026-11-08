import sys, wave, numpy as np
d, out = sys.argv[1], sys.argv[2]
names=["1 Backing Vocals","2 Drums","3 Bass","4 Guitar","5 Keyboard","6 Strings","7 Synth","8 Brass"]
acc=None
for n in names:
    with wave.open(f"{d}/{n}.wav") as w:
        x=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)
    acc = x if acc is None else acc + x
pk=np.abs(acc).max(); acc = acc * min(1.0, 32000/pk)
with wave.open(out,'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(48000); w.writeframes(acc.astype(np.int16).tobytes())
print('peak scale', min(1.0, 32000/pk))

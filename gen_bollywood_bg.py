import numpy as np, wave
sr=44100
DUR=72.0
n=int(sr*DUR); t=np.linspace(0,DUR,n,endpoint=False)
audio=np.zeros(n)
def place(sig,start):
    s=int(sr*start); e=min(n,s+len(sig)); audio[s:e]+=sig[:e-s]

# Scale: Raag Bhairavi-ish around Sa=C3 (130.81)
Sa=130.81
def note(semis): return Sa*2**(semis/12)
# tanpura drone Sa-Pa
for f,a in ((Sa,0.06),(note(7),0.045),(Sa*2,0.03)):
    audio+=np.sin(2*np.pi*f*t)*a*(0.7+0.3*np.sin(2*np.pi*0.08*t))

# Harmonium chords (sustained), gentle reedy tone via harmonics
def reed(f,dur,amp):
    tt=np.linspace(0,dur,int(sr*dur),endpoint=False)
    s=sum(np.sin(2*np.pi*f*h*tt)*(amp/h) for h in (1,2,3,4))
    e=np.ones(len(tt)); a=int(sr*0.05); r=int(sr*0.2)
    e[:a]=np.linspace(0,1,a); e[-r:]=np.linspace(1,0,r)
    return s*e
chords=[(0,[0,4,7]),(8,[5,9,12]),(16,[7,11,14]),(24,[0,4,7])]
bar=8
for start in range(0,int(DUR),bar):
    for semi in [0,4,7]:
        place(reed(note(semi),bar,0.03),start)

# Bansuri-style melody line (sine + slight vibrato), simple phrase
mel=[0,2,4,7,4,2,0,-1,0,4,7,11,7,4,2,0]
mdur=0.8
melbuf=np.zeros(n)
for i,semi in enumerate(mel*6):
    st=6.0+i*mdur
    if st>=DUR: break
    tt=np.linspace(0,mdur,int(sr*mdur),endpoint=False)
    vib=1+0.006*np.sin(2*np.pi*5*tt)
    s=np.sin(2*np.pi*note(semi)*tt*vib)+0.3*np.sin(2*np.pi*note(semi)*2*tt)
    e=np.ones(len(tt)); a=int(sr*0.04); r=int(sr*0.1)
    e[:a]=np.linspace(0,1,a); e[-r:]=np.linspace(1,0,r)
    place(s*e*0.05,st)

# Tabla-ish rhythm: teentaal feel, dha/tin thumps
def tabla_hit(f,dur,amp,noisy=0.0):
    tt=np.linspace(0,dur,int(sr*dur),endpoint=False)
    s=np.sin(2*np.pi*f*tt)*np.exp(-tt*22)
    if noisy: s+=noisy*np.random.randn(len(tt))*np.exp(-tt*40)
    return s*amp
beat=0.42
pattern=[('dha',1),('dhin',0.7),('dhin',0.7),('dha',1),('dha',1),('tin',0.6),('tin',0.6),('ta',0.8)]
i=0; tcur=4.0
while tcur<DUR-2:
    name,amp=pattern[i%len(pattern)]
    f=190 if name in('dha','dhin') else 320
    place(tabla_hit(f,0.18,0.16*amp,0.3 if name=='tin' else 0.15),tcur)
    tcur+=beat; i+=1

# fades
fi=int(sr*1.5); fo=int(sr*2.5)
audio[:fi]*=np.linspace(0,1,fi); audio[-fo:]*=np.linspace(1,0,fo)
audio=audio/np.max(np.abs(audio))*0.8
pcm=(audio*32767).astype(np.int16)
with wave.open("bg_bollywood.wav","w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
print("bollywood bg done")

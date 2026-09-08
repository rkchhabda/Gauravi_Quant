import numpy as np, wave
sr=44100; DUR=80.0
n=int(sr*DUR); t=np.linspace(0,DUR,n,endpoint=False)
audio=np.zeros(n)
def place(sig,start,buf=None):
    b=audio if buf is None else buf
    s=int(sr*start); e=min(len(b),s+len(sig)); b[s:e]+=sig[:e-s]
Sa=146.83  # D3
def note(s): return Sa*2**(s/12)

# ---- Tanpura drone (Sa-Pa-Sa), plucked shimmer ----
def tanpura(f,dur):
    tt=np.linspace(0,dur,int(sr*dur),endpoint=False)
    s=np.zeros(len(tt)); pl=1.5
    for k in range(int(dur/pl)):
        seg=int(sr*pl); o=k*seg
        x=np.linspace(0,pl,seg,endpoint=False)
        tone=sum(np.sin(2*np.pi*f*h*x)*(1/h) for h in (1,2,3,4,5))
        tone*=np.exp(-x*1.2)
        s[o:o+seg]+=tone[:len(s)-o]
    return s
for f,a in ((Sa,0.05),(note(7),0.04),(Sa*2,0.03)):
    place(tanpura(f,DUR)*a,0)

# ---- Harmonium pad chord progression with a lift in chorus ----
def reed(f,dur,amp):
    tt=np.linspace(0,dur,int(sr*dur),endpoint=False)
    s=sum(np.sin(2*np.pi*f*h*tt)*(amp/h) for h in (1,2,3,4,5))
    e=np.ones(len(tt)); a=int(sr*0.08); r=int(sr*0.3)
    e[:a]=np.linspace(0,1,a); e[-r:]=np.linspace(1,0,r); return s*e
# verse chords vs chorus chords (lift = higher voicing + louder)
prog=[(0,[0,3,7],0.028),(8,[5,8,12],0.028),(16,[0,3,7],0.028),
      (24,[7,10,14],0.04),(32,[5,9,12],0.04),(40,[0,4,7],0.04),  # chorus lift
      (48,[0,3,7],0.028),(56,[5,8,12],0.028),(64,[7,10,14],0.04),(72,[0,3,7],0.03)]
for start,chord,amp in prog:
    for semi in chord: place(reed(note(semi),8,amp),start)

# ---- Sitar-ish plucks (bright, decaying) for texture ----
def pluck(f,dur,amp):
    tt=np.linspace(0,dur,int(sr*dur),endpoint=False)
    s=sum(np.sin(2*np.pi*f*h*tt)*(1/(h*h)) for h in range(1,9))
    return s*np.exp(-tt*3.5)*amp
sitar_phrase=[0,3,5,7,10,7,5,3,0,-2,0,3,7,10,7,5]
for i,semi in enumerate(sitar_phrase*7):
    st=2.0+i*0.5
    if st>DUR-1: break
    place(pluck(note(semi+12),0.5,0.045),st)

# ---- Bansuri lead melody (breathy sine + vibrato), a real singable line ----
lead=[0,2,3,5,7,5,3,2, 0,2,3,7,5,3,2,0, 7,5,3,5,7,10,7,5, 3,5,7,10,12,10,7,5]
mdur=1.0
for i,semi in enumerate(lead*3):
    st=6.0+i*mdur
    if st>DUR-1: break
    tt=np.linspace(0,mdur,int(sr*mdur),endpoint=False)
    vib=1+0.007*np.sin(2*np.pi*5.5*tt)
    s=np.sin(2*np.pi*note(semi)*tt*vib)+0.25*np.sin(2*np.pi*note(semi)*2*tt)
    s+=0.05*np.random.randn(len(tt))*np.exp(-tt*3)  # breath
    e=np.ones(len(tt)); a=int(sr*0.06); r=int(sr*0.15)
    e[:a]=np.linspace(0,1,a); e[-r:]=np.linspace(1,0,r)
    place(s*e*0.06,st)

# ---- Strings swell pad under chorus ----
def strings(f,dur,amp):
    tt=np.linspace(0,dur,int(sr*dur),endpoint=False)
    det=sum(np.sin(2*np.pi*f*(1+d)*tt) for d in (-0.004,0,0.004))
    e=np.hanning(len(tt)); return det*e*amp
for start in (24,64):
    for semi in (0,7,12): place(strings(note(semi),16,0.02),start)

# ---- Tabla groove: teentaal, stronger, with sam accent ----
def hit(f,dur,amp,noisy=0.0):
    tt=np.linspace(0,dur,int(sr*dur),endpoint=False)
    s=np.sin(2*np.pi*f*tt)*np.exp(-tt*20)
    s+=0.5*np.sin(2*np.pi*f*0.5*tt)*np.exp(-tt*12)
    if noisy: s+=noisy*np.random.randn(len(tt))*np.exp(-tt*45)
    return s*amp
beat=0.40
# 16-beat teentaal: strong on 1(sam),9; dhin/na/tin pattern
pat=[('DHA',1.0),('dhin',0.6),('dhin',0.6),('dha',0.8),
     ('dha',0.8),('tin',0.55),('tin',0.55),('ta',0.7),
     ('DHA',0.95),('dhin',0.6),('dhin',0.6),('dha',0.8),
     ('dha',0.8),('dhin',0.6),('dhin',0.6),('na',0.65)]
i=0; tc=4.0
while tc<DUR-2:
    name,amp=pat[i%16]
    bass = name in ('DHA','dha','dhin','na')
    f=150 if name=='DHA' else (185 if bass else 330)
    place(hit(f,0.2,0.18*amp,0.35 if name in('tin','ta','na') else 0.12),tc)
    if bass: place(hit(330,0.12,0.06*amp,0.3),tc)  # treble layer
    tc+=beat; i+=1

# section fades: intro build + outro
fi=int(sr*2.0); fo=int(sr*3.0)
audio[:fi]*=np.linspace(0,1,fi); audio[-fo:]*=np.linspace(1,0,fo)
# light compression + normalize
audio=np.tanh(audio*1.4)
audio=audio/np.max(np.abs(audio))*0.85
pcm=(audio*32767).astype(np.int16)
with wave.open("bg_rich.wav","w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
print("rich bg done")

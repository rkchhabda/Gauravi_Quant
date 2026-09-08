import numpy as np, wave
sr=44100
DUR=float("66.360000")+3.0
n=int(sr*DUR)
t=np.linspace(0,DUR,n,endpoint=False)
audio=np.zeros(n)

def env(sig,a=0.01,r=0.05):
    e=np.ones(len(sig)); ai=int(sr*a); ri=int(sr*r)
    if ai>0: e[:ai]=np.linspace(0,1,ai)
    if ri>0: e[-ri:]=np.linspace(1,0,ri)
    return sig*e

# --- Railway whistle: two-tone steam whistle ---
def whistle(dur):
    tt=np.linspace(0,dur,int(sr*dur),endpoint=False)
    w=np.zeros(len(tt))
    for f in (620,930,1560,1240):
        w+=np.sin(2*np.pi*f*tt)*(1.0/ (1+abs(620-f)/400))
    vib=1+0.008*np.sin(2*np.pi*6*tt)
    w=w*vib
    # breathy noise
    w+=0.15*np.random.randn(len(tt))*np.exp(-tt*0.5)
    return env(w/np.max(np.abs(w)),0.08,0.25)

# place whistles: intro, middle, end
def place(sig,start):
    s=int(sr*start); e=min(n,s+len(sig)); audio[s:e]+=sig[:e-s]

wh=whistle(1.6)*0.32
place(wh,0.2)
place(wh,DUR/2-0.8)
place(whistle(2.2)*0.3,DUR-2.6)

# --- Chugging rhythm (train on tracks), soft, low ---
chug=np.zeros(n)
period=0.5  # seconds per chug
for k in range(int(DUR/period)):
    st=int(sr*(k*period))
    ln=int(sr*0.12)
    seg=np.random.randn(ln)*np.exp(-np.linspace(0,8,ln))
    # low thump
    tt=np.linspace(0,0.12,ln,endpoint=False)
    seg=seg*0.4+np.sin(2*np.pi*70*tt)*np.exp(-tt*20)
    e=min(n,st+ln); chug[st:e]+=seg[:e-st]
audio+=chug*0.10

# --- Soft harmonium-style drone (tonic + fifth), gentle ---
for f,amp in ((146.83,0.05),(220.0,0.035),(293.66,0.025)):
    audio+=np.sin(2*np.pi*f*t)*amp*(0.6+0.4*np.sin(2*np.pi*0.1*t))

# gentle overall fade in/out
fi=int(sr*1.0); fo=int(sr*2.0)
audio[:fi]*=np.linspace(0,1,fi)
audio[-fo:]*=np.linspace(1,0,fo)

audio=audio/np.max(np.abs(audio))*0.8
pcm=(audio*32767).astype(np.int16)
with wave.open("bg_railway.wav","w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
print("bg done", DUR)

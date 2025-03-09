import numpy as np # linear algebra
import IPython.display as ipd

def beep(rate=8000, freq=440*2, dur=0.2, vol=0.1, ramp=0.1):
  steps = int(rate*dur)
  #amp = vol * np.ones( shape=(steps,) )  # Nasty beep
  amp = vol * np.concat( [np.linspace(0,1., int(steps*ramp)), np.linspace(1.,0,int(steps*(1.-ramp))+10) ])  # ping
  tone = np.sin(2*np.pi*freq*np.arange(steps)/rate) # sine wave
  #tone = np.sign( tone ) # Make it into a square wave - unpleasant
  return ipd.Audio(tone * amp[:steps], rate=rate, autoplay=True)
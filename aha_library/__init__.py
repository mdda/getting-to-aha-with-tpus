import numpy as np # linear algebra
import IPython.display as ipd

def beep(rate=8000, freq=440, dur=0.2, vol=0.1):
  wav = np.sin(2*np.pi*freq*np.arange(rate*dur)/rate)*vol
  return ipd.Audio(wav, rate=rate, autoplay=True)
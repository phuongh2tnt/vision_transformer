import numpy as np
import matplotlib.pyplot as plt
import pywt
train_dataset = ImageFolder(root='/content/drive/My Drive/AI/el/datasetelearning/train', transform=transform)
Fs = 128;
fb = cwtfilterbank(SignalLength=1000, ...
    SamplingFrequency=Fs, ...
    VoicesPerOctave=12);
sig = train_dataset.Data(1,1:1000);
[cfs,frq] = wt(fb,sig);
t = (0:999)/Fs;
figure
pcolor(t,frq,abs(cfs))
set(gca,"yscale","log")
shading interp
axis tight
title("Scalogram")
xlabel("Time (s)")
ylabel("Frequency (Hz)")

import numpy as np
from scipy.signal import lfilter
from scikits.talkbox import lpc
import math


def getFormants(Fs, signal):
    signalnew = signal * np.hamming(len(signal))
    signalnew = lfilter([1], [1., 0.63], signalnew)

    ncoeff = 2 + Fs / 1000
    A, e, k = lpc(signalnew, ncoeff)
    for i in range(0, len(A)):
        if math.isnan(A[i]) or math.isinf(A[i]):
            A[i] = 0.0
    A = np.array(A)

    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    angz = np.arctan2(np.imag(rts), np.real(rts))
    angz = angz * (Fs / (2 * math.pi))

    # indices = sorted(range(len(angz)), key=lambda k: angz[k])
    frqs = sorted(angz)

    # bw = []
    # for i in range(0, len(indices)):
    #     bw.append(-1 / 2 * (Fs / (2 * math.pi)) * math.log(abs(rts[indices[i]])))
    # bw = np.array(bw)
    #
    # formants = []
    # for kk in range(0, len(frqs)):
    #     if (frqs[kk] > 90 and bw[kk] < 400):
    #         formants.append(frqs[kk])

    return np.array(frqs[0:5])

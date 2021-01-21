import rescomp as rc
import sys
import numpy as np
import scipy as sp

system = sys.argv[1]
T = float(sys.argv[2])
if len(sys.argv) < 4:
    N = 100
else:
    N = int(sys.argv[3])

def orbit_fft(U):
    N = U.shape[0]
    Uf = sp.fft.fft(U, axis=0)/N
    return Uf

t, U = rc.orbit(system, duration=T, trim=True)
meanUf = orbit_fft(U)
for i in range(N-1):
    if i % 50 == 0:
        print(f"{i}th obit complete")
    t, U = rc.orbit(system, duration=T, trim=True)
    meanUf += orbit_fft(U)

np.save(f"{system}_mean_fourier.npy", meanUf/N)

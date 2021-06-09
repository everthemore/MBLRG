from MBLHamiltonian import MBLHamiltonian
import os
import sys
import numpy as np

def doFlow(H, threshold):
    h = []
    J = []

    hstep, Jstep = H.getCoefficientDistributions()
    h.append(hstep)
    J.append(Jstep)

    maxSteps = 500
    currentStep = 0
    finished = False

    evals_vs_step = [np.linalg.eigh(H.H.toMatrix())[0]]
    while not finished:
        print("")
        print("Flow step #%d"%currentStep)

        if( currentStep != 0 and currentStep %50 == 0 ):
            evals, evecs = np.linalg.eigh(H.H.toMatrix())
            evals_vs_step.append(evals)

        finished = H.rotateOut(threshold)
        hstep, Jstep = H.getCoefficientDistributions()

        h.append(hstep)
        J.append(Jstep)

        currentStep += 1
        if currentStep > maxSteps:
            break

    data = {'h':np.array(h), 'J':np.array(J), 'evals_vs_step':np.array(evals_vs_step)}
    return data
    

if __name__ == "__main__":
    output = sys.argv[1]
    os.makedirs("{0}/data/raw/".format(output), exists_ok=True)
    
    L = int(sys.argv[2])
    hscale = float(sys.argv[3])
    Jscale = float(sys.argv[4])
    Uscale = float(sys.argv[5])
    seed = int(sys.argv[6])

    np.random.seed(seed)
    H = MBLHamiltonian(L, hscale, Jscale, Uscale)

    data = doFlow(H, threshold=1e-5)
    np.save("{0}/data/raw/hJ-L-{1}-h-{2}-U-{3}-J-{4}-seed-{5}.npy".format(output,L,hscale,Uscale,Jscale,seed), data, allow_pickle=True)

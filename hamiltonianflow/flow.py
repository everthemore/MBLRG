from MBLHamiltonian import MBLHamiltonian
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
    np.save("data/hJ-L-{0}-h-{1}-U-{2}-J-{3}-seed-{4}.npy".format(L,hscale,Uscale,Jscale,seed), data, allow_pickle=True)

if __name__ == "__main__":
    L = int(sys.argv[1])
    hscale = float(sys.argv[2])
    Jscale = float(sys.argv[3])
    Uscale = float(sys.argv[4])
    seed = int(sys.argv[5])

    np.random.seed(seed)
    H = MBLHamiltonian(L, hscale, Jscale, Uscale)

    doFlow(H, threshold=1e-5)

from MBLHamiltonian import *
from pathlib import Path

def doFlow(L, hscale, Jscale, Uscale, seed, method):
    np.random.seed(seed)
    H = MBLHamiltonian(L,hscale, Jscale, Uscale)

    h = []
    J = []
    energyscales = []
    hstep, Jstep = H.getCoefficientDistributions()
    h.append(hstep)
    J.append(Jstep)

    maxSteps = 50
    currentStep = 0
    finished = False

    while not finished:
        finished, energyscale = H.rotateOut(method)
        energyscales.append(energyscale)

        hstep, Jstep = H.getCoefficientDistributions()
        h.append(hstep)
        J.append(Jstep)

        currentStep += 1
        if currentStep > maxSteps:
            break

        print(currentStep)

    np.save("data/compare-norms/flowresult-L-{0}-seed-{1}-method-{2}.npy".format(L,seed,method), np.array([h,J]), allow_pickle=True)
    np.savetxt("data/compare-norms/energyscales-L-{0}-seed-{1}-method-{2}.txt".format(L,seed,method), np.array(energyscales))

if __name__ == "__main__":
    L = int(sys.argv[1])
    hscale = float(sys.argv[2])
    Jscale = float(sys.argv[3])
    Uscale = float(sys.argv[4])
    seed = int(sys.argv[5])

    Path("data/compare-norms").mkdir(parents=True, exist_ok=True)
    doFlow(L, hscale, Jscale, Uscale, seed, 0)
    doFlow(L, hscale, Jscale, Uscale, seed, 1)

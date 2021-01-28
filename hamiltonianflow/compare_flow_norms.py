from MBLHamiltonian import *
from pathlib import Path

def doFlow(L, hscale, Jscale, Uscale, seed, method, threshold=1e-8):
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

    np.save("data/compare-norms/threshold-{6}/flowresult-L-{0}-h-{1}-U-{2}-J-{3}-seed-{4}-method-{5}.npy".format(L,hscale,Uscale,Jscale,seed,method,threshold), np.array([h,J]), allow_pickle=True)
    np.savetxt("data/compare-norms/threshold-{6}/energyscales-L-{0}-h-{1}-U-{2}-J-{3}-seed-{4}-method-{5}.txt".format(L,hscale,Uscale,Jscale,seed,method,threshold), np.array(energyscales))

if __name__ == "__main__":
    L = int(sys.argv[1])
    hscale = float(sys.argv[2])
    Jscale = float(sys.argv[3])
    Uscale = float(sys.argv[4])
    seed = int(sys.argv[5])

    for threshold in [1e-3, 1e-5, 1e-8]:
        Path("data/compare-norms/threshold-{0}".format(threshold)).mkdir(parents=True, exist_ok=True)
        doFlow(L, hscale, Jscale, Uscale, seed, 0, threshold)
        doFlow(L, hscale, Jscale, Uscale, seed, 1, threshold)

from MBLHamiltonian import *

L = int(sys.argv[1])
hscale = float(sys.argv[2])
Jscale = float(sys.argv[3])
Uscale = float(sys.argv[4])
seed = int(sys.argv[5])
np.random.seed(seed)

H = MBLHamiltonian(L,hscale, Jscale, Uscale)

print("Starting H")
for t in H.H.opterms:
    print(t)

h = []
J = []
energyscales = []
hstep, Jstep = H.getCoefficientDistributions()
h.append(hstep)
J.append(Jstep)

maxSteps = 300
currentStep = 0
finished = False

while not finished:
    finished, energyscale = H.rotateOut()
    energyscales.append(energyscale)

    hstep, Jstep = H.getCoefficientDistributions()
    
    # [J,h] = energy diff, average that instead of <h>

    h.append(hstep)
    J.append(Jstep)

    currentStep += 1
    if currentStep > maxSteps:
        break

    if (currentStep % 10 == 0) and (currentStep != 0):
        np.save("data/W-{2}/chkpts/flowresult-L-{0}-seed-{1}.npy.chkpt".format(L,seed,hscale), np.array([h,J]), allow_pickle=True)
        np.savetxt("data/W-{2}/chkpts/energyscales-L-{0}-seed-{1}.txt.chkpt".format(L,seed,hscale), np.array(energyscales))

    print(currentStep)
    for k in Jstep.keys():
        print("range ", k)
        print(np.mean(Jstep[k]))
    print("")

np.save("data/W-{2}/flowresult-L-{0}-seed-{1}.npy".format(L,seed,hscale), np.array([h,J]), allow_pickle=True)
np.savetxt("data/W-{2}/energyscales-L-{0}-seed-{1}.txt".format(L,seed,hscale), np.array(energyscales))

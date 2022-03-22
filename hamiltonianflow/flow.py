from MBLHamiltonian import MBLHamiltonian
import os
import sys
import numpy as np

def doFlow(H, threshold):
    hJ = []

    hJ.append( H.H.terms )

    maxSteps = 500
    currentStep = 0
    finished = False

    eigenvalues = [np.linalg.eigh(H.H.toMatrix())[0]]
    steps = []
    while not finished:
        finished = H.rotateOut(threshold)

        if currentStep % 10 == 0:
            steps.append(currentStep)
            hJ.append(H.H.terms)

        currentStep += 1
        if currentStep > maxSteps:
            break

    eigenvalues.append(np.linalg.eigh(H.H.toMatrix())[0])
    data = {'hJ':hJ, 'start_and_final_evals':np.array(eigenvalues), 'steps':steps}
    return data 

if __name__ == "__main__":
    output = sys.argv[1]
    L = int(sys.argv[2])
    
    os.makedirs("{0}/L-{1}/raw/".format(output,L), exists_ok=True)
    
    hscale = float(sys.argv[3])
    Jscale = float(sys.argv[4])
    Uscale = float(sys.argv[5])
    seed = int(sys.argv[6])

    np.random.seed(seed)
    H = MBLHamiltonian(L, hscale, Jscale, Uscale)

    data = doFlow(H, threshold=1e-5)
    np.save("{0}/L-{1}/raw/hJ-L-{1}-h-{2}-U-{3}-J-{4}-seed-{5}.npy".format(output,L,hscale,Uscale,Jscale,seed), data, allow_pickle=True)

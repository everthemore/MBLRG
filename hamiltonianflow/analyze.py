from MBLHamiltonian import MBLHamiltonian
import os
import sys
import numpy as np
import numpy.ma as ma
import numpy as np

def isdiagonal(term):
    return (('1' not in term) and ('2' not in term))

def get_range(term):
    tmpterm = "".join(['1' if c != '0' else '0' for c in term])

    if(tmpterm == "0"*len(term)):
        return 0

    start = tmpterm.find('1')
    tmpterm = tmpterm[::-1]
    end = len(term) - tmpterm.find('1')
    return end - start

def sort_by_range(L,terms):
    # Zero out the dictionaries
    h = {}
    for r in range(0,L+1):
      h[r] = []

    J = {}
    for r in range(2,L+1):
      J[r] = []

    # This list only contains each conjugate once
    for term in terms:
        r = get_range(term)

        # Track diagonals
        if not isdiagonal(term):
            J[r].append(np.abs(terms[term]))
        else:
            h[r].append(np.abs(terms[term]))
    
    return h, J

def load_data(output,L,h,U,J,seed):
    data = np.atleast_2d(np.load("%s/L-%d/raw/hJ-L-%d-h-%.1f-U-%.1f-J-%.1f-seed-%d.npy"%(output,L,L,h,U,J,seed), allow_pickle=True))[0][0]
    terms = data['hJ']
       
    h_by_range = []
    J_by_range = []
    for flowstep in range(len(terms))[::10]:
        hJ = terms[flowstep]
        h,J = sort_by_range(L, hJ)
        
        h_by_range_step = {}
        for r in range(1,L+1):
            h_by_range_step[r] = np.nan_to_num(np.mean(h[r]))
        h_by_range.append(h_by_range_step)
        
        J_by_range_step = {}
        for r in range(2,L+1):
            J_by_range_step[r] = np.nan_to_num(np.mean(J[r]))
        J_by_range.append(J_by_range_step)
                
    return h_by_range, J_by_range

def load_and_average(output,L,h,U,J):
    numSeeds = 1000
    maxFlowSteps = 600
    
    # Create empty dicts
    all_h_for_range = np.ones((numSeeds, L, maxFlowSteps))*np.inf
    all_J_for_range = np.ones((numSeeds, L, maxFlowSteps))*np.inf

    # Fill in the data
    for seed in range(numSeeds):
        try:
            hr,Jr = load_data(output,L,h,U,J,seed)
            for r in range(1,L):
                all_h_for_range[seed,r][:len(hr)] = [hr[t][r] for t in range(len(hr))]
            
            for r in range(2,L):
                all_J_for_range[seed,r][:len(Jr)] = [Jr[t][r] for t in range(len(Jr))]
                
        except Exception as e:
            print(e)
            continue
            
    avg_h_for_range = ma.masked_invalid(all_h_for_range).mean(axis=0)
    avg_J_for_range = ma.masked_invalid(all_J_for_range).mean(axis=0)
    num_avgs_for_h = numSeeds-ma.masked_invalid(all_h_for_range).mask.sum(axis=0)
    num_avgs_for_J = numSeeds-ma.masked_invalid(all_J_for_range).mask.sum(axis=0)
    
    return avg_h_for_range, avg_J_for_range

if __name__ == "__main__":
    output = sys.argv[1]
    L = int(sys.argv[2])
    os.makedirs("{0}/L-{1}/".format(output,L), exist_ok=True)

    hscale = float(sys.argv[3])
    Jscale = float(sys.argv[4])
    Uscale = float(sys.argv[5])

    avg_h_vs_range, avg_J_vs_range = load_and_average(output, L, hscale, Uscale, Jscale)
    data = {'avg_h_vs_range':avg_h_vs_range, 'avg_J_vs_range':avg_J_vs_range}
    np.save("{0}/L-{1}/averaged-hJ-L-{1}-h-{2}-U-{3}-J-{4}.npy".format(output,L,hscale,Uscale,Jscale), data, allow_pickle=True)

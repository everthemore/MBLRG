import sys
import numpy as np
import numpy.ma as ma

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

def load_data(path,L,h,U,J,seed):
    data = np.atleast_2d(np.load("%s/hJ-L-%d-h-%.1f-U-%.1f-J-%.1f-seed-%d.npy"%(path,L,h,U,J,seed), allow_pickle=True))[0][0]
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

def load_and_average(path, L,h,U,J):
    numSeeds = 1000
    maxFlowSteps = 1000

    # Create empty dicts
    all_h_for_range = np.ones((numSeeds, L, maxFlowSteps))*np.inf
    all_J_for_range = np.ones((numSeeds, L, maxFlowSteps))*np.inf

    # Fill in the data
    for seed in range(numSeeds):
        try:
            hr,Jr = load_data(path, L,h,U,J,seed)

            for r in range(1,L):
                all_h_for_range[seed,r][:len(hr)] = [hr[t][r] for t in range(len(hr))]

            for r in range(2,L):
                all_J_for_range[seed,r][:len(Jr)] = [Jr[t][r] for t in range(len(Jr))]

        except Exception as e:
            print(e)
            continue

    avg_h_for_range = ma.masked_invalid(all_h_for_range).mean(axis=0)
    avg_J_for_range = ma.masked_invalid(all_J_for_range).mean(axis=0)

    return avg_h_for_range, avg_J_for_range

if __name__ == "__main__":
    L = int(sys.argv[1])
    path = sys.argv[2]
    avg_h_vs_range_deloc, avg_J_vs_range_deloc = load_and_average(L, 1, 3, 4)
    avg_h_vs_range_MBL, avg_J_vs_range_MBL = load_and_average(L, 4, 1, 1)

    data = {'avg_h_vs_range':avg_h_vs_range_deloc, 'avg_J_vs_range':avg_J_vs_range_deloc}
    np.save("data/avg-hJ-L-{0}-h-{1}-U-{2}-J-{3}.npy".format(L,1,3,4), data, allow_pickle=True)
    data = {'avg_h_vs_range':avg_h_vs_range_MBL, 'avg_J_vs_range':avg_J_vs_range_MBL}
    np.save("data/avg-hJ-L-{0}-h-{1}-U-{2}-J-{3}.npy".format(L,4,1,1), data, allow_pickle=True)

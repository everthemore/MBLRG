import numpy as np
from fermions import operator
import time

def make_term_from_indices(L, cdagsites, csites, nsites):
    # Sanity checks
    overlap = [x for x in cdagsites if x in csites]
    if len(overlap) > 0:
          print("Invalid cdagsites and csites have overlap (should be in nsites!)")
          return

    opstring = ["0"] * L

    for c in nsites:
      opstring[c] = "3"; #ns
    for c in cdagsites:
      opstring[c] = "1"; #cdags
    for c in csites:
      opstring[c] = "2"; #cs
    opstring = "".join(opstring)

    return opstring

class MBLHamiltonian:
    def __init__(self, L, hscale, Jscale, Uscale):

        # Store system size
        self.L = L

        # Store scales
        self.hscale = hscale
        self.Jscale = Jscale
        self.Uscale = Uscale

        self.verbose = False

        # All of the operators
        self.H = operator({})

        # Add random on-site terms
        self.on_site_coeff = []
        for i in range(self.L):
            coeff = np.random.uniform(-hscale, hscale)
            self.on_site_coeff.append(coeff)
            self.H.terms[make_term_from_indices(L, [], [], [i])] = coeff

        # Add NN interaction terms
        self.nn_coeff = []
        for i in range(self.L-1):
            coeff = np.random.uniform(-Uscale, Uscale)
            self.nn_coeff.append(coeff)
            self.H.terms[make_term_from_indices(L, [], [], [i,i+1])] = coeff

        # Add NN hopping
        self.hopping_coeff = []
        for i in range(self.L-1):
            coeff = np.random.uniform(-Jscale, Jscale)
            self.hopping_coeff.append(coeff)
            self.H.terms[make_term_from_indices(L, [i], [i+1], [])] = coeff
            self.H.terms[make_term_from_indices(L, [i+1], [i], [])] = coeff

        #-----------------
        # We want to group similar offdiagonals, so that we can base our choice
        # of which to rotate off of that
        #-----------------
        self.group_terms()

        #-----------------
        # Construct the transformation matrix that takes the operator basis
        # (which has a 1 whenever there is a density op on a site) to the
        # state-basis (computational basis, with a 1 if there is a particle
        # on a site)
        #-----------------
        mSingle = np.array([[1,0],[1,1]]) # Transformation for a single site
        self.m = mSingle
        for i in range(self.L-1):
            self.m = np.kron(self.m,mSingle)
        # Also store the inverse
        self.minv = np.linalg.inv(self.m)

    def set_from_lists(self, onsite, nn, hop):

        self.on_site_coeff = onsite
        self.nn_coeff = nn
        self.hopping_coeff = hop

        # All of the operators
        self.H = operator([])

        # Add random on-site terms
        for i in range(self.L):
            self.H = self.H + onsite[i] * operator([make_term_from_indices(self.L, [], [], [i])])

        # Add NN interaction terms

        for i in range(self.L-1):
            self.H = self.H + nn[i] * operator([make_term_from_indices(self.L, [], [], [i,i+1])])

        # Add NN hopping
        for i in range(self.L-1):
            coeff = hop[i]
            self.H = self.H + coeff * operator([make_term_from_indices(self.L, [i], [i+1], [])])
            self.H = self.H + coeff * operator([make_term_from_indices(self.L, [i+1], [i], [])])

        #-----------------
        # We want to group similar offdiagonals, so that we can base our choice
        # of which to rotate off of that
        #-----------------
        self.group_terms()

    def group_terms(self):
        '''
        Populates the self.diagonals and self.offdiagonals dictionaries
        '''
        self.diagonals = {}
        self.offdiagonals = {}

        # Now we extract a list of all operators from the Hamiltonian
        for term in self.H.terms:
            if( "1" in term or "2" in term ):
                self.offdiagonals[term] = self.H.terms[term]
            else:
                self.diagonals[term] = self.H.terms[term]

    def convertToStateBasis(self, op):
        operator_basis = np.zeros( 2**(self.L) )

        for term in op.terms:
            binary_op_str = "".join(["1" if c == "3" else "0" for c in term])
            integer_op = int(binary_op_str, 2)
            operator_basis[integer_op] = op.terms[term]

        # Convert to state basis
        state_basis = np.dot(self.m, operator_basis)
        return state_basis

    def convertToOperator(self, state_basis):
         # Convert to operator basis
        operator_basis = np.dot(self.minv,state_basis)

        # Extract operators
        inverse_operator = operator({})
        for i in range(len(operator_basis)):
            if operator_basis[i] != 0:
                i_to_string_w_3s = format(i,'0%db'%self.L)
                i_to_string_w_3s = i_to_string_w_3s.replace('1','3')
                inverse_operator.terms[i_to_string_w_3s] = inverse_operator.terms.get(i_to_string_w_3s,0) + operator_basis[i]

        return inverse_operator

    def pick_term_to_rotate_out(self):
        return max(self.offdiagonals, key=lambda key: self.offdiagonals[key])

    def same_offdiag(self, term1, offdiag):
        offdiag1 = [c if (c == "1" or c == "2") else "0" for c in term1]
        return offdiag1 == offdiag

    def rotateOut(self, threshold=1e-8):

        def diag(term):
            return "".join([c if c == "3" else "0" for c in term])

        # See if there is anything to rotate out; return True if we're diagonal
        if( len(self.offdiagonals) == 0 ):
            return True

        # Pick the term we are going to rotate out
        term = self.pick_term_to_rotate_out()
        term = [c if (c == "1" or c == "2") else "0" for c in term]

        # Gather all the diagonal parts for this term
        diagt = operator({diag(t):self.H.terms[t] for t in self.H.terms if self.same_offdiag(t,term)})

        # Get the offdiagonal part, but the - version
        A = operator({"".join(term):1})
        A = A - A.conj()

        # Construct unitary
        diags = operator(self.diagonals)
        deltaEterms = (A*diags - diags*A) #.cleanup()

        # Extract only the diagonal parts of all the operators
        deltaV = operator({})
        for d in deltaEterms.terms:
            diagd = "".join(["3" if c == "3" else "0" for c in d])
            if( "1" in d ) or ("2" in d):
                deltaV.terms[diagd] = deltaV.terms.get(diagd,0) + deltaEterms.terms[d]/2
            else: # /2 because?
                deltaV.terms[diagd] = deltaV.terms.get(diagd,0) + deltaEterms.terms[d]

        delta_state_basis = self.convertToStateBasis(deltaV)
        t_state_basis = self.convertToStateBasis(diagt)

        r = []
        for i in range(len(delta_state_basis)):
            if( delta_state_basis[i] == 0 ):
                r.append(np.pi/4)
            else:
                r.append( np.arctan(2*t_state_basis[i]/delta_state_basis[i])/2 )
        r = np.array(r)

        sin_rotator = self.convertToOperator( np.sin(r) )
        cos_rotator = self.convertToOperator( 1 - np.cos(r) )
        sinrotA = sin_rotator*A
        cosrotAA = cos_rotator*A*A

        Sm = operator({"0"*self.L:1}) - sinrotA + cosrotAA
        Sp = operator({"0"*self.L:1}) + sinrotA + cosrotAA

        # Rotate out
        newH = Sm * self.H * Sp

        # Update the Hamiltonian
        self.H = newH.cleanup(threshold=threshold)

        # Regroup
        self.group_terms()

        isDiag = self.H.isDiagonal()
        end_time = time.time()
        return isDiag

    def getCoefficientDistributions(self, mean = False):

        def get_range(term):
            tmpterm = "".join(['1' if c != '0' else '0' for c in term])

            if(tmpterm == "0"*self.L):
                return 0

            start = tmpterm.find('1')
            tmpterm = tmpterm[::-1]
            end = len(term) - tmpterm.find('1')
            return end - start

        # Zero out the dictionaries
        h = {}
        for r in range(0,self.L+1):
          h[r] = []

        J = {}
        for r in range(2,self.L+1):
          J[r] = []

        # This list only contains each conjugate once
        for term in self.H.terms:
            r = get_range(term)

            # Track diagonals
            if ("1" in term) or ("2") in term:
                J[r].append(np.abs(self.H.terms[term]))
            else:
                h[r].append(np.abs(self.H.terms[term]))

        if not mean:
            return h, J

        # Convert to means
        ha = []
        for r in range(self.L+1):
            if( len(h[r]) == 0 ):
              ha.append(0)
            else:
              ha.append(np.mean(h[r]))

        Ja = []
        for r in range(self.L+1):
            if( len(J[r]) == 0 ):
              Ja.append(0)
            else:
              Ja.append(np.mean(J[r]))

        return ha, Ja

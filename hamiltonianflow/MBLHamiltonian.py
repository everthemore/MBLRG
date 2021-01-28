import sys
import numpy as np
from fermions import *
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

    return opterm(1,opstring)

class hoppingOperator:
    def __init__(self, diag, offdiag):
        self.diagonal    = diag
        self.offdiagonal = offdiag

    def addToDiagonal(self, diagonal):
        self.diagonal = self.diagonal + diagonal

    def getFullOperator(self):
        return self.diagonal * self.offdiagonal

    def getOffdiagonalRepresentation(self):
        rep = sorted([self.offdiagonal.opterms[0].string, self.offdiagonal.opterms[1].string])
        return "+".join(rep)

    def __str__(self):
        return self.getFullOperator().__str__()

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
        self.H = operator([])

        # Add random on-site terms
        for i in range(self.L):
            coeff = np.random.uniform(-hscale, hscale)
            self.H = self.H + coeff * operator([make_term_from_indices(L, [], [], [i])])

        # Add NN interaction terms
        for i in range(self.L-1):
            coeff = np.random.uniform(-Uscale, Uscale)
            self.H = self.H + coeff * operator([make_term_from_indices(L, [], [], [i,i+1])])

        # Add NN hopping
        for i in range(self.L-1):
            coeff = np.random.uniform(-Jscale, Jscale)
            self.H = self.H + coeff * operator([make_term_from_indices(L, [i], [i+1], [])])
            self.H = self.H + coeff * operator([make_term_from_indices(L, [i+1], [i], [])])

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

    def group_terms(self):
        '''
        Populates the self.diagonals and self.offdiagonals dictionaries
        '''
        offdiagonalindices = {}
        diagonal = operator([])
        offdiagonal_list = []

        # Now we extract a list of all operators from the Hamiltonian
        already_seen_operators = []
        for term in self.H.opterms:

            # Check if we already considered the conjugate of this term
            if( term.conj().string in already_seen_operators ):
                continue

            # If we get here, we add it to the list
            already_seen_operators.append(term.string)

            # If the term is diagonal, add it to the diagonal list
            if( term.isDiagonal() ):
                diagonal = diagonal + operator([term])
                continue

            # Must be off-diagonal, so build a hoppingOperator
            # Extract the diagonal part
            newop_diag = term.coeff*operator([term.getDiagonal()])
            # Extract the offdiagonal part
            newop_offdiag = operator([term.getOffDiagonal()]) + operator([term.getOffDiagonal()]).conj()
            # Build the hoppingOperator
            ho = hoppingOperator(newop_diag, newop_offdiag)

            # See if we already have this
            strrep = ho.getOffdiagonalRepresentation()
            index = offdiagonalindices.get( strrep, -1 )

            if( index == -1 ):
                offdiagonalindices[ strrep ] = len(offdiagonal_list)
                offdiagonal_list.append( ho )
            else:
                offdiagonal_list[ index ].addToDiagonal( newop_diag )

        # Store diagonals and offdiagonals
        self.diagonals = diagonal
        self.offdiagonals = offdiagonal_list

    def convertToStateBasis(self, op):
        operator_basis = np.zeros( 2**(self.L) )

        for term in op.opterms:
            binary_op_str = term.diagonal_str.replace('3', '1')
            integer_op = int(binary_op_str, 2)
            operator_basis[integer_op] = term.coeff

        # Convert to state basis
        state_basis = np.dot(self.m,operator_basis)
        return state_basis

    def convertToOperator(self, state_basis):
         # Convert to operator basis
        operator_basis = np.dot(self.minv,state_basis)

        if self.verbose:
            print("Convert to op: ")
            print(operator_basis)

        # Extract operators
        inverse_operator = operator([])
        for i in range(len(operator_basis)):
            if operator_basis[i] != 0:
                i_to_string_w_3s = format(i,'0%db'%self.L)
                i_to_string_w_3s = i_to_string_w_3s.replace('1','3')

                if self.verbose:
                    print("Nonzero: " + i_to_string_w_3s)
                    print("With val: ", operator_basis[i])
                newopterm = opterm(operator_basis[i], i_to_string_w_3s)
                inverse_operator = inverse_operator + operator([newopterm])

        return inverse_operator

    def computeInverse(self, op):
        # TODO: This can be cached!

        # In the state basis, we can just invert the coefficients
        state_basis = self.convertToStateBasis(op)
        # Invert
        state_basis = 1/state_basis
        # Convert back
        return self.convertToOperator(state_basis)

    def rotateOut(self, method=0):

        #print("Rotating Out")
        # See if there is anything to rotate out; return True if we're diagonal
        if( len(self.offdiagonals) == 0 ):
            return True

        # Compute a list of all the coeffs of the offdiagonals
        #t0 = time.time()
        amplitudes = []
        if( method == 0 ):
            for o in self.offdiagonals:
                tmp = 0
                for i in o.diagonal.opterms:
                    tmp += np.abs(i.coeff)**2
                amplitudes.append(tmp)

        # Use L2 norm
        if( method == 1 ):
            for o in self.offdiagonals: # c1^dagger c_2 ( J1 + J2 * n3)

                # Construct a list of all the abs^2 of coeffs
                altnorm = []
                for i in o.diagonal.opterms: #( J1, J2 n3, ... )
                    altnorm.append(np.abs(i.coeff)**2)

                # Take the max of this list
                max_for_this_offdiagonal = np.max(altnorm)
                amplitudes.append(max_for_this_offdiagonal)

        #t1 = time.time()
        #total = t1-t0
        #print("\t List of off-diagonals: %.3f"%total)

        # Pick the max
        index = np.argmax(amplitudes)
        op = self.offdiagonals[index]

        # Get the diagonal part
        diagt = op.diagonal

        #t0 = time.time()
        # Get the offdiagonal part, but the - version
        A = operator([op.offdiagonal.opterms[0]]) - operator([op.offdiagonal.opterms[1]])

        #t1 = time.time()
        #total = t1-t0
        #print("\t Computing A: %.3f"%total)

        if self.verbose:
            print("Rotating out with:")
            print(A)
            print("Which has diagonal prefactor: ", diagt)

        #t0 = time.time()
        # Construct unitary
        deltaEterms = (A*self.diagonals - self.diagonals*A) #.cleanup()

        #t1 = time.time()
        #total = t1-t0
        #print("\t Computing DeltaE: %.3f"%total)

        if self.verbose:
            print("Commutator with diagonals: ")
            print(deltaEterms)

        #deltaEterms = [deltaEterms.opterms[i] for i in range(0,len(deltaEterms.opterms),2)]

        #t0 = time.time()
        # Extract only the diagonal parts of all the operators
        deltaV = operator([])
        for d in deltaEterms.opterms:
            if( d.isDiagonal() ):
                deltaV = deltaV + operator([opterm(d.coeff, d.diagonal_str)])
            else: # /2 because?
                deltaV = deltaV + operator([opterm(d.coeff/2, d.diagonal_str)])

        #t1 = time.time()
        #total = t1-t0
        #print("\t Computing DeltaV: %.3f"%total)

        if self.verbose:
            print("Delta V is: ")
            print(deltaV)

        #t0 = time.time()
        delta_state_basis = self.convertToStateBasis(deltaV)
        t_state_basis = self.convertToStateBasis(diagt)

        #t1 = time.time()
        #total = t1-t0
        #print("\t Converting to state bases: %.3f"%total)

#        if( not delta_state_basis.any() ):
#            r = np.ones_like(delta_state_basis)*np.pi/4
#        else:
#            r = np.arctan(2*t_state_basis/delta_state_basis)/2

        r = []
        for i in range(len(delta_state_basis)):
            if( delta_state_basis[i] == 0 ):
                r.append(np.pi/4)
            else:
                r.append( np.arctan(2*t_state_basis[i]/delta_state_basis[i])/2 )
        r = np.array(r)

        if self.verbose:
            print("vd: ", delta_state_basis)
            print("vt: ", t_state_basis)
            print("vq: ", r)

        #t0 = time.time()
        sin_rotator = self.convertToOperator( np.sin(r) )
        cos_rotator = self.convertToOperator( 1 - np.cos(r) )

        if self.verbose:
            print("Sin rot")
            print(np.sin(r))
            print(self.convertToOperator(np.sin(r)))
            print(sin_rotator)
            print("Cos rot")
            print(cos_rotator)
            print(A)
            print(A*A)

        sinrotA = sin_rotator*A
        cosrotAA = cos_rotator*A*A

        Sm = operator([opterm(1,"0"*A.length)]) - sinrotA + cosrotAA
        Sp = operator([opterm(1,"0"*A.length)]) + sinrotA + cosrotAA

        #t1 = time.time()
        #total = t1-t0
        #print("\t Computing rotators and Sm and Sp: %.3f"%total)

        if self.verbose:
            print("transformation")
            print(Sm)

        # Rotate out
        newH = Sm * self.H * Sp

        #t0 = time.time()
        # Update the Hamiltonian
        self.H = newH.cleanup(threshold=1e-5)

        #t1 = time.time()
        #total = t1-t0
        #print("\t Cleaning up: %.3f"%total)

        if self.verbose:
            print("regrouping")

        #t0 = time.time()
        # Regroup
        self.group_terms()

        #t1 = time.time()
        #total = t1-t0
        #print("\t Regrouping: %.3f"%total)

        return self.H.isDiagonal(), amplitudes[index]

    def invert(self, op, index):
        '''
        Recursive function that inverts an operator
        '''

        # If the operator is empty, nothing to invert
        if( len(op.opterms) == 0):
            return None

        # If we get to the identity, we're done
        if len(op.opterms) == 1:
            if( op.opterms[0].diagonal_str == "0"*self.L ):
        #        print("  "*index + "Is identity, so we will return 1/coeff: %.3f"%(1/op.opterms[0].coeff))
                return operator([opterm(1/op.opterms[0].coeff,"0"*self.L)])

        # We loop over all possible local density operators
        # Construct density on this site
        opstring = ["0"] * self.L
        opstring[index] = "3";
        opstring = "".join(opstring)
        densop = opterm(1,opstring)
        #print("  "*index + "Checking for density term: ")
        #print("  "*index + densop.__str__())
        densop = operator([densop])

        # We haven't yet expaned this term
        expanded = False

        # now for each term in the operator, we see if the site bit is set.

        # Setting that density to 1 is the same as:
        # If it is, we replace it by a 0, and we just keep it
        # Setting that density to 0 is the same as:
        # If it is, we remove the term, and otherwise we just keep it
        newop1 = operator([])
        newop2 = operator([])
        encountered = False
        for term in op.opterms:

            if term.diagonal_str[index] == '3':
            #    print("  "*index + "This term has that density")
                # So we'll split off two operators
                # One in which we replace this density by an identity
                newopstring = term.diagonal_str[:index] + "0" + term.diagonal_str[index+1:]
                newop1 = newop1 + operator([opterm(term.coeff,newopstring)])
                # And one in which we set it to zero, so it disappears
                encountered = True
            else:
            #    print("  "*index + "This term does not have that density")
                # So we'll keep the term as-is
                newop1 = newop1 + operator([term])
                newop2 = newop2 + operator([term])


        if( not encountered ):
            # Skip this and go to the next index
            print("%d"%index + "  "*index + op.__str__())
            return self.invert(op, index+1)
        else:
            #print("%d"%index + "  "*index + "Current operators")
            #print("%d"%index + "  "*index + newop1.__str__())
            #print("%d"%index + "  "*index + newop2.__str__())
            newterm1 = self.invert(newop1, index+1)
            newterm2 = self.invert(newop2, index+1)

            result = newterm1*densop

            #if( len(newterm2.opterms) != 0):
            if( newterm2 != None ):
                result = result + newterm2*(operator([opterm(1,"0"*self.L)])-densop)

            print("%d"%index + "  "*index + result.__str__())
            return result

    def getCoefficientDistributions(self, mean = False):
        # Zero out the dictionaries
        h = {}
        for r in range(1,self.L+1):
          J[r] = []

        J = {}
        for r in range(2,self.L+1):
          J[r] = []

        # This list only contains each conjugate once
        for term in self.H.opterms:
            r = term.range

            # Track diagonals
            if term.isDiagonal():
                h[r].append(np.abs(term.coeff))
            else:
                # Otherwise, add to the offdiags
                J[r].append(np.abs(term.coeff))

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

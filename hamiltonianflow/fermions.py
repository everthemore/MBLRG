import numpy as np
from itertools import product
from functools import reduce
import time

class operator:
    def __init__(self, opterms):
        self.opterms = []
        for term in opterms:
            self.opterms.append(opterm(term.coeff, term.string))

    def contains(self, term):
        for term1 in self.opterms:
            if term.string == term1.string:
                return self.opterms.index(term1)
        return -1

    def conj(self):
        newOperator = operator([])

        for term in self.opterms:
            conjTerm = term.conj()
            newOperator = newOperator + \
                operator([opterm(conjTerm.coeff, conjTerm.string)])

        return newOperator

    @property
    def length(self):
        if(len(self.opterms) == 0):
            return 0
        return len(self.opterms[0].string)

    def expand(self, term):

        if( term == None ):
            return

        # Recursive call
        if "5" not in term.string:
            return operator([term])

        newOperator = operator([])
        string = term.string
        for i in range(len(term.string)):
            if string[i] == "5":
                newCoeff1 = 1
                newString1 = string[:i] + "0" + string[i + 1:]
                newCoeff2 = -1
                newString2 = string[:i] + "3" + string[i + 1:]

                newOperator = newOperator + \
                    self.expand(opterm(newCoeff1 * term.coeff, newString1))
                newOperator = newOperator + \
                    self.expand(opterm(newCoeff2 * term.coeff, newString2))

                # We recursively expand every 5, so once we've found one, we can leave here
                break

        return newOperator

    def cleanup(self, threshold=1e-8):
        # Cleanup
        keepops = [x for x in self.opterms if np.abs(x.coeff) >= threshold]
        this = operator(keepops)
        return this

    def isDiagonal(self):
        for term in self.opterms:
            if not term.isDiagonal():
                return False
        return True

    def toMatrix(self):
        '''
        Return this operator as a dense matrix
        '''
        matrix = 0
        for term in self.opterms:
            matrix += term.coeff * term.toMatrix()
        return matrix

    def __add__(self, other):
        # Add two operators
        print("\t Adding")
        s = time.time()
        newOperator = operator(self.opterms)
        print("\t Copying myself took ", time.time() - s)

        s=time.time()
        for term1 in other.opterms:
            index = self.contains(term1)
            if index != -1:
                newOperator.opterms[index].coeff += term1.coeff
            else:
                newOperator.opterms.append(term1)
        print("\t The rest took ", time.time() - s)

        return newOperator #.cleanup()

    def __sub__(self, other):
       # Subtract two operators

        newOperator = operator(self.opterms)
        for term1 in other.opterms:
            index = self.contains(term1)
            if index != -1:
                newOperator.opterms[index].coeff -= term1.coeff
            else:
                newOperator.opterms.append(
                    opterm(term1.coeff * -1, term1.string))

        return newOperator #.cleanup()

    def multiply_single(self, terms):
        result = terms[0] * terms[1]
        return self.expand(result)

    def __mul__(self, other):
        newOperator = operator([])

        if(type(other) in (int, float, complex)):
            for term1 in self.opterms:
                newOperator = newOperator + \
                    operator([opterm(term1.coeff * other, term1.string)])

        else:
            s = time.time()
            t = product(self.opterms, other.opterms)
            print("Making list took ", time.time() - s)

            s = time.time()
            t2 = filter(None,map(self.multiply_single, t))
            print("Multiplying and filtering list took ", time.time() - s)

            s = time.time()
            t3 = list(t2)
            print("Listing list took ", time.time() - s)

            if t3:
                return np.sum(t3).cleanup()
            else: return operator([])

            allterms = list(filter(None,map(self.multiply_single, product(self.opterms, other.opterms))))
            if allterms:
                return np.sum(allterms).cleanup()
            else:
                return operator([])

            # #map(multiply_terms, self.opterms, other.opterms)
            # for term1 in self.opterms:
            #     # Multiply each other term in other to the current
            #     for term2 in other.opterms:
            #         # Multiply, and see if we need to add anything
            #         multiplied, dead = term1 * term2
            #
            #         if not dead:
            #             # Could be a list or a term
            #             multiplied = self.expand(multiplied)
            #             newOperator = newOperator + multiplied

        return newOperator.cleanup()

    def __rmul__(self, other):
        newOperator = operator([])

        if(type(other) in (int, float, complex)):
            for term1 in self.opterms:
                newOperator = newOperator + \
                    operator([opterm(term1.coeff * other, term1.string)])
            return newOperator.cleanup()

        else:  # We're multiplying other * operator
            # Always perform multiplication with left term first
            return other.__mul__(self)

    def __str__(self):
        if(len(self.opterms) == 0):
            return "zero"

        string = ""
        for i, term in enumerate(self.opterms):
            string += "Term {0}: {1} {2}\n".format(i, term.coeff, term.string)
        return string


class opterm:
    m = [ np.array([[1,0],[0,1]]), np.array([[0,1],[0,0]]), np.array([[0,0],[1,0]]), np.array([[0,0],[0,1]]) ]

    def __init__(self, coeff, string):
        self.coeff = coeff
        self.string = string

    def conj(self):
        '''
            Returns the complex conjugate of this term
        '''
        newString = ["-1"] * len(self.string)
        for i, c in enumerate(self.string):
            if c == "1":
                newString[i] = "2"
            elif c == "2":
                newString[i] = "1"
            else:
                newString[i] = c

        return opterm(np.conjugate(self.coeff), "".join(newString))

    def isDiagonal(self):
        if(("1" not in self.string) and ("2" not in self.string)):
            return True
        return False

    def getDiagonal(self):
        string = self.diagonal_str
        return opterm(1,string)
    def getOffDiagonal(self):
        string = self.offdiagonal_str
        return opterm(1,string)

    @property
    def diagonal_str(self):
        newstring = ""
        for c in self.string:
            if c == "1" or c == "2":
                newstring += "0"
            else:
                newstring += c
        return newstring

    @property
    def offdiagonal_str(self):
        newstring = ""
        for c in self.string:
            if c == "0" or c == "3":
                newstring += "0"
            else:
                newstring += c

        return newstring

    @property
    def range(self):
        # Catch identity operator
        if self.string == "0" * len(self.string):
            return 0

        start = 0
        end = len(self.string)
        for i in range(start, end):
            if self.string[i] != "0":
                start = i
                break

        for i in range(len(self.string) - 1, -1, -1):
            if self.string[i] != "0":
                end = i
                break

        if (start == end):
            return 1

        return end - start + 1

    def toMatrix(self):
        '''
        Return this operator as a dense matrix
        '''
        if( len(self.string) < 2 and len(self.string) != 0 ):
            return opterm.m[int(self.string[0])]

        matrix = np.kron( opterm.m[int(self.string[0])], opterm.m[int(self.string[1])] )
        for c in self.string[2:]:
            matrix = np.kron(matrix, opterm.m[int(c)])
        return matrix

    def multiply_single(self, c1, c2):
        newString = None

        # Take care of the identity
        if c1 == "0":
            newString = c2
        elif c2 == "0":
            newString = c1

        # If the first is a cdag
        elif c1 == "1":
            # If c2 is also a cdag
            if c2 == "1":
                return None

            # If c2 is a c
            if c2 == "2":
                newString = "3"  # cdag*c == n

            if c2 == "3":  # cdag*n = cdag cdag -> dead
                return None

        # If the first is a c
        elif c1 == "2":
            # If c2 is a dagger
            if c2 == "1":
                newString = "5"  # Needs to be expanded into 1 - density

            if c2 == "2":
                return None

            if c2 == "3":
                newString = "2"  # c*n = c

        # If the first is a density
        elif c1 == "3":
            # If c2 is a cdag
            if c2 == "1":
                newString = "1"  # Equiv to just having a cdag

            if c2 == "2":
                return None

            if c2 == "3":
                newString = "3"  # density**2 = density

        return newString

    def __mul__(self, other):
        newString = list( map(self.multiply_single, self.string, other.string) )
        if None in newString:
            return None
        return opterm(self.coeff * other.coeff, "".join(newString))

        if None in newString:
            return "", True
        return opterm(self.coeff * other.coeff, "".join(newString)), False

    def __str__(self):
        return "Term 0: {0} {1}".format(self.coeff, self.string)

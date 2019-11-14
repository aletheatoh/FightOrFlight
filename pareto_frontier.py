import numpy as np

def simple_cull(inputPoints):
    paretoPoints = []
    candidateRowNr = 0
    dominatedPoints = set()

    while len(inputPoints) > 0:
        candidateRow = inputPoints.pop()
        rowNr = 0
        dominated = False
        while len(inputPoints) > 0 and rowNr < len(paretoPoints):
            currRow = paretoPoints[rowNr]
            comp = dominates(currRow, candidateRow)

            if not comp: # candidateRow dominates currRow
                # inputPoints.remove(currRow)
                paretoPoints.remove(currRow)
                dominatedPoints.add(tuple(currRow))
                rowNr += 1
            elif comp: # currRow dominates candidateRow
                dominatedPoints.add(tuple(candidateRow))
                dominated = True
                break

        if not dominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.append(tuple(candidateRow))

    return paretoPoints, dominatedPoints

'''
Returns True if a dominates b, False otherwise
'''
def dominates(a, b):
    sum = 0
    if a[0] <= b[0]: sum += 1 # cost
    if a[1] <= b[1]: sum += 1 # duration
    if a[2] >= b[2]: sum += 1 # frequency

    return sum == 3

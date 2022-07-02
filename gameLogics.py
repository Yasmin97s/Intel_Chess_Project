from checkers.piece import *

# get the index of the cell
def convertPos(i,j):
    return i*8+j

# check if the is a legal cell in a 8X8 board
def isLegal(i, j):
    return i>=0 and i<8 and j >=0 and j<8

# get the position of the piece that were killed
def getKilled(i,j, nextMove_0, nextMove_1):
    killed_0 = i+1 if i < nextMove_0 else i-1
    killed_1 = j + 1 if j < nextMove_1 else j - 1
    return killed_0, killed_1

# get the changes that were made in the last move
def find_diff(prev, curr, currentPlayer, otherPlayer):
    directions1 = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    directions2 = [[2, 2], [2, -2], [-2, 2], [-2, -2]]
    killed = []
    for i in range(len(prev)):
        for j in range(len(prev[i])):
         if prev[i][j] != curr[i][j]: # there is difference
             if prev[i][j] == currentPlayer: # empty
                 for dir in directions1:
                     nextMove_0 = i + dir[0]
                     nextMove_1 = j + dir[1]
                     if isLegal(nextMove_0, nextMove_1): # in bounds of the board
                         if prev[nextMove_0][nextMove_1] == 0 and curr[nextMove_0][nextMove_1] == currentPlayer:
                             return [[i,j],[nextMove_0,nextMove_1],killed]
                 for dir in directions2:
                     nextMove_0 = i + dir[0]
                     nextMove_1 = j + dir[1]
                     if isLegal(nextMove_0, nextMove_1):  # in bounds of the board
                         if prev[nextMove_0][nextMove_1] == 0 and curr[nextMove_0][nextMove_1] == currentPlayer:
                                 killed_0, killed_1 = getKilled(i,j, nextMove_0, nextMove_1)
                                 if isLegal(killed_0, killed_1):  # in bounds of the board
                                    if prev[killed_0][killed_1] == otherPlayer and curr[killed_0][killed_1] == 0:
                                        killed = [killed_0, killed_1]
                                    return [[i, j], [nextMove_0, nextMove_1], killed]
    return None

# build a new game board due to the changes that were made
def build_new_board(currentBoard, changes, currentPlayer, otherPlayer):
    if changes:
        print("this is needed:",changes[0][0], " -- ",changes[0][1])
        piece = currentBoard.board[changes[0][0]][changes[0][1]]
        currentBoard.move(piece, changes[1][0], changes[1][1])
        if len(changes[2]) != 0:
            skip = [Piece(changes[2][0], changes[2][1], otherPlayer)]
            currentBoard.remove(skip)
    return currentBoard

# check if the two matrices are identical
def checkIdentical(m1, m2):
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            if m1[i][j] != m2[i][j]:
                return False
    return True

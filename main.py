# Assets: https://techwithtim.net/wp-content/uploads/2020/09/assets.zip
from checkers.game import Game
from minimax.algorithm import minimax
from copy import deepcopy
import logging
import datetime
from robotHandler import *

from boardVisualDetection import *
import sys

sys.path.append('..')
from positions import *
from uarm.wrapper import SwiftAPI
from uarm.utils.log import logger
logger.setLevel(logger.VERBOSE)

from gameLogics import *

FPS = 60
z = 0

if __name__ == '__main__':
        filename = "logs\\" + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + ".txt"
        logging.basicConfig(filename= filename, level=logging.DEBUG,
                        format="%(asctime)s %(message)s", filemode="w")
        logging.info("Let's begin checkers game")
        swift = SwiftAPI(port="COM9", callback_thread_pool_size=1)
        device_info = swift.get_device_info()
        print(device_info)
        #parking(swift)

        # when we start the robot will be in the exit position
        x, y, z = exit
        swift.set_position(x, y, z,speed=50000)

        playerName = input("please enter your name:\n")
        # at this stage we assume the human has already put his/her pieces
        again = False
        startState = [[0, 2, 0, 2, 0, 2, 0, 2]
            , [2, 0, 2, 0, 2, 0, 2, 0]
            , [0, 2, 0, 2, 0, 2, 0, 2]
            , [0, 0, 0, 0, 0, 0, 0, 0]
            , [0, 0, 0, 0, 0, 0, 0, 0]
            , [1, 0, 1, 0, 1, 0, 1, 0]
            , [0, 1, 0, 1, 0, 1, 0, 1]
            , [1, 0, 1, 0, 1, 0, 1, 0]]
        bd = Board_Detector()
        while not again:
            bd.findBoard()
            currMat = bd.drawSquares()
            for row in currMat:
                print(row)
            # if the the detected matrix is not identical to the static initial state the nwe need to try to detect again
            again = checkIdentical(currMat, startState)
        x = 0
        while int(x) != 1:
            x = input("please enter x:\n")

        run = True
        clock = pygame.time.Clock()
        game = Game()

        while run:
            if game.turn == WHITE: # robot move
                # take a pic for before state
                value, new_board = minimax(game.get_board(), 4, WHITE, game)
                mat = []
                rowList = []
                for row in range(ROWS):
                    for col in range(COLS):
                        if new_board.board[row][col] == 0:
                            rowList.append(0)
                        elif new_board.board[row][col].color == WHITE:
                            rowList.append(2)
                        elif new_board.board[row][col].color == RED:
                            rowList.append(1)
                        if len(rowList) == 8:
                            mat.append(rowList)
                            rowList = []
                # get the changes that should be done according to minimax
                changes = find_diff(currMat, mat, 2, 1)

                print("this is AI changes: ", changes)

                # tell the robot what changes should be done
                source = convertPos(changes[0][0],changes[0][1])
                dest = convertPos(changes[1][0], changes[1][1])
                move_and_pump(swift,coordinates[source],coordinates[dest])
                if len(changes[2]) > 0: # robot move killed
                    source = convertPos(changes[2][0], changes[2][1])
                    move_to_death(swift, coordinates[source], cemetry)
                x, y, z = exit
                swift.set_position(x, y, z, speed=50000)

                logging.info("Now is robot's robot:")
                logging.info("This is the state after the robot played:")
                for row in mat:
                    print(row)
                    logging.info("      {}".format(row))
                logging.info("this is AI/robot changes: {}".format(changes))
                currMat = mat
                game.ai_move(new_board)

            # check if there is a winner at this stage
            if game.winner() != None:
                if game.winner() == RED:
                    print(playerName," wins!")
                    logging.info("The winner is: {} !".format(playerName))
                else:
                    print("Robot wins!")
                    logging.info("The Robot is the winner!")
                run = False

            # at this stage the human will make a move and as he/she finishes he will tell us by entering 1
            f = 0
            while int(f) != 1:
                f = input("please enter 1 when you finish playing:\n")
            # take a pic for after state
            bd.findBoard()
            mat = bd.drawSquares()
            logging.info("Now is {}'s turn:".format(playerName))
            print("this is prev:\n")
            for row in currMat:
                print(row)
            print("this is mat:\n")
            logging.info("This is the state after the human played:")
            for row in mat:
                logging.info("      {}".format(row))
                print(row)
            # get the changes done by the human to the game board
            changes = find_diff(currMat, mat, 1, 2)
            print("this is {} changes: ", changes)
            logging.info("this is {}, changes: {}".format(playerName,changes))
            temp = deepcopy(game.boardObject)
            # build a new game board according to the changes and save it as the current game board
            new_board = build_new_board(temp, changes, 2, 1)
            game.ai_move(new_board)
            currMat = mat

        pygame.quit()









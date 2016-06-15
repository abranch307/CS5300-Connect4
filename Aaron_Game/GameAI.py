import numpy as np, random
import tensorflow as tf

class C4AI:
    def __init__(self, input_size, h1_size, output_size):
        'Define weights, biases, and create placeholder for yhat'
        self.reward = 0
        self.discReward = .5
        self.learnRate = .1
        self.yhat = tf.placeholder(tf.float32, [None, 1])
        self.w1 = tf.Variable(tf.truncated_normal([input_size, h1_size]))
        self.b1 = tf.Variable(tf.truncated_normal([1, h1_size]))
        self.w2 = tf.Variable(tf.truncated_normal([h1_size, output_size]))
        self.b2 = tf.Variable(tf.truncated_normal([1, output_size]))

    def sigma(self, x):
        return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.neg(x))))

    def forwardProp(self, input):
        z_1 = tf.add(tf.matmul(input, self.w1), self.b1)
        a_1 = self.sigma(z_1)
        z_2 = tf.add(tf.matmul(a_1, self.w2), self.b2)
        a_2 = self.sigma(z_2)
        return a_2

    '''
    Function getNextMoves:
        This function will take the current state of the Connect 4 board, and return an
        6x7 array of possible next moves for current player

    Parameters: self, CurrentState - a 6x7 array which holds 0 for empty, 1 for player 1, and 2 for player 2
    (note that this function doesn't care about player 1 or 2, only 0 spaces in columns)

    Returns: list of 6x7 arrays (up to 7) which represents all possible moves for current player
        with the column # for move in the last column (element 7, and all are same in tuple)
    '''
    def getNextMoves(self, CurrentState, PlayerNum):
        #Declare a next moves list object
        tempState = np.array(CurrentState)
        movesList = []
        found = False

        #Loop through columns in Connect 4 board
        for i in range(tempState.shape[0]):
            #Loop through every row within column (0 starts at top of column, so we need to loop backwards)
            for j in range(tempState.shape[1]-1, -1, -1):
                #If element is empty, put current player into move and add to next moves list
                if not tempState[i][j]:
                    #Add player to empty cell then add tempstate to movesList
                    tempState[i][j] = PlayerNum
                    movesList.append(np.append(tempState, [[i, i, i, i, i, i]], axis=0))

                    #Reset tempState to CurrentState for finding next valid move/state
                    tempState = np.array(CurrentState)

                    #Set found to true to exit inner loop
                    found = True

                #Exit inner loop if found
                if found:
                    found = False
                    break


        #Return list
        return movesList

    '''
        Function evaluateMoves:
        This function will choose the next best move.

        Parameters: NextMoves - list of entire game state of future next potential moves

        Returns: int - column for next move
    '''
    def evaluateMoves(self, CurrentState, NextMoves, CurPlayerWin, OpponentWin, BoardFull):
        #Convert CurrentState into input vector
        currentState = np.matrix(np.ravel(np.array(CurrentState)))
        currentState[np.where(currentState == 'black')] = -1
        currentState[np.where(currentState == 'red')] = 1
        currentState[np.equal(currentState, None)] = 0

        #Separate move states from their columns and convert NextMoves into input vectors
        nextStates = []
        Columns = []
        for st in NextMoves:
            st[np.where(st == 'black')] = -1
            st[np.where(st == 'red')] = 1
            st[np.equal(st, None)] = 0
            nextStates.append(np.ravel(st[:7]))
            Columns.append(st[7])

        nextStates = np.array(nextStates)
        Columns = np.array(Columns)
        cStateInput = tf.constant(currentState) #Doesn't work currently (trying to convert ndarray to Tensorflow variable
        nStatesInput = tf.constant(nextStates) #Doesn't work currently (trying to convert ndarray to Tensorflow variable

        #Perform forward propagation on current state to get reward value
        preR = self.forwardProp(cStateInput)

        #Perform forward propagation on next states to see which has higher outcome
        futR = self.forwardProp(nStatesInput)

        #Choose future based on policy (max, or random)
        chosenIndex = np.argmax(futR)
        move = Columns(chosenIndex)

        #Choose are random move from states
        chosenIndex = random.randint(0, (Columns.shape[0] - 1))
        move = Columns(chosenIndex)

        #Calculate current reward
        if CurPlayerWin:
            reward = 50
        elif OpponentWin:
            reward = -50
        elif BoardFull:
            reward = 0.5
        else:
            reward = 0.5

        #Calculate error cost
        cost = (self.discReward * self.forwardProp(nextStates[chosenIndex])) - preR

        #Perform backpropagation based on error
        step = tf.train.GradientDescentOptimizer(self.learnRate).minimize(cost)

        #Return next move
        return move[0]

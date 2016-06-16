import numpy as np, random
import tensorflow as tf

class C4AI:
    def __init__(self, input_size, h1_size, output_size):
        'Define weights, biases, and create placeholder for yhat'
        self.reward = 0
        self.discReward = .5
        self.learnRate = .1
        self.costs = []
        #self.yhat = tf.placeholder(tf.float32, [None, 1])
        self.w1 = tf.constant(np.random.rand(input_size, h1_size), np.float32)
        self.b1 = tf.constant(np.ones((1, h1_size), np.float32))
        self.w2 = tf.constant(np.random.rand(h1_size, output_size), np.float32)
        self.b2 = tf.constant(np.ones((1, output_size), np.float32))
        #self.w1 = tf.constant(tf.truncated_normal([input_size, h1_size]), name='W1')
        #self.b1 = tf.constant(tf.truncated_normal([1, h1_size]), name='b1')
        #self.w2 = tf.constant(tf.truncated_normal([h1_size, output_size]), name='W2')
        #self.b2 = tf.constant(tf.truncated_normal([1, output_size]), name='b2')

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
        currentState = np.array(currentState, np.float32)

        #Separate move states from their columns and convert NextMoves into input vectors
        nextStates = []
        Columns = []
        for st in NextMoves:
            st[np.where(st == 'black')] = -1
            st[np.where(st == 'red')] = 1
            st[np.equal(st, None)] = 0
            nextStates.append(np.ravel(st[:7]))
            Columns.append(st[7])

        nextStates = np.array(nextStates, np.float32)
        Columns = np.array(Columns)

        with tf.Session() as session:
            #Convert ndarrays into tensorflow constants
            cStateInput = tf.constant(currentState, name='cStateInput')
            nStatesInput = tf.constant(nextStates, name='nStatesInput')

            #Perform forward propagation on current state to get reward value
            preR = tf.Variable(self.forwardProp(cStateInput), name='previousReward')

            #Perform forward propagation on next states to see which has higher outcome
            futR = tf.Variable(self.forwardProp(nStatesInput), name='futureReward')

            # Initialize all variables
            model = tf.initialize_all_variables()
            session.run(model)

            #Find current state's reward (for sake of curiousity, not really needed to be done here)
            pr = session.run(preR)
            print('Reward for previous state:')
            print(pr)
            print('\n')

            #Find rewards for all next states
            fr = session.run(futR)
            fr = np.array(fr, np.float32)
            print('Supposed future rewards are: ')
            print(fr)
            print('\n')

            '''
                I need to base the below move selection on a policy
            '''
            #Choose future based on policy (max, or random)
            chosenIndex = np.argmax(fr)
            move = Columns[chosenIndex]

            #Choose are random move from states
            chosenIndex = random.randint(0, (Columns.shape[0] - 1))
            move = Columns[chosenIndex]

            #Calculate current reward
            if CurPlayerWin:
                self.reward = 50
            elif OpponentWin:
                self.reward = -50
            elif BoardFull:
                self.reward = 0.5
            else:
                self.reward = 0.5

            #Define model for our error equation
            chosenNextState = tf.constant(np.matrix(nextStates[chosenIndex]), np.float32)
            error = tf.Variable(self.reward + (self.discReward * self.forwardProp(chosenNextState)) - self.forwardProp(cStateInput))
            errorsq = tf.square(error)

            #Calculate error cost (for curiosity purposes, not really needed here...)
            cost = self.reward + ((self.discReward * fr[chosenIndex]) - pr)
            print('Final cost is:')
            print(cost)
            print('\n')

            # Define backpropagation calculation
            train_op = tf.train.GradientDescentOptimizer(self.learnRate).minimize(errorsq)

            # Initialize all variables
            model = tf.initialize_all_variables()
            session.run(model)

            #Perform backpropagration
            self.costs.append(session.run(train_op))
            print(self.costs)

            #Print current weights
            print('W1: ')
            print(self.w1.eval())
            print('\n')
            print('W2: ')
            print(self.w2.eval())
            print('\n\n')

        #Return next move
        return chosenIndex

import numpy as np, random
import tensorflow as tf

class C4AI:
    def __init__(self, input_size, h1_size, output_size):
        'Define weights, biases, and create placeholder for yhat'
        self.reward = 0
        self.discReward = .5
        self.learnRate = .1
        self.costs = []
        self.lastBoardState = None
        self.w1 = np.random.rand(input_size, h1_size)
        self.b1 = np.random.rand(1, h1_size)
        self.w2 = np.random.rand(h1_size, output_size)
        self.b2 = np.random.rand(1, output_size)

    '''
        Function evaluateMoves:
        This function will choose the next best move.

        Parameters: NextMoves - list of entire game state of future next potential moves

        Returns: int - column for next move
    '''

    def evaluateMoves(self, CurrentState, NextMoves, CurPlayerColor, OpponentColor, CurPlayerWin, OpponentWin, BoardFull):
        # Convert CurrentState into input vector
        currentState = np.matrix(np.ravel(np.array(CurrentState)))
        currentState[np.where(currentState == 'black')] = -1
        currentState[np.where(currentState == 'red')] = 1
        currentState[np.equal(currentState, None)] = 0
        currentStateInput = np.array(currentState, np.float32)

        # Separate move states from their columns and convert NextMoves into input vectors
        nextStates = []
        Columns = []

        if CurPlayerWin or OpponentWin or BoardFull:
            nextStates = np.matrix(np.ravel(np.array(NextMoves)))
            nextStates[np.where(nextStates == 'black')] = -1
            nextStates[np.where(nextStates == 'red')] = 1
            nextStates[np.equal(nextStates, None)] = 0
            nextStateInputs = np.array(nextStates, np.int32)

            chosenIndex = 0
            chosenNextState = nextStateInputs
        else:
            for st in NextMoves:
                st[np.where(st == 'black')] = -1
                st[np.where(st == 'red')] = 1
                st[np.equal(st, None)] = 0
                nextStates.append(np.ravel(st[:7]))
                Columns.append(st[7])

            # Change possible next states into array
            nextStateInputs = np.matrix(np.array(nextStates, np.int32))

            # Change Columns list of column indices into an ndarray
            Columns = np.matrix(np.array(Columns, np.int32))

            '''
                I need to base the below move selection on a policy
            '''
            # Choose future based on policy (max, or random)
            chosenIndex, chosenNextState = self.chooseNextState(nextStateInputs, Columns)
            #chosenNextState = nextStateInputs[chosenIndex]

        # Calculate current reward
        if CurPlayerWin:
            self.reward = 50

            # Get expected reward
            self.yHat = self.forwardProp(chosenNextState)
            print('The expected reward for player %s is: %f\n' % (CurPlayerColor, self.yHat))

            # Print actual reward
            print('The actual reward for player %s is: %f\n\n' % (CurPlayerColor, self.reward))

            # Perform cost function and get weight change gradients
            dJdW1, dJdW2 = self.costFunctionPrime(currentStateInput, chosenNextState)

            # Add found error to costs array for later plotting
            self.costs.append(self.err)

            # Make weight changes (element-wise multiplication)
            self.w1 = dJdW2 * .1
            self.w2 = dJdW1 * .1
        elif OpponentWin:
            self.reward = -50

            # Get expected reward
            self.yHat = self.forwardProp(chosenNextState)
            print('The expected reward for player %s is: %f\n' % (CurPlayerColor, self.yHat))

            # Print actual reward
            print('The actual reward for player %s is: %f\n\n' % (CurPlayerColor, self.reward))

            # Perform cost function and get weight change gradients
            dJdW1, dJdW2 = self.costFunctionPrime(currentStateInput, chosenNextState)

            # Add found error to costs array for later plotting
            self.costs.append(self.err)

            # Make weight changes (element-wise multiplication)
            self.w1 = dJdW2 * .1
            self.w2 = dJdW1 * .1

        elif BoardFull:
            self.reward = 0.5

            self.reward = -50

            # Get expected reward
            self.yHat = self.forwardProp(chosenNextState)
            print('The expected reward for player %s is: %f\n' % (CurPlayerColor, self.yHat))

            # Print actual reward
            print('The actual reward for player %s is: %f\n\n' % (CurPlayerColor, self.reward))

            # Perform cost function and get weight change gradients
            dJdW1, dJdW2 = self.costFunctionPrime(currentStateInput, chosenNextState)

            # Add found error to costs array for later plotting
            self.costs.append(self.err)

            # Make weight changes (element-wise multiplication)
            self.w1 = dJdW2 * .1
            self.w2 = dJdW1 * .1
        else:
            self.reward = 0.5



        # Print gradients
        #print('Gradients for dJdW1 and dJdW2 respectively are: ')
        #print(dJdW1)
        #print('\n')
        #print(dJdW2)
        #print('\n')

        # Print current weights
        #print('W1 before weight update: ')
        #print(self.w1)
        #print('\n')
        #print('W2 before weight update: ')
        #print(self.w2)
        #print('\n\n')

        # Print new weights
        #print('W1 after weight update: ')
        #print(self.w1)
        #print('\n')
        #print('W2 after weight update: ')
        #print(self.w2)
        #print('\n\n')

        # Return next move
        return chosenIndex

    def costFunctionPrime(self, currentStateInput, chosenNextState):
        # Compute derivative with respect to W1, W2, and W3

        # Compute error
        self.err = self.reward + (self.discReward * self.forwardProp(chosenNextState) - self.forwardProp(currentStateInput))

        delta3 = np.multiply(-(self.err), self.sigmoidPrime(self.z3))
        delta2 = np.multiply(np.dot(delta3, self.w2.T), self.sigmoidPrime(self.z2))

        dJdW1 = np.dot(self.a2.T, delta3)
        dJdW2 = np.dot(chosenNextState.T, delta2)

        return dJdW1, dJdW2

    def chooseNextState(self, input, columns):
        # Forward propagate to find best
        # statesChoice = tf.Variable(self.forwardProp(input))
        statesRewards = self.forwardProp(input)

        policy = np.random.rand(1, 1)
        if policy > .2:
            nextIndex = np.argmax(statesRewards, 0)
            chosenIndex = nextIndex.item(0)
            chosenIndex = columns[chosenIndex].item(0)
        else:
            # Choose are random move from states
            nextIndex = random.randint(0, (columns.shape[0] - 1))
            chosenIndex = columns[nextIndex].item(0)



        return chosenIndex, input[nextIndex]

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
        # Declare a next moves list object
        tempState = np.array(CurrentState)
        movesList = []
        found = False

        # Loop through columns in Connect 4 board
        for i in range(tempState.shape[0]):
            # Loop through every row within column (0 starts at top of column, so we need to loop backwards)
            val = tempState.shape[1] - 1
            for j in range(tempState.shape[1] - 1, -1, -1):
                # If element is empty, put current player into move and add to next moves list
                if tempState[i][j] == None:
                    # Add player to empty cell then add tempstate to movesList
                    tempState[i][j] = PlayerNum
                    movesList.append(np.append(tempState, [[i, i, i, i, i, i]], axis=0))

                    # Reset tempState to CurrentState for finding next valid move/state
                    tempState = np.array(CurrentState)

                    # Set found to true to exit inner loop
                    found = True

                # Exit inner loop if found
                if found:
                    found = False
                    break

        # Return list
        return movesList

    def sigmoid(self, z):
        # Apply sigmoid activation function
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoidPrime(self, z):
        # Derivative of Sigmoid Function
        return np.multiply((1.0 - self.sigmoid(z)), self.sigmoid(z))

    def forwardProp(self, input):
        #z_1 = tf.add(tf.matmul(input, self.w1), self.b1)
        #a_1 = self.sigmoid(z_1)
        #z_2 = tf.add(tf.matmul(a_1, self.w2), self.b2)
        #a_2 = self.sigmoid(z_2)
        self.z2 = np.dot(input, self.w1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.w2)
        yHat = self.sigmoid(self.z3)
        return yHat

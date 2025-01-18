import os
import sys
import random
import json
from TicTacToe import *

import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

#import copy

"""
You may import additional, commonly used libraries that are widely installed.
Please do not request the installation of new libraries to run your program.
"""

class SQNModelManager:

    def __init__(self):
        #Initializes the SQNModelManager class.
        self.__model = None #model is a private member variable holding the SQN Model
        self.__replayBuffer = None  #Replay Buffer is a private member variable holding training data
        self.__replayBufferMaxLen=10000         #Max size of replay buffer
        self.__ModelFilePath = "2021B5A72517G_MODEL.keras"
        self.__replayBufferRandomBootstrapdataLen = 200

    #Function to initialize model, train with random data and save model and training data buffer
    def initialTrainingWithRandomData (self):
        #print ("Bootstrap training in SQNModelManager called")

        # Step 1: Initialize Replay Buffer and Generate Random Experience Data
        self.__replayBuffer = deque(maxlen=self.__replayBufferMaxLen)    # Old experiences are removed when full

        # Step 2 : Generate random experience data: (s, a, r, s', done). Fill 20% of replay buffer witn random experience
        np.random.seed(42)  # For reproducibility
        for _ in range(self.__replayBufferRandomBootstrapdataLen):
            state = np.random.rand(9)         # Random state (9-dimensional input)
            action = np.random.randint(0, 9)  # Random action (0 to 8)
            reward = np.random.rand()         # Random reward
            next_state = np.random.rand(9)    # Random next state
            done = np.random.choice([True, False])  # Randomly mark if episode ended
            self.__replayBuffer.append((state, action, reward, next_state, done))

        # Step 3: Train the SQN basis the random experience data.
        self.__train()

        # Step 4: Save Replay Buffer to File for subsequent finetuning
        self.saveReplayBuffer ()

        print("Boostrap Training completed.")

    def __train(self):

        if self.__replayBuffer is None:
            self.__load_replay_buffer()

        # Step 1: Define the neural network
        self.__model = Sequential()
        self.__model.add(Dense(64, input_dim=9, activation='relu'))  # First hidden layer
        self.__model.add(Dense(64, activation='relu'))               # Second hidden layer
        self.__model.add(Dense(9, activation='linear'))              # Output layer

        # Step 2: Compile the model with Mean Squared Error loss and Adam optimizer
        self.__model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        # Step 3: Training Parameters
        BATCH_SIZE = 32  # Size of mini-batch
        GAMMA = 0.99     # Discount factor
        EPOCHS = 7      # Number of training epochs

        # Step 4: Train the Neural Network Using Mini-Batches
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch + 1}/{EPOCHS}")

            # Sample mini-batches from replay buffer
            for step in range(len(self.__replayBuffer) // BATCH_SIZE):

                # Randomly sample a mini-batch of experiences
                #old code
                #mini_batch = random.sample(self.__replayBuffer, BATCH_SIZE)

                #New code to reduce latency from random.sample
                indices = np.random.choice(len(self.__replayBuffer), BATCH_SIZE, replace=False)
                mini_batch = [self.__replayBuffer[idx] for idx in indices]


                # Prepare input and target batches
                states = np.zeros((BATCH_SIZE, 9))
                q_values_batch = np.zeros((BATCH_SIZE, 9))

                #New code
                # Prepare arrays for states and next_states
                states = np.array([sample[0] for sample in mini_batch])  # Batch of current states
                next_states = np.array([sample[3] for sample in mini_batch])  # Batch of next states

                # Predict Q-values for all states and next_states in the batch
                q_values_batch = self.__model.predict(states, verbose=0)  # Shape (batch_size, num_actions)
                q_next_batch = self.__model.predict(next_states, verbose=0)  # Shape (batch_size, num_actions)


                for j, (state, action, reward, next_state, done) in enumerate(mini_batch):

                    #old code
                    # # Predict Q-values for current state and next state
                    # q_values = self.__model.predict(state[np.newaxis, :], verbose=0)  # Shape (1, 9)
                    # q_next = self.__model.predict(next_state[np.newaxis, :], verbose=0)  # Shape (1, 9)

                    # Compute the Q-target using Bellman equation
                    if done:
                        q_target = reward  # No future rewards if the episode ended
                    else:
                        q_target = reward + GAMMA * np.max(q_next_batch[j])


                    # Update the Q-value for the chosen action
                    q_values_batch[j, action] = q_target

                    # Store the updated values in the batch
                    #states[j] = state
                    #q_values_batch[j] = q_values

        # Step 5: Perform gradient descent step on the mini-batch
        self.__model.fit(states, q_values_batch, epochs=1, verbose=0)

        # Step 6: Save the model to file
        self.__model.save(self.__ModelFilePath)  # Saves the entire model to a file

        #Step 7: Load the model from the file for later use.
        self.__model = None
        self.loadModel ()

    def train (self):
        self.__train()

    #Function to load pretrained model for inference.
    def loadModel (self):
        if self.__model is None:
            self.__model = load_model(self.__ModelFilePath)

    #Function to get next move.
    def getNextMove (self, state):
        if self.__model is None:
            self.loadModel ()

        #Todo : Check what reshaping does and if it is needed.
        action = self.__getNextMoveUsingEpsilonGreedy(state)
        return action

    #Epsilong greedy impl of selecting action
    def __getNextMoveUsingEpsilonGreedy (self, state) :
        epsilon = 0.1       # Exploration happens 10% of the times. Rest is exploitation.

        currentBoard = np.array(state)

        # Predict Q-values for the current state
        q_values = self.__model.predict(currentBoard[np.newaxis, :], verbose=0)[0]  # Flatten the (1, 9) shape to (9,)

        # Get valid actions
        valid_actions = [i for i in range(9) if state[i] == 0]  # 0 indicates an empty cell

        if random.random() < epsilon:           #Random generated between 0 and 1.
            # Exploration: choose a random action
            action = random.choice(valid_actions)  # Assuming 9 actions (for each cell on the Tic-Tac-Toe board)
        else:
            # Exploitation: choose the action with the highest Q-value

            #Old code:
            action = np.argmax(q_values)

            #New code:
            valid_q_values = [q_values[i] if i in valid_actions else -float('inf') for i in range(9)]
            action = np.argmax(valid_q_values)

        return action


    def saveReplayBuffer (self):
        # Iterate through each tuple in the replay buffer
        json_array = []
        for state, action, reward, next_state, done in self.__replayBuffer:
            json_element = {
                "state": state.tolist(),
                "action": action,
                "reward": reward,
                "next_state": next_state.tolist(),
                "done": bool(done),
            }
            json_array.append(json_element)

        filename="replay_buffer.json"
        with open(filename, "w") as f:
             json.dump(json_array, f, indent=4)


    def __load_replay_buffer(self):
        filename="replay_buffer.json"

        # Open the file in read mode and read its contents into a string variable

        with open(filename, "r") as file:
            file_contents = file.read()

        # Convert String containing JSON Array to JSON Object Array.
        data_list = []
        try:
            data_list = json.loads(file_contents)
        except Exception as e:
            # Catch all exceptions
            print(f"An error occurred: {e}")

        #Allocate replay buffer and add values from the JSON to the deque
        self.__replayBuffer = deque(maxlen=self.__replayBufferMaxLen)
        for item in data_list:
            state = item.get("state")
            action = item.get("action")
            reward = item.get("reward")
            next_state = item.get("next_state")
            done = item.get("done")

            #Convert the variables to the numpy data types
            state = np.array(state)
            next_state = np.array(next_state)
            done = np.bool_(done)

            self.__replayBuffer.append((state, action, reward, next_state, done))


    def addExperienceData(self, state, action, reward, next_state, done):
        #Step 1: Load replay buffer
        self.__load_replay_buffer()

        #Step 2 : Check if the replay buffer is full, it yes, remove oldest item.
        if len(self.__replayBuffer) == self.__replayBufferMaxLen:
            self.__replayBuffer.popleft()  # Remove the first (oldest) item

        #Step 3: Create tuple out of new experience data and append to tuple.
        stateVar = np.array(state) if isinstance(state, list) else state
        actionVar = int(action)
        rewardVar = int(reward)
        next_stateVar = np.array(next_state) if isinstance(next_state, list) else next_state
        doneVar = np.bool_(done) if isinstance(done, bool) else done

        #self.__replayBuffer.append((np.array(state), action, reward, np.array(next_state), np.bool_(done)))
        #self.__replayBuffer.append((state, action, reward, next_state, done))
        self.__replayBuffer.append((stateVar, actionVar, rewardVar, next_stateVar, doneVar))

        #Step 4: Save the replay buffer.
        #self.saveReplayBuffer ()


    def finetune (self):
        # The replay buffer would / should have been updated earlier by calling
        # addExperienceData function to enhance training data.
        # Hence only training remains. Better to just load the replay buffer once.

        #Step 1: Load replay buffer
        self.__load_replay_buffer()

        #Step 2 : Train the model.
        self.__train()



class PlayerSQN:
    def __init__(self, inTraining=False):
            self.__sqnModelManager = None
            self.__inTraining = inTraining
            self.__ticTacToe = None
            self.__state1 = None
            self.__prevAction = None

    def move(self, state):
        """
        Determines Player 2's move based on the current state of the game.

        Parameters:
        state (list): A list representing the current state of the TicTacToe board.

        Returns:
        int: The position (0-8) where Player 2 wants to make a move.
        """
        # In your final submission, PlayerSQN must be controlled by an SQN. Use an epsilon-greedy action selection policy.
        # In this implementation, PlayerSQN is controlled by terminal input.
        #print(f"Current state: {state}")
        #action = int(input("Player 2 (You), enter your move (1-9): ")) - 1

        #Initialize SQL Model Manager if needed.
        if self.__sqnModelManager is None:
            self.__sqnModelManager = SQNModelManager()

        #Step 1 : call SQN Model Manager to get the next move. Ensure it is between 1 and 9.
        action = -2
        while self.__isInValidMove (state, action):
            action = self.__sqnModelManager.getNextMove(state)

        #Step 2 : IF inTraining == true
        if self.__inTraining:
            #Step 2.1 Check if the given state is same as prev state
            if self.__state1 == None:
                #First time move is called.
                self.__state1 = state

            elif self.__state1 == state:
                #This means prev action was wrong. So just update the action which happens later. So just pass.
                pass
            else:
                #Means the prev action was valid and we have a new state. Construct tuple for training.
                training_prev_state = self.__state1
                training_next_state = state
                training_Action = self.__prevAction
                training_reward = self.__ticTacToe.get_reward()
                if self.__ticTacToe.is_full() or self.__ticTacToe.current_winner is not None:
                    training_gameDoneVal = True
                else:
                    training_gameDoneVal = False

                self.__sqnModelManager.addExperienceData(training_prev_state, training_Action, training_reward, training_next_state, training_gameDoneVal)
                #self.__sqnModelManager.saveReplayBuffer () #Will be called post game ending
                self.__state1 = state  #Update the self.__state1 with new valid state.

        self.__prevAction = action

        return action

    def __isInValidMove (self, state, action):
        if action < 1 or action > 9:
            return True

        emptyPositions = self.__empty_positions(state)
        inValidMove = True
        if action in emptyPositions:
            inValidMove = False
        return inValidMove

    def __empty_positions(self, state):
            return [i for i in range(9) if state[i] == 0]

    def storeTicTacToeClass (self, ticTacToe):
        self.__ticTacToe = ticTacToe

    def addExperienceDataForEndState (self, endState, reward):
        #Initialize SQL Model Manager if needed.
        if self.__sqnModelManager is None:
            self.__sqnModelManager = SQNModelManager()

        #Add exprerience data for the end state.
        training_prev_state = self.__state1
        training_next_state = endState
        training_Action = self.__prevAction
        training_reward = reward
        training_gameDoneVal = True     #Always true here since game ended.
        self.__sqnModelManager.addExperienceData(training_prev_state, training_Action, training_reward, training_next_state, training_gameDoneVal)
        self.__sqnModelManager.saveReplayBuffer ()

    def saveExperienceData (self):
        #Initialize SQL Model Manager if needed.
        if self.__sqnModelManager is None:
            self.__sqnModelManager = SQNModelManager()

        self.__sqnModelManager.saveReplayBuffer ()

def testAddExperience ():
    print ("testAddExperience called")
    sqnModelManager = SQNModelManager()

    state = np.random.rand(9)         # Random state (9-dimensional input)
    action = np.random.randint(0, 9)  # Random action (0 to 8)
    reward = np.random.rand()         # Random reward
    next_state = np.random.rand(9)    # Random next state
    done = np.random.choice([True, False])  # Randomly mark if episode ended
    sqnModelManager.addExperienceData(state, action, reward, next_state, done)
    sqnModelManager.saveReplayBuffer ()

def bootstrapTraining (smartMovePlayer1):
    #print ("bootstrapTraining called")

    sqnModelManager = SQNModelManager()
    sqnModelManager.initialTrainingWithRandomData()

    #Save the trained model
    saveTrainedModel(smartMovePlayer1)
    test (smartMovePlayer1)


def iterative_finetune_test (smartMovePlayer1):
    cycleCount = 10
    for i in range (cycleCount):
        generateExperienceDataAndfinetune (smartMovePlayer1)
        saveTrainedModel(smartMovePlayer1)
        test (smartMovePlayer1)
        print ("Completed cycle No : ", i+1)


def generateExperienceDataAndfinetune (smartMovePlayer1):

    #Play multiple games with training option as True to generate experience data in replay buffer.
    trainingGameCount = 30
    playerSQN = PlayerSQN(True)

    for i in range(trainingGameCount):
        #sqnModelManager = SQNModelManager()
        game = TicTacToe(smartMovePlayer1,playerSQN)
        playerSQN.storeTicTacToeClass (game)
        game.play_game()
        reward = game.get_reward()
        endState = game.board

        #Save the experience data for the end state of the game.
        playerSQN.addExperienceDataForEndState(endState, reward)

        if i % 10 == 0:
            print(f"Training Games Played {i}")

    playerSQN.saveExperienceData()

    #Now train the model
    print(f"Starting Finetuning Training")
    sqnModelManager = SQNModelManager()
    sqnModelManager.train()
    print(f"Finetuning Training Completed")

def test (smartMovePlayer1):

    smartMovePlayer1List = [0, 0.25, 0.5, 0.8]
    results = {}
    for smartMovePlayerVal in smartMovePlayer1List:
        no_of_test = 10
        total_reward = 0
        for test in range(no_of_test):
            playerSQN = PlayerSQN()
            game = TicTacToe(smartMovePlayerVal,playerSQN)
            playerSQN.storeTicTacToeClass (game)
            game.play_game()
            reward = game.get_reward()
            total_reward = total_reward + reward

        results[smartMovePlayerVal] = total_reward / no_of_test

    print("Average Reward from 10 games each : ", results)

    #Save the values to a text file as comma seperated values.
    with open("results" + str(smartMovePlayer1) + ".txt", "a") as file:
        # Join values by commas and write to the file
        file.write(",".join(map(str, results.values())) + "\n")



def main(smartMovePlayer1):
    """
    Simulates a TicTacToe game between Player 1 (random move player) and Player 2 (SQN-based player).

    Parameters:
    smartMovePlayer1: Probability that Player 1 will make a smart move at each time step.
                     During a smart move, Player 1 either tries to win the game or block the opponent.
    """
#    random.seed(42)
    #sqnModelManager = SQNModelManager()
    playerSQN = PlayerSQN()
    game = TicTacToe(smartMovePlayer1,playerSQN)
    playerSQN.storeTicTacToeClass (game)
    game.play_game()

    # Get and print the reward at the end of the episode
    reward = game.get_reward()
    print(f"Reward for Player 2 (You): {reward}")


def saveTrainedModel(smartMovePlayer1):
    """
    Creates a folder and a subfolder based on the largest integer-named folder inside it.
    Copies specified files to the newly created subfolder using basic file operations.

    Parameters:
    smartMovePlayer1 (str): Name of the main folder to work with.
    """

    smartMovePlayer1Str = str(smartMovePlayer1)

    # Check if the main folder exists; if not, create it
    if not os.path.exists(smartMovePlayer1Str):
        os.makedirs(smartMovePlayer1Str)

    # Get the list of folders inside the main folder
    existing_folders = [
        int(folder) for folder in os.listdir(smartMovePlayer1Str)
        if folder.isdigit() and os.path.isdir(os.path.join(smartMovePlayer1Str, folder))
    ]

    # Determine the name for the new folder
    next_folder_name = str(max(existing_folders) + 1) if existing_folders else "1"
    new_folder_path = os.path.join(smartMovePlayer1Str, next_folder_name)

    # Create the new folder
    os.makedirs(new_folder_path)

    # Files to copy
    files_to_copy = ["2021B5A72517G_MODEL.keras", "replay_buffer.json"]
    for file_name in files_to_copy:
        # Check if the file exists in the current directory
        if os.path.exists(file_name):
            # Copy file manually
            src_path = file_name
            dest_path = os.path.join(new_folder_path, file_name)
            with open(src_path, 'rb') as src_file:
                with open(dest_path, 'wb') as dest_file:
                    dest_file.write(src_file.read())
        else:
            print(f"Warning: {file_name} does not exist in the current directory.")

    print(f"Created folder: {new_folder_path}")
    print(f"Copied files: {files_to_copy}")



if __name__ == "__main__":
    if len(sys.argv) == 3:
        trainingParam = sys.argv[2]
        if trainingParam == "bootstrap":
            try:
                smartMovePlayer1 = float(sys.argv[1])
                assert 0<=smartMovePlayer1<=1
            except:
                print("Usage: python 2021B5A72517G.py <smartMovePlayer1Probability> test")
                print("Example: python 2020A7PS0001.py 0.5 test")
                print("There is an error. Probability must lie between 0 and 1.")
                sys.exit(1)

            bootstrapTraining(smartMovePlayer1)

        elif trainingParam == "test":
            try:
                smartMovePlayer1 = float(sys.argv[1])
                assert 0<=smartMovePlayer1<=1
            except:
                print("Usage: python 2021B5A72517G.py <smartMovePlayer1Probability> test")
                print("Example: python 2020A7PS0001.py 0.5 test")
                print("There is an error. Probability must lie between 0 and 1.")
                sys.exit(1)

            test(smartMovePlayer1)


        elif trainingParam == "finetune":
            try:
                smartMovePlayer1 = float(sys.argv[1])
                assert 0<=smartMovePlayer1<=1
            except:
                print("Usage: python 2021B5A72517G.py <smartMovePlayer1Probability> finetune")
                print("Example: python 2020A7PS0001.py 0.5 finetune")
                print("There is an error. Probability must lie between 0 and 1.")
                sys.exit(1)

            #generateExperienceDataAndfinetune(smartMovePlayer1)
            iterative_finetune_test (smartMovePlayer1)
        else:
            print("You are looking to train this model")
            print("Usage: python 2021B5A72517G.py <smartMovePlayer1Probability> <bootstrap or finetune>")
            print("Example 1: python 2020A7PS0001.py 0.5 bootstrap")
            print("Example 1: python 2020A7PS0001.py 0.75 finetune")
            print("There is an error. Probability must lie between 0 and 1.")
            sys.exit(1)
    else:
        try:
            smartMovePlayer1 = float(sys.argv[1])
            assert 0<=smartMovePlayer1<=1
        except:
            print("Usage: python 2021B5A72517G.py <smartMovePlayer1Probability>")
            print("Example: python 2020A7PS0001.py 0.5")
            print("There is an error. Probability must lie between 0 and 1.")
            sys.exit(1)

        main(smartMovePlayer1)





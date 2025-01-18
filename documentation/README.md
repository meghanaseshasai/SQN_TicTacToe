# TicTacToe Model Evaluation

## Directory Structure for checking
```
Evaluation/
    |
    |--YourBITSid.py
    |--YourBITSid_MODEL.h5 (you can save it in any format you like)
    |--evaluate.py   (DO NOT MAKE CHANGES IN THIS FILE)
    |--TicTacToe.py  (DO NOT MAKE CHANGES IN THIS FILE)
```


## Evaluation Marks Breakdown (8 Marks)

### 0 Smartness Game (4 Marks)
- **Test Case**: Play 3 full games, using 3 different seed values.
- **Task**: You will get +1 point for win, 0 for Tie, -1 for Loss. 
- **Marking Scheme**: 
    - 4 Marks if total reward > 0
    - 2.5 Marks if total reward = 0
    - 1.5 Marks if total reward < 0

### 0.8 Smartness Game (4 Marks)
- **Test Case**: Play 3 full games, using 3 different seed values.
- **Task**: You will get +1 point for win, 0 for Tie, -1 for Loss. 
- **Marking Scheme**: 
    - 4 Marks if total reward > 0
    - 2.5 Marks if total reward = 0
    - 1.5 Marks if total reward < 0

## Note:
1. To open any file/model - do NOT use absolute or relative path, instead open files/models as follows:

    ```python
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "YourBITSid_MODEL.keras")
    ```

    Do the same for any file that you want to open.

2. You are free to change the model architecture for training. You can add layers and use different techiniques to improve model accuracy.

3. Do not make any changes in the TicTacToe.py file. We will be using the original file while checking.

4. We will be changing the seed values for random function during evaluation, so your final marks might not be the same as shown by the script right now.


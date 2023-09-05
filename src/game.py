import numpy as np

class GameBoard:

    def __init__(self):
        self.reset_board()

    def get_board(self): 
        return self.board

    def set_cell(self, index, value):
        self.board[index] = value

    def reset_board(self):
        self.board = np.zeros(shape=(9,1), dtype=np.int32)

class GameState:
    #TODO: Figure how to handle opponents
    def __init__(self) -> None:
        self.board = GameBoard()
        self.player = 1
        self.turn = 0
        self.opponent = ...

    def reset_state(self):
        self.turn = 0
        self.board.resetBoard()

    def get_board_state(self):
        return self.board.get_board()

    def is_game_done(self):
        if self.turn > 8: 
            return True

        for i in range(3): 
            if (i == 0 or i == 2):
                # Diagonals
                if (self.board.get_cell(i) == self.board.getCell(4) and
                    self.board.get_cell(8 - i) == self.board.getCell(i) and self.board.get_cell(i) != 0):
                    return True
            # Columns
            if (self.board.get_cell(i) == self.board.get_cell(i + 3) and
                self.board.get_cell(i + 6) == self.board.get_cell(i) and self.board.get_cell(i) != 0):
                return True
            # Rows
            row = i * 3
            if (self.board.get_cell(row) == self.board.get_cell(row + 1) and
                self.board.get_cell(row + 2) == self.board.get_cell(row) and self.board.get_cell(row) != 0):
                return True

        return False

    def make_move(self, action):
        if not self.is_game_done(): 
            self.board.set_cell(action, self.player)
            self.turn+=1

            if not self.is_game_done():
                self.make_opponent_move()

    def make_opponent_move(self):
        #TODO: Might require handling the case when the action is not valid
        action = ...
        self.board.set_cell(action, self.player+1)
        self.turn+=1
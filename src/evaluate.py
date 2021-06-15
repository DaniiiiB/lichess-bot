from ai import *
import os
import chess
import copy

material_value = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}
#max_material_value = 8*100 + 2*320 + 2*330 + 2*500 + 900 + 20000

black_pawn_table =  [[0,  0,  0,  0,  0,  0,  0,  0],
                    [50, 50, 50, 50, 50, 50, 50, 50],
                    [10, 10, 20, 35, 35, 20, 10, 10],
                    [5,  5, 10, 30, 30, 10,  5,  5],
                    [0,  0,  0, 25, 25,  0,  0,  0],
                    [5, -5,-10,  0,  0,-10, -5,  5],
                    [5, 10, 10,-20,-20, 10, 10,  5],
                    [0,  0,  0,  0,  0,  0,  0,  0]]
white_pawn_table = list(reversed(black_pawn_table))

black_knight_table =  [[-50,-40,-30,-30,-30,-30,-40,-50],
                      [-40,-20,  0,  0,  0,  0,-20,-40],
                      [-30,  0, 10, 15, 15, 10,  0,-30],
                      [-30,  5, 15, 20, 20, 15,  5,-30],
                      [-30,  0, 15, 20, 20, 15,  0,-30],
                      [-40,  5, 10, 15, 15, 10,  5,-40],
                      [-40,-20,  0,  5,  5,  0,-20,-40],
                      [-50,-35,-30,-30,-30,-30,-35,-50]]
white_knight_table = list(reversed(black_knight_table))

black_bishop_table = [[-20,-10,-10,-10,-10,-10,-10,-20],
                     [-10,  0,  0,  0,  0,  0,  0,-10],
                     [-10,  0,  5, 10, 10,  5,  0,-10],
                     [-10,  5,  5, 10, 10,  5,  5,-10],
                     [-10,  0, 10, 10, 10, 10,  0,-10],
                     [-10, 10, 10, 10, 10, 10, 10,-10],
                     [-10,  5,  0,  0,  0,  0,  5,-10],
                     [-20,-10,-10,-10,-10,-10,-10,-20]]
white_bishop_table = list(reversed(black_bishop_table))

black_rook_table = [[0,  0,  0,  0,  0,  0,  0,  0],
                   [5, 10, 10, 10, 10, 10, 10,  5],
                   [-5,  0,  0,  0,  0,  0,  0, -5],
                   [-5,  0,  0,  0,  0,  0,  0, -5],
                   [-5,  0,  0,  0,  0,  0,  0, -5],
                   [-5,  0,  0,  0,  0,  0,  0, -5],
                   [-5,  0,  0,  0,  0,  0,  0, -5],
                   [0,  0,  0,  5,  5,  0,  0,  0]]
white_rook_table = list(reversed(black_rook_table))

black_queen_table = [[-20,-10,-10, -5, -5,-10,-10,-20],
                    [-10,  0,  0,  0,  0,  0,  0,-10],
                    [-10,  0,  5,  5,  5,  5,  0,-10],
                    [-5,  0,  5,  5,  5,  5,  0, -5],
                    [0,  0,  5,  5,  5,  5,  0, -5],
                    [-10,  5,  5,  5,  5,  5,  0,-10],
                    [-10,  0,  5,  0,  0,  0,  0,-10],
                    [-20,-10,-10, -5, -5,-10,-10,-20]]
white_queen_table = list(reversed(black_queen_table))

black_king_table_early = [[-30,-40,-40,-50,-50,-40,-40,-30],
                         [-30,-40,-40,-50,-00,-40,-40,-30],
                         [-30,-40,-40,-50,-50,-40,-40,-30],
                         [-30,-40,-40,-50,-50,-40,-40,-30],
                         [-20,-30,-30,-40,-40,-30,-30,-20],
                         [-10,-20,-20,-20,-20,-20,-20,-10],
                         [20, 20, -5, -10,-10,-5, 20, 20],
                         [20, 30, 10,  0,  0, 10, 30, 20]]
white_king_table_early = list(reversed(black_king_table_early))

black_king_table_late = [[-50,-40,-30,-20,-20,-30,-40,-50],
                        [-30,-20,-10,  0,  0,-10,-20,-30],
                        [-30,-10, 20, 30, 30, 20,-10,-30],
                        [-30,-10, 30, 40, 40, 30,-10,-30],
                        [-30,-10, 30, 40, 40, 30,-10,-30],
                        [-30,-10, 20, 30, 30, 20,-10,-30],
                        [-30,-30,  0,  0,  0,  0,-30,-30],
                        [-50,-30,-30,-30,-30,-30,-30,-50]]
white_king_table_late = list(reversed(black_king_table_late))

white_position_table = {chess.PAWN: white_pawn_table, chess.KNIGHT: white_knight_table, chess.BISHOP: white_bishop_table, chess.ROOK: white_rook_table, chess.QUEEN: white_queen_table, chess.KING: white_king_table_early}
black_position_table = {chess.PAWN: black_pawn_table, chess.KNIGHT: black_knight_table, chess.BISHOP: black_bishop_table, chess.ROOK: black_rook_table, chess.QUEEN: black_queen_table, chess.KING: black_king_table_early}
position_table = {True: white_position_table, False: black_position_table}

prediction_contribution = 0.25
material_contribution = 0.25
position_contribution = 0.5

# loading the board from the API communication part

def load_board():
    moves = open("moves.txt", "r").read().split(" ")
    board = chess.Board()
    if '' in moves:
        moves.remove('')
    if moves:
        for move in moves:
            board.push(chess.Move.from_uci(move))
    return board
    

# material evaluation

def check_material_value(board, is_white):
    base_board = chess.BaseBoard(board.fen().split(' ')[0])
    value = 0
    for square in chess.SQUARES:
        if base_board.color_at(square) == is_white:
            value += material_value[base_board.piece_type_at(square)]
    return value
        
def material_difference(board):
    return check_material_value(board, True) - check_material_value(board, False)

def evaluate_material(board, turn):
    #turn = 1 if board.turn else -1
    return material_difference(board)/4000 * turn

# position evaluation

def check_position_value(board, is_white):
    base_board = chess.BaseBoard(board.fen().split(' ')[0])
    value = 0
    for square in chess.SQUARES:
        if base_board.color_at(square) == is_white:
            piece = base_board.piece_type_at(square)
            value += position_table[is_white][piece][chess.square_rank(square)][chess.square_file(square)]
    return value

def position_difference(board):
    return check_position_value(board, True) - check_position_value(board, False)

def evaluate_position(board, turn):
    #turn = 1 if board.turn else -1
    return position_difference(board)/380 * turn

# NN prediction checkmate evaluation

def evaluate_predict(board, model):
    matrix = matrix_from_board(board)
    encoded_matrix = encode(matrix)
    return model.predict([encoded_matrix])[0][0]

# final evaluation

def evaluate_all(board, model, turn):
    return prediction_contribution*evaluate_predict(board, model) + material_contribution*evaluate_material(board, turn) + position_contribution*evaluate_position(board, turn)

# subtrees

class Node:
    def __init__(self, data):
        self.children = []
        self.data = data


def make_subtree(root):
    current_root = Node(chess.Board(root.data.fen()))
    #print("Current Root")
    #print(current_root.data)
    moves = list(current_root.data.legal_moves)
    for move in moves:
        copy_board = chess.Board(current_root.data.fen())
        copy_board.push(move)
        node = Node(copy_board)
        #print("Child")
        #print(node.data)
        current_root.children.append(node)
    return current_root 

# searching algorithm

def alphabeta(node, depth, alpha, beta, maximizingPlayer, model, turn):
    node = make_subtree(node)
      
    if depth == 0 or not node.children:
        return evaluate_all(node.data, model, turn)
              
    if maximizingPlayer:
        value = float('-inf')
        best_move = None
        for child in node.children:
            child_copy = Node(chess.Board(child.data.fen()))
            value = max(value, alphabeta(child_copy, depth - 1, alpha, beta, False, model, turn))
            alpha = max(alpha, value)
            if value >= beta:
                break #(* β cutoff *)
        return value
    else:
        value = float('inf')
        best_move = None
        for child in node.children:
            child_copy = Node(chess.Board(child.data.fen()))
            value = min(value, alphabeta(child_copy, depth - 1, alpha, beta, True, model, turn))
            beta = min(beta, value)
            if value <= alpha:
                break #(* α cutoff *)
        return value



def generate_move(board, model):
    turn = 1 if board.turn else -1 # 1 for white and -1 for black
    
    if check_material_value(board, True) <= 20900 and check_material_value(board, False) <= 20900:
        white_position_table[chess.KING] = white_king_table_late
        black_position_table[chess.KING] = black_king_table_late
    else:
        white_position_table[chess.KING] = white_king_table_early
        black_position_table[chess.KING] = black_king_table_early

    if check_material_value(board, not board.turn) >= 23000:
        prediction_contribution = 0.25
        position_contribution = 0.5
    elif check_material_value(board, not board.turn) >= 22000:
        prediction_contribution = 0.37
        position_contribution = 0.38
    else:
        prediction_contribution = 0.5
        position_contribution = 0.25

    possible_moves = list(board.legal_moves)    
    maximum = -1
    the_move = None
    copy_board = board.copy()
    root = Node(copy_board)
    root = make_subtree(root)
    for child in root.children:
        eval = alphabeta(child,1,float('-Inf'), float('Inf'), False, model, turn)
        move = child.data.peek()
        
        #print("Final Evaluation: " + str(eval))
        #print(move)
        #print("\n")
        
        if eval > maximum:
            maximum = eval
            the_move = move
    return [the_move, maximum]

os.chdir('F:\\lichess-bot\\src\\Best Model')

model = initialize_network()
model.load_weights('best_model.h5')

os.chdir('F:\\lichess-bot\\src\\Comunication')

board = load_board()

[the_move, maximum] = generate_move(board,model)

print(the_move, file=open("move.txt","w"))





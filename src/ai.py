import os
import chess.pgn
import numpy as np
from keras import callbacks, optimizers
from keras.layers import (LSTM, BatchNormalization, Dense, Dropout, Flatten, TimeDistributed)
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model, model_from_json
from matplotlib import pyplot as plt
from chessboard import display

chess_encoder = {
    'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
    'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
    'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
    'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
    'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
    'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
    'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
    'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
    'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
    'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
    'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
    'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
    '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
}

def load_raw_data():
    os.getcwd()
    os.chdir('F:\\lichess-bot\\src')

    pgn = open("lichess_db_standard_rated_2020-02.pgn")

    os.chdir('F:\\lichess-bot\\src\\Training Data')
    open("games.pgn", "w").close()

    game_counter = 0
    game = chess.pgn.read_game(pgn)
    i=1
    while game:
        black_elo = int(game.headers["BlackElo"])
        white_elo = int(game.headers["WhiteElo"])
        termination = game.headers["Termination"]
        result = game.headers["Result"]
        filter_condition = black_elo > 2000 and white_elo > 2000 and termination == "Normal" and (result == "1-0" or result == "0-1")
        if filter_condition:
            game_counter += 1
            os.chdir('F:\\lichess-bot\\src\\Training Data')
            print(game, file=open("games.pgn", "a"), end="\n\n")
        game = chess.pgn.read_game(pgn)
        i+=1
        if i==1000000:
            break
    print(str(game_counter), file=open("number_of_games.txt","w"))


def load_training_data():
    os.chdir('F:\\lichess-bot\\src\\Training Data')
    pgn = open("games.pgn")
    games=[]
    game = chess.pgn.read_game(pgn)
    i=1
    while game:
        games.append(game)
        game = chess.pgn.read_game(pgn)
        if i == 20000:
            break
        i += 1       
    return games
    

def matrix_from_board(board):
    epd = board.epd().split(" ",1)[0]
    epd_rows = epd.split("/")
    matrix_rows = []
    for row in epd_rows:
        matrix_row = []
        for slot in row:
            if slot.isdigit():
                matrix_row = matrix_row + ['.'] * int(slot)
            else:
                matrix_row.append(slot)
        matrix_rows.append(matrix_row)
    return matrix_rows

def encode(matrix):
    encoded_rows = []
    for row in matrix:
        encoded_row = []
        for square in row:
            encoded_row.append(chess_encoder[square])
        encoded_rows.append(encoded_row)
    return encoded_rows

def create_dataset(games):
    X=[]
    y=[]
    i = 0
    for game in games:
        result = game.headers["Result"].split("-")
        winner=1 if result[0]=="1" else -1
        number_of_moves = 0
        for move in game.mainline_moves():
            number_of_moves += 1
        board = game.board()
        move_counter = 0
        for move in game.mainline_moves():
            board.push(move)
            value = winner * (move_counter/number_of_moves)
            matrix = matrix_from_board(board)
            encoded_matrix = encode(matrix)
            X.append([encoded_matrix])
            y.append(value)
            move_counter += 1
        i += 1
    X = np.array(X).reshape(len(X),8,8,12)
    y = np.array(y)
    X.shape
    os.chdir('F:\\lichess-bot\\src\\Training Data')
    np.save('X.npy',X)
    np.save('y.npy',y)
    return [X,y]

def initialize_network():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=1, activation='relu', input_shape=(8,8,12)))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=24, kernel_size=1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=10, kernel_size=1, activation='relu'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(1,activation = 'tanh'))
    return model

def train(model,X,y):
    model.compile(optimizer='Nadam', loss='mse')
    os.chdir('F:\\lichess-bot\\src\\Best Model')
    hdf5 = 'best_model.h5'
    checkpoint = callbacks.ModelCheckpoint(hdf5, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    es = callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=100)
    callback = [checkpoint,es]
    json = 'chess' + '_best_model' + '.json'
    model_json = model.to_json()
    with open(json, "w") as json_file:
        json_file.write(model_json)
    print('Training Network...')
    history = model.fit(X,y,epochs = 100,verbose = 2,callbacks = callback)
    plt.plot(history.history['loss'])
   
    
#games = load_training_data()
#[X,y] = create_dataset(games)
#model = initialize_network()
#train(model,X,y)











    

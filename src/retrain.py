from ai import *
import os
import chess


def train(model,X,y):
    optimizer = optimizers.Adam(0.001)
    #optimizer.learning_rate.assign(0.01)
    model.compile(optimizer, loss='mse')
    model.load_weights('best_model.h5')
    hdf5 = 'best_model.h5'
    checkpoint = callbacks.ModelCheckpoint(hdf5, monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', save_freq='epoch')
    es = callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=100)
    callback = [checkpoint,es]
    print('Training Network...')
    history = model.fit(X,y,epochs = 250,verbose = 2,callbacks = callback)
    plt.plot(history.history['loss'])
    plt.show()



os.chdir('F:\\lichess-bot\\src\\Training Data')
X = np.load('X.npy')
y = np.load('y.npy')
model = initialize_network()
os.chdir('F:\\lichess-bot\\src\\Best Model')
train(model,X,y)


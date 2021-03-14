import keras
from keras.layers import (
    Conv2DTranspose,
    ConvLSTM2D,
    TimeDistributed,
    Conv2D,
)
from keras.models import Sequential, load_model
from keras.layers import LayerNormalization

from clip_generator import DataGenerator 
from config import *

def get_model(reload_model=True):
    """
    Parameters
    ----------
    reload_model : bool
        Load saved model or retrain it
    """
    
    if not reload_model:
        return load_model(
            MODEL_PATH,
            custom_objects={'LayerNormalization': LayerNormalization}
        )

    training_generator = DataGenerator(
        DATASET_PATH,
        CLIP_LEN,
        STRIDE,
        DIM,
        BATCH_SIZE,
        N_CHANNELS,
        SHUFFLE
    )

    seq = Sequential()
    seq.add(TimeDistributed(
        Conv2D(16, (11,11), strides=4, padding="same"),
        batch_input_shape=(None, 10, 256, 256, 1)
    ))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(8, (8,8), strides=4, padding="same")))
    seq.add(LayerNormalization())
    ######
    seq.add(ConvLSTM2D(8, (3,3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(4, (2,2), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    seq.add(ConvLSTM2D(8, (3,3), padding="same", return_sequences=True))
    seq.add(LayerNormalization())
    ######
    seq.add(TimeDistributed(Conv2DTranspose(8, (8,8), strides=4, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2DTranspose(16, (11, 11), strides=4, padding="same")))
    seq.add(LayerNormalization())
    seq.add(TimeDistributed(Conv2D(1, (11,11), activation="sigmoid", padding="same")))
    
    print(seq.summary())

    seq.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6)
    )
    seq.fit(
        x=training_generator,
        epochs=EPOCHS,
        verbose=True,
        workers=0,
        use_multiprocessing=False
    )
    seq.save(MODEL_PATH)

    return seq
    


    
# Model Parameters

## Config

```py
EPOCHS = 1

CLIP_LEN = 10
STRIDE = 1
DIM = (CLIP_LEN, 256, 256)
BATCH_SIZE = 4
N_CHANNELS = 1
SHUFFLE = True
```

## Model

```py
seq.add(TimeDistributed(
    Conv2D(16, (11,11), strides=4, padding="same"),
    batch_input_shape=(None, *DIM, N_CHANNELS)
))
seq.add(LayerNormalization())
seq.add(TimeDistributed(Conv2D(8, (8,8), strides=2, padding="same")))
seq.add(LayerNormalization())
######
seq.add(ConvLSTM2D(8, (3,3), padding="same", return_sequences=True))
seq.add(LayerNormalization())
seq.add(ConvLSTM2D(4, (3,3), padding="same", return_sequences=True))
seq.add(LayerNormalization())
seq.add(ConvLSTM2D(8, (3,3), padding="same", return_sequences=True))
seq.add(LayerNormalization())
######
seq.add(TimeDistributed(Conv2DTranspose(8, (8,8), strides=2, padding="same")))
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
```

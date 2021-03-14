import numpy as np
import matplotlib.pyplot as plt

from config import *
from model import get_model
from data import get_single_test

def evaluate():
    model = get_model(reload_model=True)
    print("got model")
    test = get_single_test()
    print("got test")

    num_clips = test.shape[0] - CLIP_LEN
    clips = np.zeros((num_clips, *DIM, N_CHANNELS))
    
    # apply sliding window technique to get the clips
    for i in range(num_clips):
        clip = np.zeros((*DIM, N_CHANNELS))
        for j in range(CLIP_LEN):
            clip[j] = test[i+j, :, :, :]
        clips[i] = clip
    
    # get reconstruction cost of all the clips
    reconstructed_clips = model.predict(clips, batch_size=BATCH_SIZE)
    cost = np.array([
        np.linalg.norm( clips[i] - reconstructed_clips[i] )
        for i in range(num_clips)
    ])
    # arregularity score
    sa = (cost - np.min(cost)) / np.max(cost)
    # regularity score
    sr = 1.0 - sa

    # plot scores
    plt.plot(sr)
    plt.ylabel('regularity score Sr(t)')
    plt.xlabel('frame t')
    plt.show()

if __name__=="__main__":
    evaluate()

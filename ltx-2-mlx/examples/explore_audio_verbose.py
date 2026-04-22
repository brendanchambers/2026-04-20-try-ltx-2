from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder
import mlx.core as mx
import matplotlib.pyplot as plt
import numpy as np
# audio playback
import sounddevice as sd
import librosa

#### config #####
sr = 16000


decoder = AudioVAEDecoder()
print(decoder)
print('loaded dummy vae decoder')


def visualize_decoding():

    # latent = mx.zeros((1, 8, 10, 16))
    latent = mx.random.uniform(low=0, high=1, shape=(1, 8, 10, 16))
    
    mel = decoder.decode(latent)

    # check shape of mel spectrogram
    assert mel.shape[0] == 1
    assert mel.shape[1] == 2  # stereo
    # After 2 upsample stages (2x each): freq 16 -> 64
    assert mel.shape[3] == 64  # mel bins

    print(mel.shape)

    mel_L = np.array(mel[0, 0, :, :]).squeeze()
    # mel_R = np.array(mel[0, 1, :, :]).squeeze()  

    print('converting mel to stft')
    S_inv = librosa.feature.inverse.mel_to_stft(mel_L)
    # 3. Use the Griffin-Lim algorithm to reconstruct the phase and waveform
    print('griffin lim')
    y_reconstructed = librosa.griffinlim(S_inv)

    # 4. Play the audio
    print('playing audio')
    sd.play(y_reconstructed, sr)

    # plot
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    cax = ax.imshow(mel_L, interpolation='nearest', cmap='coolwarm', origin='lower')
    ax.set_title('Mel spec (L channel)')
    plt.show()



visualize_decoding()
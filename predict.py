"""

Import all useful packages

"""
import os
import librosa
import numpy as np
import math
import pandas as pd
from tensorflow.keras.models import load_model


# Data hyper-parameters 
# -----------------------------
lowpass_cutoff = 1500 # Cutt off for low pass filter
downsample_rate = 22050 # Frequency to downsample to
nyquist_rate = 11025 # Nyquist rate (half of sampling rate)
segment_duration = 3# how long should a segment be


# Spectrogram hyper-parameters 
# -----------------------------
n_fft = 1024 # Hann window length
hop_length = 256 # Sepctrogram hop size
n_mels = 128 # Spectrogram number of mells
f_min = 150 # Spectrogram, minimum frequency for call
f_max = 15000 # Spectrogram, maximum frequency for call


species_folder = "./Data/"
audio_dir = species_folder+"/audios/"
file_list_path = species_folder+"/DataFiles/Testfiles.txt"

def check_wav_file_exists(filename):
    """
    Function to check if a .wav file exists in the audio directory
    
    Args:
        filename (str): The name of the .wav file (without the extension).

    Returns:
        bool: True if the file exists, False otherwise.
    """
    wav_filename = os.path.join(audio_dir, filename + ".wav")
    return os.path.exists(wav_filename)


def audio_to_spectrogram(audio,
                         sr=downsample_rate,
                         n_fft=n_fft,
                         hop_length=hop_length,
                         n_mels=n_mels):

    """
    Convert audio signal into a mel-scaled spectrogram.

    Args:
    - audio (numpy ndarray): Input audio signal.
    - sr (integer): Sampling rate of the audio.
    - n_fft (integer): Number of FFT bins.
    - hop_length (integer): Hop length for STFT.
    - n_mels (integer): Number of mel bands to generate.

    Returns:
    - numpy ndarray: Mel-scaled spectrogram of the input audio.

    """
    spectrum = librosa.feature.melspectrogram(y=audio,
                                              sr=sr,
                                              n_fft=n_fft,
                                              hop_length=hop_length,
                                              n_mels=n_mels)

    spectrum_dB = librosa.power_to_db(spectrum, ref=np.max)

    ## Apply normalization
    image = spectrum_dB
    image_np = np.asmatrix(image)
    image_np_scaled_temp = (image_np - np.min(image_np))
    image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)
    mean = image.flatten().mean()
    std = image.flatten().std()
    eps=1e-8
    spec_norm = (image - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)

    return spectrum_dB

def butter_lowpass(cutoff, nyq_freq, order=4):
    """
    Design a lowpass Butterworth filter.

    Args:
        cutoff (float): The cutoff frequency of the filter.
        nyq_freq (float): The Nyquist frequency of the signal.
        order (integer): The order of the filter.

    Returns:
        tuple: The numerator (b) and denominator (a) polynomials of the filter.
    """
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):

    """
    Apply a lowpass Butterworth filter to the input data.

    Args:
        data (ndarray): The input data to be filtered.
        cutoff_freq (float): The cutoff frequency of the filter.
        nyq_freq (float): The Nyquist frequency of the signal.
        order (integer): The order of the filter. 
    Returns:
        ndarray: The filtered data.
    """
    # Source: https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def downsample_file( amplitudes, original_sr, new_sample_rate):
    """
    Downsample the input audio file to a new sample rate.

    Args:
        amplitudes (ndarray): The audio waveform.
        original_sr (int): The original sample rate of the audio.
        new_sample_rate (int): The desired sample rate for downsampling.

    Returns:
        tuple: A tuple containing the downsampled audio waveform and the new sample rate.
    """
    return librosa.resample(y=amplitudes,
                            orig_sr=original_sr,
                            target_sr=new_sample_rate,
                            res_type='kaiser_fast'), new_sample_rate

def predict_on_entire_file(audio, sample_rate, lowpass_cutoff,
                           downsample_rate, nyquist_rate, model):
                           """
    Predict bird call events on the entire audio file.

    Args:
        audio (ndarray): The audio waveform.
        sample_rate (int): The sample rate of the audio.
        lowpass_cutoff (float): The cutoff frequency for the lowpass filter.
        downsample_rate (int): The desired sample rate for downsampling.
        nyquist_rate (float): The Nyquist frequency of the signal.
        model (keras.Model): The trained CNN model.

    Returns:
        dict: A dictionary containing the predictions for each segment of the audio file.
    """

    # Apply a low pass fitler to get rid of high frequency components
    filtered = butter_lowpass_filter(audio, lowpass_cutoff, nyquist_rate)

    # Downsample the audio
    amplitudes, sample_rate = downsample_file(filtered, sample_rate, downsample_rate)

    # Duration of file
    file_duration = len(amplitudes)/sample_rate

    # Number of segments
    segments = math.floor(file_duration) - 3

    # Store predictions in this dictionary
    predictions = {}

    # Loop over the file and work in small "segments"
    for position in range (0, segments):

        # Determine start of segment
        start_position = position

        # Determine end of segment
        end_position = start_position + 3

        # Extract a 3 second segment from the audio file
        audio_segment = amplitudes[start_position*downsample_rate:end_position*downsample_rate]

        # Create the spectrogram
        S = audio_to_spectrogram(audio_segment)

        # Input spectrogram into model
        softmax = model.predict(np.reshape(S, (1,128,259,1)))

        # Record result
        predictions[f"{start_position}-{end_position}"] = 'no-call' if np.argmax(softmax,-1)[0]== 0 else 'call'

    return predictions


# Read the contents of the text file
with open(file_list_path, 'r', encoding='utf-8') as file:
    filenames = file.read().splitlines()

## List that will contains all test files
wav_filenames = []

# Check each filename and append to the list if it exists with .wav extension
for filename in filenames:
    if check_wav_file_exists(filename):
        wav_filenames.append(filename + ".wav")

model = load_model('model3.keras')


# Iterate over each audio file
for audio_file in wav_filenames:   
    # Load audio file
    audio_data, sample_rate = librosa.load(audio_dir+audio_file)  
    # Make prediction for each audio file
    predictions = predict_on_entire_file(audio_data, sample_rate, lowpass_cutoff,
                                          downsample_rate, nyquist_rate, model)   
   # Create file name that maps each audio file with a specific csv file
    output_csv = f"{os.path.splitext(audio_file)[0]}_predictions.csv"
    # Convert the output into dataframe
    pred = pd.DataFrame(list(predictions.items()), columns=['Time (s)', 'Prediction'])  
    # Convert the dataframe to csv file
    pred.to_csv(output_csv, index=False)

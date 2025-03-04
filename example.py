import pandas as pd
from scipy.io import wavfile
import PyOctaveBand

def main():
    fs, data = wavfile.read(r"\\192.168.205.123\aac_server\CALIBRATION\AAC_SENSOR\3-Medidas\0001\AUDIO\20250303_000001.wav")
    print(f"Sample rate: {fs} Hz")
    duration = len(data) / fs
    print(f"Duration: {duration} seconds")
    


    # mono
    if data.ndim > 1:
        data = data[:, 0]

    n_seconds = len(data) // fs
    results = [] 
    freq_labels = None

    for sec in range(n_seconds):
        segment = data[sec * fs : (sec + 1) * fs]

        # calculate octave band levels
        levels, freqs = PyOctaveBand.octavefilter(segment, fs, fraction=3)
        levels = [round(level, 2) for level in levels]
        if freq_labels is None:
            freq_labels = [f"{round(freq, 1)}Hz" for freq in freqs]
        results.append(levels)

    

    # create csv
    df = pd.DataFrame(results, columns=freq_labels)
    df.insert(0, "Second", range(1, n_seconds + 1))

    # save csv final file
    df.to_csv("frequency_levels_per_second.csv", index=False)
    print("CSV file 'frequency_levels_per_second.csv' has been saved.")


if __name__ == '__main__':
    main()

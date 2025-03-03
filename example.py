import csv
from scipy.io import wavfile
import PyOctaveBand


def main():
    fs, data = wavfile.read(r"\\192.168.205.123\aac_server\CALIBRATION\AAC_SENSOR\3-Medidas\0001\AUDIO\20250303_000001.wav")
    print(f"Sample rate: {fs} Hz")
    print(f"Duration: {len(data) / fs} seconds")
    

    #mono
    if data.ndim > 1:
        data = data[:, 0]

    

    # 1/3-octave filter bank (set fraction=3 for 1/3 octave bands)
    levels, freqs = PyOctaveBand.octavefilter(data, fs, fraction=3)
    print(f"Levels: {levels}")
    print(f"Freqs: {freqs}")

    levels_A, freqs = PyOctaveBand.octavefilter(data, fs, fraction=3)
    levels_C, freqs = PyOctaveBand.octavefilter(data, fs, fraction=3)
    levels_Z, freqs = PyOctaveBand.octavefilter(data, fs, fraction=3)

    print(f"Levels A: {levels_A}")
    print(f"Levels C: {levels_C}")
    print(f"Levels Z: {levels_Z}")

    

if __name__ == '__main__':
    main()
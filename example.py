import pandas as pd
from scipy.io import wavfile
import PyOctaveBand
import os
import logging
import argparse
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    filename='calibration_test.log', 
    filemode='w'
)


def find_audio_folders(base_path):
    """Recursively find all subdirectories containing an 'AUDIOMOTH' folder."""
    for root, dirs, files in os.walk(base_path):
        if 'AUDIO' in dirs:
            yield root


def get_audiofiles(path):
    """
    Args:
        path (str): The path to the directory containing the audio files.
    Returns:
        list: A list containing the full paths to all '.wav' files in the specified directory.
    """    
    audio_files = [file for file in os.listdir(path) if file.lower().endswith('.wav')]
    return audio_files


def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate SPL levels for audio files in a directory')
    parser.add_argument('-p', '--path', type=str, required=True, help='Directory to be processed')
    # parser max and min
    parser.add_argument('-lb', '--lower_bound', type=float, default=90.0, help='Lower bound for lineal differenciation')
    parser.add_argument('-ub', '--upper_bound', type=float, default=112.0, help='Upper bound for lineal differenciation')
    # calibration constant from the multifunction
    parser.add_argument('-t', '--threshold', type=int, default=94.0, help='Threshold constant for the microphone')
    return parser.parse_args()



def main():
    args = parse_arguments()

    base_path = args.path
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    threshold_multifinction = args.threshold
    logging.info(f"Processing files in: {base_path}")
    logging.info(f"Lower bound: {lower_bound}")
    logging.info(f"Upper bound: {upper_bound}")
    logging.info(f"Threshold multifuntion: {threshold_multifinction}")
    logging.info("")

    audio_folders = list(find_audio_folders(base_path))
    logging.info(f"Found {len(audio_folders)} folders to process.")

    for subfolder in tqdm(audio_folders, desc='Processing folders'):
        logging.info(f"Processing audio files in: {subfolder}...")
        audio_path = os.path.join(subfolder, "AUDIO")
        if not os.path.exists(audio_path):
            logging.warning(f"Skipping {subfolder}, AUDIO folder not found.")
            continue

        audio_files = get_audiofiles(audio_path)
        if not audio_files:
            logging.warning(f"No audio files found in: {audio_path}")
            continue


        for audio_file in tqdm(audio_files, desc='Processing audio files'):
            # fs, data = wavfile.read(r"\\192.168.205.123\aac_server\CALIBRATION\AAC_SENSOR\3-Medidas\0001\AUDIO\20250303_000001.wav")
            fs, data = wavfile.read(os.path.join(audio_path, audio_file))
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

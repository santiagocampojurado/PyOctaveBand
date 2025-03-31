import os
import pandas as pd
import datetime
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from tqdm import tqdm

import logging
import argparse

from utils import *
from config import *

import PyOctaveBand
import PyOctaveBand_reduced



logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    filename='calibration_test.log', 
    filemode='w'
)


SPECTRUM_COLUMNS = ['31.6Hz', '63.1Hz', '125.9Hz', '251.2Hz', '501.2Hz',
                    '1000.0Hz', '1995.3Hz', '3981.1Hz', '7943.3Hz', '12589.3Hz',
                    '15848.9Hz'
                ]

SPECTRUM_COLUMNS_LITTLE = ['31.6Hz', '63.1Hz', '125.9Hz', '251.2Hz', '501.2Hz',
                            '1000.0Hz', '1995.3Hz', '3981.1Hz']



def leq(levels):
    e_sum = (np.sum(np.power(10, np.multiply(0.1, levels)))) / len(levels)
    eq_level = 10 * np.log10(e_sum)
    eq_level = round(eq_level, 2)
    return eq_level



def find_audio_folders(base_path: str):
    for root, dirs, files in os.walk(base_path):
        if 'AUDIO' in dirs:
            yield root



def get_audiofiles(path: str) -> list:
    audio_files = [file for file in os.listdir(path) if file.lower().endswith('.wav')]
    return audio_files



def plot_calibration_test(df: pd.DataFrame, output_path_plot: str, audio_file: str, spectrum_columns: list = SPECTRUM_COLUMNS) -> None:
    try:
        logging.info("Plotting calibration test")

        plt.figure(figsize=(20, 10))
        for column in spectrum_columns:
            plt.plot(df[column], label=column)

        # legend outsode the plot
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.85))
        # grreid
        plt.grid()

        # drawing a line on the 94db level
        plt.axhline(y=94, color='r', linestyle='--')

        plt.title(f"Spectrum levels {audio_file}")
        plt.xlabel("Time")
        plt.ylabel("SPL")


        # # x limits
        plt.xlim(0, len(df))
        plt.tight_layout()


        # save plot
        plt.savefig(output_path_plot)
        logging.info(f'Plot saved to {output_path_plot}')


    except Exception as e:
        logging.error(f"Error plotting calibration test: {e}")





def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Calculate SPL levels for audio files in a directory')
    parser.add_argument('-p', '--path', type=str, required=True, help='Directory to be processed')
    parser.add_argument('-lb', '--lower_bound', type=float, default=160.0, help='Lower bound for lineal differenciation')
    parser.add_argument('-ub', '--upper_bound', type=float, default=166.0, help='Upper bound for lineal differenciation')
    parser.add_argument('-t', '--threshold', type=int, default=94, help='Threshold constant for the microphone')
    parser.add_argument('-c', '--calibration', type=float, default=None, help='Calibration coefficient')
    parser.add_argument('--lineal_diff', action='store_true', help='Perform lineal differenciation')
    return parser.parse_args()



def main() -> None:
    r"""
        usage: python.exe .\main.py -p "\\192.168.205.123\aac_server\CALIBRATION\NOISEPORT_TENERIFE_RP\3-Medidas\mic_1\" --lineal_diff
    """
    args = parse_arguments()

    base_path = args.path
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound
    threshold_multifinction = args.threshold
    calibration_coeff = args.calibration
    lineal_diff = args.lineal_diff
    logging.info(f"Processing files in: {base_path}")
    logging.info(f"Lower bound: {lower_bound}")
    logging.info(f"Upper bound: {upper_bound}")
    logging.info(f"Threshold multifuntion: {threshold_multifinction}")
    logging.info(f"Calibration coefficient: {calibration_coeff}")
    logging.info(f"Lineal differenciation: {lineal_diff}")
    logging.info("")
    

    # ----------------
    # calbration str
    # ----------------
    if calibration_coeff is not None:
        cal_str = "_calibrated"
    else:
        cal_str = ""



    # ----------------
    # FIND AUDIO FILES
    # ----------------
    audio_folders = list(find_audio_folders(base_path))
    logging.info(f"Found {len(audio_folders)} folders to process.")



    # ----------------
    # PROCESS FOLDERS & AUDIO FILES
    # ----------------
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
            try:
                logging.info(f"Processing audio file: {audio_file}")
                # Metadata
                full_audio_path = os.path.join(audio_path, audio_file)
                fs, data = wavfile.read(full_audio_path)
                logging.info(f"Sample rate: {fs} Hz")
                duration = len(data) / fs
                logging.info(f"Duration: {duration} seconds")
                window_size = fs  # 1-second window

                # NAME SPLIT and TIMESTAMP
                name_split = audio_file.split(".")[0]
                start_timestamp = datetime.datetime.strptime(name_split, '%Y%m%d_%H%M%S')
                timestamps = [start_timestamp + datetime.timedelta(seconds=sec) for sec in range(0, len(data) // fs)]

                # ---------------
                # MONO CONVERSION
                # ---------------
                if data.ndim > 1:
                    data = data[:, 0]



                # ----------------
                # 1 SEC PROCESSING
                # ----------------
                results = [] 
                freq_labels = None
                for start_idx in range(0, len(data) - window_size + 1, window_size):
                    segment = data[start_idx:start_idx + window_size]

                    #octave band levels for the segment
                    # levels, freqs = PyOctaveBand.octavefilter(segment, fs, fraction=3, order=4, show=0, sigbands=0, calibration_coeff=calibration_coeff)
                    # levels = [round(level, 2) for level in levels]

                    # testing octave band reduced
                    # def third_octave_filter(x, fs, order=6, limits=[12, 20000], show=False, sigbands=False, calibration_coeff=None):
                    # logging.info("Processing third octave filter")
                    levels, freqs = PyOctaveBand_reduced.third_octave_filter(segment, fs, order=4, show=0, sigbands=0, calibration_coeff=calibration_coeff)

                    # logging.info("Rounding levels")
                    levels = [round(level, 2) for level in levels]
                    
                    if freq_labels is None:
                        freq_labels = [f"{round(freq, 1)}Hz" for freq in freqs]
                    results.append(levels)



                # ----------------
                # SAVE CSV FILE
                # ----------------
                df = pd.DataFrame(results, columns=freq_labels)
                df.insert(0, "date", timestamps)
                df.insert(0, "filename", audio_file)
                
                output_folder = audio_path.replace('3-Medidas', '5-Results')
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                    logging.info(f"Created output folder: {output_folder}")


                # output_filename = f'{name_split}{cal_str}.csv'
                output_filename = f'{name_split}{cal_str}_test.csv'

                output_path = os.path.join(output_folder, output_filename)
                

                #save
                df.to_csv(output_path, index=False)
                logging.info(f'Output saved to {output_path}')



                # ---------------- 
                # PLOT CALIBRATION TEST
                # ----------------
                try:
                    output_path_plot = os.path.join(output_folder, f'calibration_test_{name_split}{cal_str}.png')
                    # output_path_plot = os.path.join(output_folder, f'calibration_test_{name_split}{cal_str}_test.png')
                    plot_calibration_test(df, output_path_plot, audio_file)
                    logging.info(f"Calibration test plot saved to: {output_path_plot}")
                except Exception as e:
                    logging.error(f"Error plotting calibration test: {e}")

                
                # ----------------
                # LINEAL DIFFERENCIATION
                # ----------------
                if lineal_diff:
                    try:
                        output_path_lineal = os.path.join(output_folder, f'lineal_diff_{name_split}{cal_str}.csv')
                        lineal_diff = lineal_differenciation(df, lower_bound, upper_bound, threshold_multifinction)
                        lineal_diff.to_csv(output_path_lineal, index=False)
                        logging.info(f"Lineal differenciation saved to: {output_path_lineal}")
                    except Exception as e:
                        logging.error(f"Error in lineal differenciation: {e}")
                else:
                    logging.info("Skipping lineal differenciation")



            # ----------------
            # END
            # ----------------
            except Exception as e:
                logging.error(f"Error processing audio file: {audio_file}")
                logging.error(e)


    logging.info("")
    logging.info("Processing finished.")

if __name__ == '__main__':
    main()

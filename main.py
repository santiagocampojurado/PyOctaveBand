import pandas as pd
import numpy as np
from scipy.io import wavfile
import PyOctaveBand
import os
import logging
import argparse
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt



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


def find_audio_folders(base_path):
    for root, dirs, files in os.walk(base_path):
        if 'AUDIO' in dirs:
            yield root



def get_audiofiles(path):
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


def lineal_differenciation(df: pd.DataFrame, lower_bound: float, upper_bound: float, calibration_multifinction: int, columns: list = SPECTRUM_COLUMNS) -> np.ndarray:
    df = df[columns]
    print(df)
    results = {}

    for col in df.columns:
        # print(f"Processing column: {col}")
        values = df[col].to_numpy()

        #boolean mask for the range [lower_bound, upper_bound]
        mask = (values >= lower_bound) & (values <= upper_bound)
        logging.info(f"Mask: {mask}")
        

        #finding the indices where mask is True
        true_indices = np.where(mask)[0] # [0] to get the indices
        logging.info(f"True indices: {true_indices}")
        
        
        if len(true_indices) == 0:
            results[col] = np.array([])
            continue

        
        # true indices for contiguous blocks
        segments = []
        current_segment = [true_indices[0]]
        
        for idx in true_indices[1:]:
            if idx == current_segment[-1] + 1:
                current_segment.append(idx)
            else:
                segments.append(current_segment)
                current_segment = [idx]
        segments.append(current_segment)

        
        
        longest_segment = max(segments, key=len)
        results[col] = values[longest_segment]
        logging.info(f"Longest segment: {longest_segment}")
        logging.info(f"Length of longest segment: {len(longest_segment)}")
        logging.info(f"These are the actual values in the longest segment: {values[longest_segment]}")
        logging.info(f"Results: {results}")
        logging.info("")
        
    
    #average for each column to get just one value
    for key, value in results.items():
        #apply the leq function
        results[key] = leq(value)
    reference_1khz = results['1000.00Hz']


    # calibration coefficient
    calibr_coeff = calibration_multifinction - reference_1khz
    calibr_coeff = round(calibr_coeff, 2)
    logging.info(f"Calibration coefficient: {calibr_coeff}")

    
    # difference between the reference and the other values
    differences = {}
    for key, value in results.items():
        differences[key] = round(value - reference_1khz, 2)
        
    
    #sort the results and renaming the columns
    freq_cols = sorted(results.keys(), key=lambda x: float(x.replace('Hz','')))
    numeric_freqs = [float(f.replace('Hz', '')) for f in freq_cols]

    df_out = pd.DataFrame({
        "Frequency": numeric_freqs,
        "PowerAvg": [results[f] for f in freq_cols],
        "Lineality": [differences[f] for f in freq_cols]
    })

    df_out["Calibr_Coeff"] = np.nan
    if not df_out.empty:
        df_out.loc[df_out.index[0], "Calibr_Coeff"] = calibr_coeff

    return df_out


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Calculate SPL levels for audio files in a directory')
    parser.add_argument('-p', '--path', type=str, required=True, help='Directory to be processed')
    parser.add_argument('-lb', '--lower_bound', type=float, default=90.0, help='Lower bound for lineal differenciation')
    parser.add_argument('-ub', '--upper_bound', type=float, default=112.0, help='Upper bound for lineal differenciation')
    parser.add_argument('-t', '--threshold', type=int, default=94, help='Threshold constant for the microphone')
    return parser.parse_args()



def main() -> None:
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
                    levels, freqs = PyOctaveBand.octavefilter(segment, fs, fraction=3)
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

                output_filename = f'leq_oct_{name_split}.csv'
                output_path = os.path.join(output_folder, output_filename)
                

                #save
                df.to_csv(output_path, index=False)
                logging.info(f'Output saved to {output_path}')



                # ---------------- 
                # PLOT CALIBRATION TEST
                # ----------------
                try:
                    output_path_plot = os.path.join(output_folder, f'calibration_plot_{name_split}.png')
                    plot_calibration_test(df, output_path_plot, audio_file)
                    logging.info(f"Calibration test plot saved to: {output_path_plot}")
                except Exception as e:
                    logging.error(f"Error plotting calibration test: {e}")

                
                # ----------------
                # LINEAL DIFFERENCIATION
                # ----------------
                try:
                    output_path_lineal = os.path.join(output_folder, f'lineal_diff_{name_split}.csv')
                    lineal_diff = lineal_differenciation(df, lower_bound, upper_bound, threshold_multifinction)
                    lineal_diff.to_csv(output_path_lineal, index=False)
                    logging.info(f"Lineal differenciation saved to: {output_path_lineal}")
                except Exception as e:
                    logging.error(f"Error in lineal differenciation: {e}")


            # ----------------
            # END
            # ----------------
            except Exception as e:
                logging.error(f"Error processing audio file: {audio_file}")
                logging.error(e)



if __name__ == '__main__':
    main()

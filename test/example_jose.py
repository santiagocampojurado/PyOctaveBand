import os
import pandas as pd
import numpy as np 
from pyfilterbank.octbank import FractionalOctaveFilterbank
from scipy.signal import lfilter
import soundfile as sf
from pyfilterbank.splweighting import a_weighting_coeffs_design
from pyfilterbank.splweighting import c_weighting_coeffs_design
from pyfilterbank.octbank import frequencies_fractional_octaves
import datetime
import argparse
import matplotlib.pyplot as plt


SPECTRUM_COLUMNS = ['31.25', '62.5', '125.0', '250.0', '500.0',
                    '1000.0', '2000.0', '4000.0', '8000.0', '12699.21',
                    '16000.0'
                ]



def filterbanks(fs):
    third_oct = FractionalOctaveFilterbank(
        sample_rate=fs,
        order=4, 
        nth_oct=3.0,
        norm_freq=1000.0,
        start_band=-19,
        end_band=13,
        edge_correction_percent=0.01, 
        filterfun='cffi' 
        )
    
    octave = FractionalOctaveFilterbank(
        sample_rate=fs,
        order=4,
        nth_oct=1.0,
        norm_freq=1000.0,
        start_band=-5,
        end_band=4,
        edge_correction_percent=0.01,
        filterfun='cffi')

    return third_oct, octave


def get_db_level(x, C):
    pref = 0.000002
    level = 10 * np.log10(np.mean(x ** 2) / pref ** 2) + C
    return level


def get_oct_levels(y, octave, C):
    y_oct, _ = octave.filter(y)
    oct_level = [get_db_level(f, C) for f in y_oct.T]    
    return oct_level


# def write_csv_all_levels(audio_files:list ,fs_filterbanks:float , w_size: int, C:float):
def write_csv_all_levels(audio_file: str, fs_filterbanks: float, window_size: int, correction: float):
    bA, aA = a_weighting_coeffs_design(fs_filterbanks)
    bC, aC = c_weighting_coeffs_design(fs_filterbanks)
    
    freqs, _ = frequencies_fractional_octaves(-19, 13, 1000, 3)
    third_oct, octave = filterbanks(fs_filterbanks)
    print(f'Created octave filters and weighting coefficients with fs = {fs_filterbanks} Hz')

    col_names = ['LA', 'LC', 'LZ', 'LAmax', 'LAmin']
    w_fast_samples = int(window_size / 8)
    band_names = [str(np.round(band, 2)) for band in freqs]
    col_names.extend(band_names)
    
    db_records = []

    x, fs = sf.read(audio_file)
    if x.ndim > 1:
        x = x[:, 0]
    y_A_weighted = lfilter(bA, aA, x)
    y_C_weighted = lfilter(bC, aC, x)

    
    
    for start_idx in range(0, len(x) - window_size + 1, window_size):
        frame = x[start_idx:start_idx + window_size]
        yA_frame = y_A_weighted[start_idx:start_idx + window_size]
        yC_frame = y_C_weighted[start_idx:start_idx + window_size]

        
        LA = get_db_level(yA_frame, correction)
        LC = get_db_level(yC_frame, correction)
        LZ = get_db_level(frame, correction)

        
        fast_levels = []
        for fast_idx in range(0, window_size - w_fast_samples + 1, w_fast_samples):
            sub_yA = yA_frame[fast_idx:fast_idx + w_fast_samples]
            LAf = get_db_level(sub_yA, correction)
            fast_levels.append(LAf)
        LAmax = np.max(fast_levels)
        LAmin = np.min(fast_levels)

        oct_levels = get_oct_levels(frame, third_oct, correction)
        record = [LA, LC, LZ, LAmax, LAmin] + oct_levels
        db_records.append(record)

    
    db_array = np.array(db_records)
    db_array = np.round(db_array, 2)
    df_history = pd.DataFrame(db_array, columns=col_names)
    df_history['filename'] = os.path.basename(audio_file)

    
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    try:
        start_time = datetime.datetime.strptime(base_name, '%Y%m%d_%H%M%S')
    except ValueError:
        start_time = datetime.datetime.now()
        print("Filename does not match expected datetime format, using current time as start.")

    num_records = len(df_history)
    df_history['date'] = pd.date_range(start=start_time, freq='s', periods=num_records)

    
    
    output_filename = 'output.csv'
    df_history.to_csv(output_filename, index=False)
    print()
    print(f"CSV file '{output_filename}' created with audio levels.")

    return df_history



def plot_calibration_test(df: pd.DataFrame, audio_file: str, spectrum_columns: list = SPECTRUM_COLUMNS) -> None:
    try:
        print()
        print("Plotting calibration test")

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
        plt.savefig('output_plot.png')
        print()
        print(f'Plot saved to output_plot.png')


    except Exception as e:
        print(f"Error plotting calibration test: {e}")




def argument_parser():
    parser = argparse.ArgumentParser(description='Calculate audio levels and write to CSV file.')
    parser.add_argument('audio_file', type=str, help='Path to the audio file.')
    parser.add_argument('--fs', type=float, default=44100, help='Sample rate of the audio file.')
    parser.add_argument('--window', type=int, default=44100, help='Window size for processing.')
    parser.add_argument('--correction', type=float, default=0.0, help='Correction factor for dB levels.')
    return parser.parse_args()



def main():
    # python.exe .\example_jose.py "\\192.168.205.123\aac_server\CALIBRATION\AAC_SENSOR\3-Medidas\0001\AUDIO\20250303_000001.wav" --fs 44100 --window 44100 --correction 0.0
    args = argument_parser()
    
    df = write_csv_all_levels(args.audio_file, args.fs, args.window, args.correction)

    
    plot_calibration_test(df, args.audio_file)


if __name__ == '__main__':
    main()
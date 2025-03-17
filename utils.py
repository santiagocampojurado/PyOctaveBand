import logging
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from config import *




def leq(levels):
    e_sum = (np.sum(np.power(10, np.multiply(0.1, levels)))) / len(levels)
    eq_level = 10 * np.log10(e_sum)
    eq_level = round(eq_level, 2)
    return eq_level


def lineal_differenciation(df: pd.DataFrame, lower_bound: float, upper_bound: float, calibration_multifinction: int, columns: list = SPECTRUM_COLUMNS) -> np.ndarray:
    print()
    
    df = df[columns]
    print(df)
    results = {}


    for col in df.columns:
        # print(f"Processing column: {col}")
        values = df[col].to_numpy()

        #boolean mask for the range [lower_bound, upper_bound]
        mask = (values >= lower_bound) & (values <= upper_bound)
        logging.info(f"Mask: {mask}")
        print(f"Mask: {mask}")

        

        #finding the indices where mask is True
        true_indices = np.where(mask)[0] # [0] to get the indices
        logging.info(f"True indices: {true_indices}")
        print(f"True indices: {true_indices}")
        # exit()
        
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
    reference_1khz = results['1000.0Hz']


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

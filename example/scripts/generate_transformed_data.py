import pandas as pd
import numpy as np
from TSB_AD.transformer.transformer import transform
from TSB_AD.utils.slidingWindows import find_length
import argparse

# =============================================================================
# --transformType 6: trans_name='_flat_region'
# --transformType 7: trans_name='flip_segment'
# --transformType 9: trans_name='_change_segment_add_scale'
# --transformType 10:trans_name='_change_segment_normalization'
# --transformType 11: trans_name='_change_segment_partial'
# --transformType 12: trans_name='_change_segment_resampling'
# =============================================================================


def main(original_file_path, output_file, transformType, contamination=None, para=None, seed=5):

    df = pd.read_csv(original_file_path, header = None)
    data = df[0].to_numpy().astype(float)
    label = df[1].to_numpy().astype(int)
    
    period = find_length(data)
    t = transform(transformType).transform(data, label, contamination=contamination, period=period, para=3, seed=15)
    
    new = np.array([t.data, t.label]).T
    
    df_new = pd.DataFrame(new, columns=['data','label'])
    
    df_new.to_csv(output_file, header=False, index=False)
    
parser = argparse.ArgumentParser()
parser.add_argument('--original_file_path', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--transformType', type=int, required=True)
parser.add_argument('--contamination', type=float, required=False)
parser.add_argument('--para', type=float, required=False)
parser.add_argument('--seed', type=int, required=True)

# Parse the argument
args = parser.parse_args()

# python generate_transformed_data.py --original_file_path ../../data/benchmark/ECG/MBA_ECG805_data.out --output_file ../data/synthetic/MBA_ECG805_data_12.out --transformType 12 --contamination 0.2 --para 3 --seed 15
if __name__ == "__main__":
    main(args.original_file_path, args.output_file, args.transformType,
         args.contamination, args.para, args.seed)
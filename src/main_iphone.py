"""Run AI-IMU dead-reckoning out-of-the-box on rigidly mounted iPhone CSVs.

Uses KITTI-trained nets + KITTIParameters unchanged (baseline, no retuning).
"""
import os
import matplotlib
matplotlib.use('Agg')  # headless plot export

from main_kitti import launch, KITTIParameters
from iphone_dataset import PhoneDataset
from utils_plot import plot_before_divergence, plot_stationary_periods


class PhoneArgs():
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path_data_csv = os.path.join(_repo_root, 'data', 'our_data')
    path_data_save = os.path.join(_repo_root, 'data', 'iphone')
    path_results = os.path.join(_repo_root, 'results', 'iphone')
    path_temp = os.path.join(_repo_root, 'temp')  # KITTI iekfnets + normalize_factors

    epochs = 400
    seq_dim = 6000

    # all phone sequences are test (no training)
    cross_validation_sequences = []
    test_sequences = []  # filled after convert; empty => none held out as train
    continue_training = False

    read_data = 1  # set 0 to skip CSV→pickle if data/iphone already built
    train_filter = 0
    test_filter = 1
    results_filter = 1
    dataset_class = PhoneDataset
    parameter_class = KITTIParameters


if __name__ == '__main__':
    args = PhoneArgs()
    # Populate test_sequences from CSV names so BaseDataset marks none as train
    if os.path.isdir(args.path_data_csv):
        args.test_sequences = [
            os.path.splitext(f)[0]
            for f in sorted(os.listdir(args.path_data_csv))
            if f.endswith('.csv')
        ]
    os.makedirs(args.path_data_save, exist_ok=True)
    os.makedirs(args.path_results, exist_ok=True)
    # Skip reconvert when pickles already exist unless forced
    if args.read_data and args.test_sequences:
        existing = {
            os.path.splitext(f)[0]
            for f in os.listdir(args.path_data_save)
            if f.endswith('.p')
        }
        if set(args.test_sequences) <= existing:
            print('Pickles already present in {}; set read_data=1 after deleting '
                  'them to reconvert.'.format(args.path_data_save))
            args.read_data = 0
    launch(args)
    if args.results_filter:
        ds = args.dataset_class(args)
        plot_before_divergence(args, ds)
        plot_stationary_periods(args, ds)

"""Run AI-IMU dead-reckoning out-of-the-box on rigidly mounted iPhone CSVs.

Uses KITTI-trained nets + KITTIParameters unchanged (baseline, no retuning).

    python main_iphone.py path/to/recording.csv
    python main_iphone.py --estimate-mount path/to/recording.csv
    python main_iphone.py --mount-rpy 180 0 -146 path/to/recording.csv
"""
import argparse
import os
import matplotlib
matplotlib.use('Agg')  # headless plot export

from main_kitti import launch, KITTIParameters
from iphone_dataset import PhoneDataset
from utils_plot import plot_before_divergence, plot_stationary_periods


class PhoneArgs():
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path_data_csv = os.path.join(_repo_root, 'data', 'our_data')
    path_data_save = os.path.join(_repo_root, 'data', 'iphone_zupt_raw')
    path_results = os.path.join(_repo_root, 'results', 'iphone_zupt_raw')
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
    csv_path = None
    estimate_mount = False  # set True with --estimate-mount
    mount_rpy_deg = None   # fixed phone->car rotation as [roll, pitch, yaw] in degrees
    zupt = True  # IMU-only zero-velocity update while the phone is still


def parse_cli():
    parser = argparse.ArgumentParser(
        description='Run AI-IMU dead-reckoning on an iPhone DeviceMotion CSV.')
    parser.add_argument(
        'csv', nargs='?', default=None,
        help='Path to one CSV. Omit to convert every CSV in data/our_data.')
    parser.add_argument(
        '--estimate-mount', dest='estimate_mount', action='store_true',
        help='Rotate IMU and attitude into the car frame using attitude and GPS heading.')
    parser.add_argument(
        '--mount-rpy', nargs=3, type=float, metavar=('ROLL', 'PITCH', 'YAW'),
        help='Rotate IMU and attitude by this fixed phone->car roll pitch yaw, in degrees.')
    parser.set_defaults(estimate_mount=False)
    return parser.parse_args()


def _pickles_match_mount(args):
    """True when every sequence pickle was built with the same mount setting."""
    for name in args.test_sequences:
        path = os.path.join(args.path_data_save, name + '.p')
        d = PhoneDataset.load(path)
        stored_est = d.get('estimate_mount', True)
        stored_rpy = d.get('mount_rpy_deg', None)
        want_rpy = getattr(args, 'mount_rpy_deg', None)
        if want_rpy is not None:
            if stored_rpy is None or list(stored_rpy) != list(want_rpy):
                return False
        elif bool(stored_est) != bool(args.estimate_mount):
            return False
        elif stored_rpy is not None:
            return False
    return True


if __name__ == '__main__':
    args = PhoneArgs()
    cli = parse_cli()
    args.estimate_mount = cli.estimate_mount
    if cli.mount_rpy is not None:
        args.mount_rpy_deg = cli.mount_rpy
        args.estimate_mount = False  # user supplied the rotation, skip estimation
    csv_arg = cli.csv
    if csv_arg:
        csv_path = os.path.abspath(csv_arg)
        if not os.path.isfile(csv_path):
            raise SystemExit('CSV not found: {}'.format(csv_path))
        args.csv_path = csv_path
        args.test_sequences = [os.path.splitext(os.path.basename(csv_path))[0]]
    elif os.path.isdir(args.path_data_csv):
        # Populate test_sequences from CSV names so BaseDataset marks none as train
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
        if set(args.test_sequences) <= existing and _pickles_match_mount(args):
            print('Pickles already present in {}; set read_data=1 after deleting '
                  'them to reconvert.'.format(args.path_data_save))
            args.read_data = 0
        elif set(args.test_sequences) <= existing:
            print('Reconverting {} because estimate_mount={} does not match '
                  'existing pickles.'.format(args.path_data_save, args.estimate_mount))
    launch(args)
    if args.results_filter:
        ds = args.dataset_class(args)
        plot_before_divergence(args, ds)
        plot_stationary_periods(args, ds)

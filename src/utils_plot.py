import matplotlib
import os
from termcolor import cprint
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from utils import *
from utils_torch_filter import TORCHIEKF

def results_filter(args, dataset):

    for i in range(0, len(dataset.datasets)):
        plt.close('all')
        dataset_name = dataset.dataset_name(i)
        file_name = os.path.join(dataset.path_results, dataset_name + "_filter.p")
        if not os.path.exists(file_name):
            print('No result for ' + dataset_name)
            continue

        print("\nResults for: " + dataset_name)

        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs = dataset.get_estimates(
            dataset_name)

        # get data
        t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)
        # get data for nets
        u_normalized = dataset.normalize(u).numpy()
        # shift for better viewing
        u_normalized[:, [0, 3]] += 5
        u_normalized[:, [2, 5]] -= 5

        t = (t - t[0]).numpy()
        u = u.cpu().numpy()
        ang_gt = ang_gt.cpu().numpy()
        v_gt = v_gt.cpu().numpy()
        p_gt = (p_gt - p_gt[0]).cpu().numpy()
        print("Total sequence time: {:.2f} s".format(t[-1]))

        ang = np.zeros((Rot.shape[0], 3))
        Rot_gt = torch.zeros((Rot.shape[0], 3, 3))
        for j in range(Rot.shape[0]):
            roll, pitch, yaw = TORCHIEKF.to_rpy(torch.from_numpy(Rot[j]))
            ang[j, 0] = roll.numpy()
            ang[j, 1] = pitch.numpy()
            ang[j, 2] = yaw.numpy()
            Rot_gt[j] = TORCHIEKF.from_rpy(torch.Tensor([ang_gt[j, 0]]),
                                        torch.Tensor([ang_gt[j, 1]]),
                                        torch.Tensor([ang_gt[j, 2]]))
            roll, pitch, yaw = TORCHIEKF.to_rpy(Rot_gt[j])
            ang_gt[j, 0] = roll.numpy()
            ang_gt[j, 1] = pitch.numpy()
            ang_gt[j, 2] = yaw.numpy()

        Rot_align, t_align, _ = umeyama_alignment(p_gt[:, :3].T, p[:, :3].T)
        p_align = (Rot_align.T.dot(p[:, :3].T)).T - Rot_align.T.dot(t_align)
        v_norm = np.sqrt(np.sum(v_gt ** 2, 1))
        v_norm /= np.max(v_norm)

        # Compute various errors
        error_p = np.abs(p_gt - p)
        # MATE
        mate_xy = np.mean(error_p[:, :2], 1)
        mate_z = error_p[:, 2]

        # CATE
        cate_xy = np.cumsum(mate_xy)
        cate_z = np.cumsum(mate_z)

        # RMSE
        rmse_xy = 1 / 2 * np.sqrt(error_p[:, 0] ** 2 + error_p[:, 1] ** 2)
        rmse_z = error_p[:, 2]

        RotT = torch.from_numpy(Rot).float().transpose(-1, -2)

        v_r = (RotT.matmul(torch.from_numpy(v).float().unsqueeze(-1)).squeeze()).numpy()
        v_r_gt = (Rot_gt.transpose(-1, -2).matmul(
            torch.from_numpy(v_gt).float().unsqueeze(-1)).squeeze()).numpy()

        p_r = (RotT.matmul(torch.from_numpy(p).float().unsqueeze(-1)).squeeze()).numpy()
        p_bis = (Rot_gt.matmul(torch.from_numpy(p_r).float().unsqueeze(-1)).squeeze()).numpy()
        error_p = p_gt - p_bis

        # plot and save plot
        folder_path = os.path.join(args.path_results, dataset_name)
        create_folder(folder_path)

        # position, velocity and velocity in body frame
        fig1, axs1 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # orientation, bias gyro and bias accelerometer
        fig2, axs2 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # position in plan
        fig3, ax3 = plt.subplots(figsize=(20, 10))
        # position in plan after alignment
        fig4, ax4 = plt.subplots(figsize=(20, 10))
        #  Measurement covariance in log scale and normalized inputs
        fig5, axs5 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))
        # input: gyro, accelerometer
        fig6, axs6 = plt.subplots(2, 1, sharex=True, figsize=(20, 10))
        # errors: MATE, CATE  RMSE
        fig7, axs7 = plt.subplots(3, 1, sharex=True, figsize=(20, 10))

        axs1[0].plot(t, p_gt)
        axs1[0].plot(t, p)
        axs1[1].plot(t, v_gt)
        axs1[1].plot(t, v)
        axs1[2].plot(t, v_r_gt)
        axs1[2].plot(t, v_r)

        axs2[0].plot(t, ang_gt)
        axs2[0].plot(t, ang)
        axs2[1].plot(t, b_omega)
        axs2[2].plot(t, b_acc)

        ax3.plot(p_gt[:, 0], p_gt[:, 1])
        ax3.plot(p[:, 0], p[:, 1])
        ax3.axis('equal')
        ax4.plot(p_gt[:, 0], p_gt[:, 1])
        ax4.plot(p_align[:, 0], p_align[:, 1])
        ax4.axis('equal')

        axs5[0].plot(t, np.log10(measurements_covs))
        axs5[1].plot(t, u_normalized[:, :3])
        axs5[2].plot(t, u_normalized[:, 3:])

        axs6[0].plot(t, u[:, :3])
        axs6[1].plot(t, u[:, 3:6])

        axs7[0].plot(t, mate_xy)
        axs7[0].plot(t, mate_z)
        axs7[0].plot(t, rmse_xy)
        axs7[0].plot(t, rmse_z)
        axs7[1].plot(t, cate_xy)
        axs7[1].plot(t, cate_z)
        axs7[2].plot(t, error_p)

        axs1[0].set(xlabel='time (s)', ylabel='$\mathbf{p}_n$ (m)', title="Position")
        axs1[1].set(xlabel='time (s)', ylabel='$\mathbf{v}_n$ (m/s)', title="Velocity")
        axs1[2].set(xlabel='time (s)', ylabel='$\mathbf{R}_n^T \mathbf{v}_n$ (m/s)',
                    title="Velocity in body frame")
        axs2[0].set(xlabel='time (s)', ylabel=r'$\phi_n, \theta_n, \psi_n$ (rad)',
                    title="Orientation")
        axs2[1].set(xlabel='time (s)', ylabel=r'$\mathbf{b}_{n}^{\mathbf{\omega}}$ (rad/s)',
                    title="Bias gyro")
        axs2[2].set(xlabel='time (s)', ylabel=r'$\mathbf{b}_{n}^{\mathbf{a}}$ (m/$\mathrm{s}^2$)',
                    title="Bias accelerometer")
        ax3.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Position on $xy$")
        ax4.set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title="Aligned position on $xy$")
        axs5[0].set(xlabel='time (s)', ylabel=r' $\mathrm{cov}(\mathbf{y}_{n})$ (log scale)',
                     title="Covariance on the zero lateral and vertical velocity measurements (log "
                           "scale)")
        axs5[1].set(xlabel='time (s)', ylabel=r'Normalized gyro measurements',
                     title="Normalized gyro measurements")
        axs5[2].set(xlabel='time (s)', ylabel=r'Normalized accelerometer measurements',
                   title="Normalized accelerometer measurements")
        axs6[0].set(xlabel='time (s)', ylabel=r'$\omega^x_n, \omega^y_n, \omega^z_n$ (rad/s)',
                    title="Gyrometer")
        axs6[1].set(xlabel='time (s)', ylabel=r'$a^x_n, a^y_n, a^z_n$ (m/$\mathrm{s}^2$)',
                    title="Accelerometer")
        axs7[0].set(xlabel='time (s)', ylabel=r'$|| \mathbf{p}_{n}-\hat{\mathbf{p}}_{n} ||$ (m)',
                    title="Mean Absolute Trajectory Error (MATE) and Root Mean Square Error (RMSE)")
        axs7[1].set(xlabel='time (s)',
                    ylabel=r'$\Sigma_{i=0}^{n} || \mathbf{p}_{i}-\hat{\mathbf{p}}_{i} ||$ (m)',
                    title="Cumulative Absolute Trajectory Error (CATE)")
        axs7[2].set(xlabel='time (s)', ylabel=r' $\mathbf{\xi}_{n}^{\mathbf{p}}$',
                    title="$SE(3)$ error on position")

        for ax in chain(axs1, axs2, axs5, axs6, axs7):
            ax.grid()
        ax3.grid()
        ax4.grid()
        axs1[0].legend(
            ['$p_n^x$', '$p_n^y$', '$p_n^z$', '$\hat{p}_n^x$', '$\hat{p}_n^y$', '$\hat{p}_n^z$'])
        axs1[1].legend(
            ['$v_n^x$', '$v_n^y$', '$v_n^z$', '$\hat{v}_n^x$', '$\hat{v}_n^y$', '$\hat{v}_n^z$'])
        axs1[2].legend(
            ['$v_n^x$', '$v_n^y$', '$v_n^z$', '$\hat{v}_n^x$', '$\hat{v}_n^y$', '$\hat{v}_n^z$'])
        axs2[0].legend([r'$\phi_n^x$', r'$\theta_n^y$', r'$\psi_n^z$', r'$\hat{\phi}_n^x$',
                        r'$\hat{\theta}_n^y$', r'$\hat{\psi}_n^z$'])
        axs2[1].legend(['$b_n^x$', '$b_n^y$', '$b_n^z$'])
        axs2[2].legend(['$b_n^x$', '$b_n^y$', '$b_n^z$'])
        ax3.legend(['ground-truth trajectory', 'proposed'])
        ax4.legend(['ground-truth trajectory', 'proposed'])
        axs5[0].legend(['zero lateral velocity', 'zero vertical velocity'])
        axs6[0].legend(['$\omega_n^x$', '$\omega_n^y$', '$\omega_n^z$'])
        axs6[1].legend(['$a_n^x$', '$a_n^y$', '$a_n^z$'])
        if u.shape[1] > 6:
            axs6[2].legend(['$m_n^x$', '$m_n^y$', '$m_n^z$'])
        axs7[0].legend(['MATE xy', 'MATE z', 'RMSE xy', 'RMSE z'])
        axs7[1].legend(['CATE xy', 'CATE z'])

        # save figures
        figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, ]
        figs_name = ["position_velocity", "orientation_bias", "position_xy", "position_xy_aligned",
                     "measurements_covs", "imu", "errors", "errors2"]
        for l, fig in enumerate(figs):
            fig_name = figs_name[l]
            fig.savefig(os.path.join(folder_path, fig_name + ".png"))

        # Non-blocking so headless / Agg runs finish; interactive backends still draw.
        plt.show(block=False)


def plot_before_divergence(args, dataset, speed_factor=1.5, horizons=(60, 300)):
    """Plot xy trajectory and error up to the first time the estimated speed exceeds
    speed_factor x the max ground-truth speed (plus fixed early horizons)."""
    for i in range(0, len(dataset.datasets)):
        plt.close('all')
        dataset_name = dataset.dataset_name(i)
        if not os.path.exists(os.path.join(dataset.path_results, dataset_name + "_filter.p")):
            continue
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs = dataset.get_estimates(
            dataset_name)
        t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)
        t = (t - t[0]).cpu().numpy()
        p_gt = (p_gt - p_gt[0]).cpu().numpy()
        v_gt = v_gt.cpu().numpy()

        speed_gt = np.linalg.norm(v_gt, axis=1)
        diverged = np.where(np.linalg.norm(v, axis=1) > speed_factor * speed_gt.max())[0]
        n_div = int(diverged[0]) if len(diverged) else len(t)
        n_last = max(n_div - 1, 0)
        err_xy = np.linalg.norm(p_gt[:, :2] - p[:, :2], axis=1)
        dist_gt = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p_gt[:, :2], axis=0), axis=1))])

        windows = [(int(np.searchsorted(t, h)), 'first {:.0f} s'.format(h)) for h in horizons if h < t[n_last]]
        windows.append((n_div, 'until divergence (t = {:.0f} s)'.format(t[n_last])))
        fig, axs = plt.subplots(2, len(windows), figsize=(20, 10), squeeze=False)
        for j, (n, label) in enumerate(windows):
            axs[0, j].plot(p_gt[:n, 0], p_gt[:n, 1])
            axs[0, j].plot(p[:n, 0], p[:n, 1])
            axs[0, j].axis('equal')
            axs[0, j].set(xlabel=r'$p_n^x$ (m)', ylabel=r'$p_n^y$ (m)', title='Position on $xy$, ' + label)
            axs[0, j].legend(['ground-truth trajectory', 'proposed'])
            axs[1, j].plot(t[:n], err_xy[:n])
            axs[1, j].set(xlabel='time (s)', ylabel=r'$|| \mathbf{p}^{xy}_{n}-\hat{\mathbf{p}}^{xy}_{n} ||$ (m)',
                          title='xy error (GT distance {:.0f} m)'.format(dist_gt[max(n - 1, 0)]))
            for ax in axs[:, j]:
                ax.grid()
        fig.tight_layout()
        folder_path = os.path.join(args.path_results, dataset_name)
        create_folder(folder_path)
        fig.savefig(os.path.join(folder_path, "position_xy_before_divergence.png"))


def _stationary_segments(t, stationary):
    """Return list of (t_start, t_end) for contiguous True runs in stationary."""
    if len(stationary) == 0 or not np.any(stationary):
        return []
    padded = np.concatenate([[False], stationary, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return [(float(t[s]), float(t[e])) for s, e in zip(starts, ends)]


def _shade_stationary(ax, segments, label_once=True):
    """GPS stops as a bar along the top of the axes, not a full-height span."""
    labeled = False
    for t0, t1 in segments:
        ax.axvspan(t0, t1, ymin=0.92, ymax=1.0, color='0.55', zorder=0, linewidth=0,
                   label='stationary (GPS)' if label_once and not labeled else None)
        labeled = True


def _mark_zupt(ax, t, zupt, label_once=True):
    """ZUPT stops as a purple bar just under the GPS bar."""
    if zupt is None or not np.any(zupt):
        return
    labeled = False
    for t0, t1 in _stationary_segments(t, zupt):
        ax.axvspan(t0, t1, ymin=0.84, ymax=0.92, color='#7E57C2', zorder=0, linewidth=0,
                   label='stationary (ZUPT)' if label_once and not labeled else None)
        labeled = True


def _zupt_mask(dataset, dataset_name, n):
    """IMU zero-velocity flags saved with the filter result. None if this run had no ZUPT."""
    file_name = os.path.join(dataset.path_results, dataset_name + "_filter.p")
    zupt = dataset.load(file_name).get('zupt')
    if zupt is None:
        return None
    zupt = np.asarray(zupt, dtype=bool)
    if len(zupt) == n:
        return zupt
    out = np.zeros(n, dtype=bool)
    m = min(len(zupt), n)
    out[:m] = zupt[:m]
    return out


def _mark_both_paths(ax, p_gt, p, i0, i1, **kwargs):
    """One stop marker on the GPS path and the same instant on the KF path."""
    ax.scatter(p_gt[i0:i1 + 1, 0].mean(), p_gt[i0:i1 + 1, 1].mean(), **kwargs)
    kwargs = dict(kwargs)
    kwargs['label'] = None
    ax.scatter(p[i0:i1 + 1, 0].mean(), p[i0:i1 + 1, 1].mean(), **kwargs)


def _mask_spans(mask):
    """Index pairs (start, end) of contiguous True runs. end is inclusive."""
    if len(mask) == 0 or not np.any(mask):
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts, ends))


def plot_stationary_periods(args, dataset, speed_thresh=0.5, speed_factor=1.5):
    """GPS speed (GT), raw IMU, and KF speed/trajectory with GPS-stationary periods shaded."""
    for i in range(0, len(dataset.datasets)):
        plt.close('all')
        dataset_name = dataset.dataset_name(i)
        if not os.path.exists(os.path.join(dataset.path_results, dataset_name + "_filter.p")):
            continue
        Rot, v, p, b_omega, b_acc, Rot_c_i, t_c_i, measurements_covs = dataset.get_estimates(
            dataset_name)
        t, ang_gt, p_gt, v_gt, u = dataset.get_data(dataset_name)
        t = (t - t[0]).cpu().numpy()
        p_gt = (p_gt - p_gt[0]).cpu().numpy()
        v_gt = v_gt.cpu().numpy()
        u = u.cpu().numpy()

        speed_gt = np.linalg.norm(v_gt, axis=1)
        speed_est = np.linalg.norm(v, axis=1)
        stationary = speed_gt < speed_thresh
        zupt = _zupt_mask(dataset, dataset_name, len(t))
        segments = _stationary_segments(t, stationary)
        diverged = np.where(speed_est > speed_factor * max(speed_gt.max(), 1e-6))[0]
        n_div = int(diverged[0]) if len(diverged) else len(t)

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(20, 14))
        _shade_stationary(axs[0], segments)
        _mark_zupt(axs[0], t, zupt)
        axs[0].plot(t, speed_gt, label='GPS speed (GT)')
        axs[0].plot(t, speed_est, label='KF speed')
        axs[0].axhline(speed_thresh, color='k', linestyle='--', linewidth=0.8,
                       label='stationary thresh ({:.1f} m/s)'.format(speed_thresh))
        if n_div < len(t):
            axs[0].axvline(t[n_div], color='r', linestyle=':', linewidth=1.0, label='KF divergence')
        axs[0].set(ylabel='speed (m/s)', title='{}: speed and stationary periods'.format(dataset_name))
        axs[0].legend(loc='upper right')

        _shade_stationary(axs[1], segments, label_once=False)
        _mark_zupt(axs[1], t, zupt, label_once=False)
        axs[1].plot(t, u[:, 0], label=r'$\omega_x$')
        axs[1].plot(t, u[:, 1], label=r'$\omega_y$')
        axs[1].plot(t, u[:, 2], label=r'$\omega_z$')
        axs[1].set(ylabel='rad/s', title='Raw gyro (car frame)')
        axs[1].legend(loc='upper right')

        _shade_stationary(axs[2], segments, label_once=False)
        _mark_zupt(axs[2], t, zupt, label_once=False)
        axs[2].plot(t, u[:, 3], label=r'$a_x$')
        axs[2].plot(t, u[:, 4], label=r'$a_y$')
        axs[2].plot(t, u[:, 5], label=r'$a_z$')
        axs[2].set(ylabel=r'm/s$^2$', title='Raw accelerometer (car frame)')
        axs[2].legend(loc='upper right')

        _shade_stationary(axs[3], segments, label_once=False)
        _mark_zupt(axs[3], t, zupt, label_once=False)
        axs[3].plot(t, v_gt[:, 0], label=r'GPS $v_e$')
        axs[3].plot(t, v_gt[:, 1], label=r'GPS $v_n$')
        axs[3].plot(t, v[:, 0], label=r'KF $\hat{v}_e$')
        axs[3].plot(t, v[:, 1], label=r'KF $\hat{v}_n$')
        axs[3].set(xlabel='time (s)', ylabel='m/s', title='Horizontal velocity components')
        axs[3].legend(loc='upper right')
        for ax in axs:
            ax.grid()
        fig.tight_layout()

        # Lines stay blue and orange. Each stop is marked on both paths.
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        n = n_div
        ax2.plot(p_gt[:n, 0], p_gt[:n, 1], color='C0', label='GPS / GT trajectory')
        ax2.plot(p[:n, 0], p[:n, 1], color='C1', label='KF trajectory')
        saw_gps = False
        for s, e in _mask_spans(stationary[:n]):
            if t[min(e, n - 1)] - t[s] < 1.0:
                continue
            _mark_both_paths(ax2, p_gt[:n], p[:n], s, e,
                             s=160, c='0.35', alpha=0.35, zorder=4,
                             label='stationary (GPS)' if not saw_gps else None)
            saw_gps = True
        saw_zupt = False
        if zupt is not None:
            for s, e in _mask_spans(zupt[:n]):
                if t[min(e, n - 1)] - t[s] < 1.0:
                    continue
                _mark_both_paths(ax2, p_gt[:n], p[:n], s, e,
                                 s=36, c='#7E57C2', zorder=5,
                                 label='stationary (ZUPT)' if not saw_zupt else None)
                saw_zupt = True
        ax2.axis('equal')
        ax2.set(xlabel=r'$p^x$ (m)', ylabel=r'$p^y$ (m)',
                title='Trajectory until divergence (t = {:.0f} s); stationary marked'.format(
                    t[max(n - 1, 0)]))
        ax2.legend(loc='best')
        ax2.grid()
        fig2.tight_layout()

        folder_path = os.path.join(args.path_results, dataset_name)
        create_folder(folder_path)
        fig.savefig(os.path.join(folder_path, "stationary_periods.png"))
        fig2.savefig(os.path.join(folder_path, "stationary_trajectory.png"))
        print('  stationary: {:.1f}% of samples, {} segments'.format(
            100.0 * stationary.mean(), len(segments)))



import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

from matplotlib.animation import FuncAnimation

plt.rcParams["animation.html"] = "jshtml"


def animate_antialiasing():
    f_sig = 2.2      # Медленное вращение (2.2 оборота в сек)
    f_s = 2.0        # Очень медленная камера (2 кадра в сек)
    duration = 10.0  # Увеличим длительность, чтобы рассмотреть медленный алиасинг
    tau = 1.0 / f_s  # Выдержка 0.5 сек

    fps_display = 60
    t_steps = np.linspace(0, duration, int(duration * fps_display))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    plt.close()

    def draw_fan(ax, angle, alpha=1.0, color='blue', label=''):
        ax.clear()
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label)

        for i in range(3):
            theta = angle + i * (2 * np.pi / 3)
            ax.plot(
                [0, np.cos(theta)], [0, np.sin(theta)],
                lw=10, color=color, alpha=alpha, solid_capstyle='round'
            )

    def update(frame_time):
        angle_ideal = 2 * np.pi * f_sig * frame_time
        draw_fan(ax1, angle_ideal, color='red', label=f'Оригинал: {f_sig} Гц')

        t_sample = (frame_time // (1 / f_s)) * (1 / f_s)
        angle_sampled = 2 * np.pi * f_sig * t_sample
        draw_fan(
            ax2, angle_sampled, color='red',
            label=f'Дискретизация: {f_s} FPS\nЛожная частота: {round(abs(f_sig-f_s),1)} Гц'
        )

        ax3.clear()
        ax3.set_xlim(-1.2, 1.2); ax3.set_yticks([]); ax3.set_xticks([])
        ax3.set_aspect('equal')
        ax3.set_title('Апертурный эффект')

        num_blur_steps = 15
        for sub in np.linspace(0, tau, num_blur_steps):
            t_sub = t_sample + sub
            angle_sub = 2 * np.pi * f_sig * t_sub
            for i in range(3):
                theta = angle_sub + i * (2 * np.pi / 3)
                ax3.plot(
                    [0, np.cos(theta)], [0, np.sin(theta)],
                    lw=10, color='red', alpha=1.2/num_blur_steps, solid_capstyle='round'
                )

    anim = FuncAnimation(fig, update, frames=t_steps, interval=1000 / fps_display)
    plt.close()
    return anim


def advanced_sampling_demo(f_sig, f_sample, T_view=2.0):
    dt, nyquist_freq, analog_res = 1 / f_sample, f_sample / 2, 2000
    
    T_generate = T_view * 1.5 

    # s(t) = exp(-10*t^2) * cos(2 * pi * f * t)
    t_analog = np.linspace(-T_view / 2, T_view / 2, analog_res)
    s_analog = np.exp(-10 * t_analog ** 2) * np.cos(2 * np.pi * f_sig * t_analog)
    
    n_samples = np.arange(np.ceil(-T_generate / 2 / dt), np.floor(T_generate / 2 / dt) + 1)
    t_samples = n_samples * dt
    s_samples = np.exp(-10 * t_samples ** 2) * np.cos(2 * np.pi * f_sig * t_samples)
    
    t_diff = (t_analog[:, None] - t_samples[None, :]) / dt
    
    # np.sinc(x) = sin(pi * x) / (pi * x)
    sinc_matrix = np.sinc(t_diff)
    
    weighted_sincs = sinc_matrix * s_samples[None, :]
    s_reconstructed = np.sum(weighted_sincs, axis=1)
    
    error = s_analog - s_reconstructed
    mse = np.mean(error ** 2)

    N_fft = len(t_analog)
    yf = fft(s_analog) / N_fft 
    xf = fftfreq(N_fft, T_view / analog_res)
    yf = fftshift(yf)
    xf = fftshift(xf)
    
    mag_spectrum = np.abs(yf)
    if np.max(mag_spectrum) > 0:
        mag_spectrum /= np.max(mag_spectrum)
    
    fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1.5, 1.5])
    
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_analog, s_analog, color='black', alpha=0.3, linewidth=4, label='Original Analog')
    ax1.plot(t_analog, s_reconstructed, color='#1f77b4', linewidth=2, linestyle='--', label='Reconstructed')
    
    # Рисуем только те сэмплы, которые попадают в окно просмотра (для красоты),
    # но помним, что в расчетах участвовали и внешние!
    mask_view = (t_samples >= -T_view / 2) & (t_samples <= T_view / 2)
    ax1.stem(
        t_samples[mask_view], s_samples[mask_view], linefmt='k-', markerfmt='ro', basefmt=" ", label='Samples (View area)'
    )
    
    ax1.set_title(
        f"Freq: {f_sig:.1f} Hz | SampleRate: {f_sample:.1f} Hz | MSE Error: {mse:.2e}",
        fontsize=14, color='#d62728' if f_sig > nyquist_freq else '#2ca02c', fontweight='bold'
    )
    ax1.legend(loc='upper right')
    ax1.set_xlim(-T_view/2, T_view/2)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t_analog, error, color='#d62728', linewidth=1)
    ax2.fill_between(t_analog, error, color='#d62728', alpha=0.1)
    ax2.set_ylabel("Error")
    ax2.set_title("Reconstruction Error (Difference)", fontsize=10)
    ax2.grid(True, alpha=0.5)
    ax2.set_xlim(-T_view/2, T_view/2)
    max_err = max(0.05, np.max(np.abs(error))*1.1)
    ax2.set_ylim(-max_err, max_err)

    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(xf, mag_spectrum, color='green', alpha=0.6, label='Spectrum')
    
    for k in [-1, 1]:
        center = k * f_sample
        ax3.fill_between(xf + center, mag_spectrum, color='red', alpha=0.2, label='Alias Image' if k==1 else "")
    
    ax3.axvline(nyquist_freq, color='blue', linestyle='--', label='Nyquist (fs/2)')
    ax3.axvline(-nyquist_freq, color='blue', linestyle='--')
    
    ax3.set_xlim(-max(20, f_sample*1.5), max(20, f_sample*1.5))
    ax3.set_title("Frequency Domain", fontsize=10)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
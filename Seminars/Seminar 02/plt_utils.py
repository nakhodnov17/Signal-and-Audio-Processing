import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["animation.html"] = "jshtml"


def animate_aliasing_fixed():
    Nx, Nh = 10, 10
    N_buffer = 10
    
    x = np.ones(Nx)
    h = np.linspace(1.0, 0.1, Nh)
    
    # Calculations
    y_lin = np.convolve(x, h, mode='full')
    L_lin = len(y_lin)
    
    # Circular convolution via FFT
    x_padded = np.zeros(N_buffer); x_padded[:Nx] = x[:min(Nx, N_buffer)]
    h_padded = np.zeros(N_buffer); h_padded[:Nh] = h[:min(Nh, N_buffer)]
    y_circ = np.real(np.fft.ifft(np.fft.fft(x_padded) * np.fft.fft(h_padded)))

    # 2. Figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [2, 1]},
        sharex='col', sharey='col'
    )
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    def update(n):
        for ax in [ax1, ax2, ax3, ax4]: ax.clear()

        # --- LINEAR CONVOLUTION (TOP) ---
        ax1.set_title(f"Linear: Kernel shift h[n-k] (step n={n})")
        ax1.set_xlim(-Nh, L_lin)
        ax1.set_ylim(-0.1, 1.2)
        ax1.grid(True, alpha=0.2)
        
        # x[k]
        ax1.stem(np.arange(Nx), x, linefmt='b-', markerfmt='bo', basefmt=' ', label='x[k]')
        # h[n-k]
        h_indices = np.arange(n - Nh + 1, n + 1)
        ax1.stem(h_indices, h[::-1], linefmt='r-', markerfmt='rs', basefmt=' ', label='h[n-k]')
        ax1.legend(loc='upper left', fontsize='small')

        ax2.set_title(f"Linear result y_lin[{n}]")
        ax2.set_xlim(-1, max(L_lin, N_buffer))
        ax2.set_ylim(-0.1, np.max(y_lin) * 1.1)
        ax2.grid(True, alpha=0.2)
        
        # Display accumulated result of linear convolution
        limit_lin = min(n + 1, L_lin)
        idx_lin = np.arange(limit_lin)
        ax2.stem(idx_lin, y_lin[idx_lin], linefmt='g-', markerfmt='go', basefmt='k-')

        # --- CIRCULAR CONVOLUTION (BOTTOM) ---
        ax3.set_title(f"Circular: Rotation in buffer N={N_buffer}")
        ax3.set_xlim(-1, max(L_lin, N_buffer))
        ax3.set_ylim(-0.1, 1.2)
        ax3.axvspan(0, N_buffer-1, color='gray', alpha=0.1, label='Buffer boundary')
        
        # x[k] in buffer
        ax3.stem(np.arange(N_buffer), x_padded, linefmt='b-', markerfmt='bo', basefmt=' ')
        
        # h[(n-k)%N] (rotation)
        n_mod = n % N_buffer
        idx_circ = np.arange(N_buffer)
        h_rot = h_padded[(n_mod - idx_circ) % N_buffer]
        ax3.stem(idx_circ, h_rot, linefmt='r-', markerfmt='rs', basefmt=' ')
        
        # Mark if tail started "wrapping around"
        if n >= N_buffer:
            ax3.text(N_buffer/2, 1.05, "TAIL WRAPPED AROUND", color='red', ha='center', weight='bold')

        ax4.set_title("Circular result y_circ")
        ax4.set_xlim(-1, max(L_lin, N_buffer))
        ax4.set_ylim(-0.1, np.max(y_circ) * 1.1)
        ax4.grid(True, alpha=0.2)
        
        # Display result of circular convolution
        # (in circular convolution values change by "jumps" after n >= N_buffer)
        limit_circ = min(n + 1, N_buffer)
        idx_circ_res = np.arange(limit_circ)
        
        # Before aliasing (n < N_buffer) draw current values
        # After - draw final buffer values to see the jump
        if n < N_buffer:
            ax4.stem(idx_circ_res, y_circ[idx_circ_res], linefmt='C1-', markerfmt='C1s', basefmt='k-')
        else:
            ax4.stem(np.arange(N_buffer), y_circ, linefmt='C1-', markerfmt='C1s', basefmt='k-')
            # Highlight which index just received an "addition" from the future
            alias_idx = n % N_buffer
            ax4.plot([alias_idx], [y_circ[alias_idx]], 'ro', markersize=12, fillstyle='none', markeredgewidth=2)

    # Animation until end of linear convolution + a few pause frames
    anim = FuncAnimation(fig, update, frames=range(L_lin + 2), interval=500)
    plt.close()
    return anim


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
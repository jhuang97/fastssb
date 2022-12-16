import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib_scalebar.scalebar import ScaleBar
from ipywidgets import AppLayout, FloatSlider, GridspecLayout, VBox, HBox, IntText, Layout, Label
import ipywidgets as widgets
from .optics import disk_overlap_function, single_sideband_reconstruction


def cx_to_hsv_img(cin, vmin=None, vmax=None, vgamma=None):
    # HSV channels
    h = .5 * np.angle(cin) / np.pi + .5
    s = np.ones(cin.shape)

    v = np.abs(cin)
    if vmin is None: vmin = 0.
    if vmax is None: vmax = v.max()
    assert vmin < vmax
    v = (v.clip(vmin, vmax) - vmin) / (vmax - vmin)
    if vgamma is not None:
        v = v**vgamma

    i = (6. * h).astype(int)
    f = (6. * h) - i
    p = v * (1. - s)
    q = v * (1. - s * f)
    t = v * (1. - s * (1. - f))
    i0 = (i % 6 == 0)
    i1 = (i == 1)
    i2 = (i == 2)
    i3 = (i == 3)
    i4 = (i == 4)
    i5 = (i == 5)

    imout = np.zeros(h.shape + (3,), dtype=h.dtype)
    imout[:, :, 0] = (i0 * v + i1 * q + i2 * p + i3 * p + i4 * t + i5 * v)
    imout[:, :, 1] = (i0 * t + i1 * v + i2 * v + i3 * q + i4 * p + i5 * p)
    imout[:, :, 2] = (i0 * p + i1 * p + i2 * t + i3 * v + i4 * v + i5 * q)
    
    return imout


def mosaic(data):
    n, w, h = data.shape
    diff = np.sqrt(n) - int(np.sqrt(n))
    s = np.sqrt(n)
    m = int(s)
    # print 'm', m
    if diff > 1e-6: m += 1
    mosaic = np.zeros((m * w, m * h)).astype(data.dtype)
    for i in range(m):
        for j in range(m):
            if (i * m + j) < n:
                mosaic[i * w:(i + 1) * w, j * h:(j + 1) * h] = data[i * m + j]
    return mosaic


def fft_doverlap_figure(masked_Gabs, Qmax, fft_intensities, Gresid, G_max, Gamma, figsize=(18,12)):
	plt.ion()

	fig, ax = plt.subplots(2,3, layout="compressed", figsize=figsize)
	ax = ax.flatten()

	im = ax[0].imshow(masked_Gabs, norm=colors.LogNorm(), cmap=plt.cm.get_cmap('bone'))
	ax[0].set_title('Summed FFT')
	ax[0].set_xticks([])
	ax[0].set_yticks([])

	ax[1].plot(Qmax[0], Qmax[1], '.')
	ax[1].plot(-Qmax[0], -Qmax[1], '.')
	ax[1].axis('equal')
	ax[1].invert_yaxis()
	ax[1].set_title('Strongest frequencies')

	ax[2].plot(np.arange(len(fft_intensities)), fft_intensities/fft_intensities[0], '.-')
	ax[2].set_title('strength of object frequencies')

	im = ax[3].imshow(cx_to_hsv_img(mosaic(Gamma)))
	ax[3].set_title('overlap function')
	ax[3].set_xticks([])
	ax[3].set_yticks([])

	im = ax[4].imshow(cx_to_hsv_img(mosaic(G_max), vgamma=0.5))
	ax[4].set_title('FFT')
	ax[4].set_xticks([])
	ax[4].set_yticks([])

	im = ax[5].imshow(cx_to_hsv_img(mosaic(Gresid), vgamma=0.5))
	ax[5].set_title('FFT * overlap function')
	ax[5].set_xticks([])
	ax[5].set_yticks([])

	# plt.tight_layout()

	return fig, ax


def manual_aberration_ui(G, G_sel, C, C_gui, dp_angle_arr, 
                         Psi_Qp, Psi_Qp_left_sb, Psi_Qp_right_sb, 
                         Qx1d, Qy1d, Qx_sel, Qy_sel, Kx, Ky, eps,
                         aberrations, alpha_max, lam, dr,
                        C_max=None, C_exp=None):
    plt.close('all')
    plt.ioff()
    
    from cupyx.scipy.fft import fft2, ifft2
    
    ab_coeffs  = ['C1', 'C12', 'C21', 'C23', 'C3', 'C32', 'C34']  # aberrations according to Krivanek
    ab_uhlemann= ['C1', 'A1', '3 B2*', 'A2', 'C3','4 S3*', 'A3']  # aberrations according to Uhlemann & Haider
    ab_isreal  = [True, False, False, False, True, False, False]
    if C_max is None:
        C_max  = [20,   10,    20,      10,    20,   30,    30]
        
    if C_exp is None:
        C_exp = np.array([1,1, 3,3, 4,4,4])
    elif type(C_exp) is not np.ndarray:
        C_exp = np.array(C_exp)
    
    ab_indices = []
    ab_i = 0
    for isreal in ab_isreal:
        ab_indices.append(ab_i)
        ab_i += 1 if isreal else 2
    c_indices = [0] * ab_i
    for idx, isreal in enumerate(ab_isreal):
        if isreal:
            c_indices[ab_indices[idx]] = idx
        else:
            c_indices[ab_indices[idx]] = idx
            c_indices[ab_indices[idx]+1] = idx

    left_panel_width = '520px'
    slider_width = '400px'
    
    Cslider_box = VBox(layout=Layout(width=left_panel_width))
    children= []
    sliders = []

    text = widgets.HTML(
        value="1",
        placeholder='',
        description='',
    )

    overlaps_output = widgets.Output()

    Psi_Rp = ifft2(Psi_Qp, norm="ortho")
    Psi_Rp_left_sb = ifft2(Psi_Qp_left_sb, norm="ortho")
    Psi_Rp_right_sb = ifft2(Psi_Qp_right_sb, norm="ortho")

    Gamma = disk_overlap_function(Qx_sel, Qy_sel, Kx, Ky, aberrations, dp_angle_arr[0], alpha_max, lam)
    gg = Gamma.conjugate() * G_sel

    overlap_figure_axes = []
    overlap_figure2_axes = []
    Gmax_figure_axes = []

    with overlaps_output:
        overlap_figure = plt.figure(constrained_layout=True,figsize=(3,3))
        gs1 = overlap_figure.add_gridspec(3, 3, wspace=0.03,hspace=0.03)
        for dd, ggs in zip(gg[:9], gs1):
            f3_ax1 = overlap_figure.add_subplot(ggs)
            imax2 = f3_ax1.imshow(cx_to_hsv_img(dd.get(), vgamma=0.5))
            f3_ax1.set_xticks([])
            f3_ax1.set_yticks([])
            overlap_figure_axes.append(imax2)

        overlap_figure2 = plt.figure(constrained_layout=True,figsize=(6,3))
        gs2 = overlap_figure2.add_gridspec(3, 6, wspace=0.03,hspace=0.03)
        for i, (dd, doverlap) in enumerate(zip(Gamma[:9], G_sel[:9])):
            f3_ax1 = overlap_figure2.add_subplot(gs2[2*i])
            imax2 = f3_ax1.imshow(cx_to_hsv_img(dd.get()))
            f3_ax1.set_xticks([])
            f3_ax1.set_yticks([])
            overlap_figure2_axes.append(imax2)

            f3_ax1 = overlap_figure2.add_subplot(gs2[2*i+1])
            imax2 = f3_ax1.imshow(cx_to_hsv_img(doverlap.get(), vgamma=0.5))
            f3_ax1.set_xticks([])
            f3_ax1.set_yticks([])
            Gmax_figure_axes.append(imax2)


    plot_box = overlap_figure.canvas
    plot_box2 = overlap_figure2.canvas

    recon_fig, recon_axes = plt.subplots(constrained_layout=True,figsize=(3.2,3.2))
    m = 5
    img = np.angle(Psi_Rp_left_sb.get()[m:-m,m:-m])
    recon_img = recon_axes.imshow(img, cmap=plt.get_cmap('bone'))
    recon_axes.set_xticks([])
    recon_axes.set_yticks([])
    scalebar = ScaleBar(dr/10,'nm') # 1 pixel = 0.2 meter
    recon_axes.add_artist(scalebar)
    plt.tight_layout()


    def update_everything():
        Psi_Qp[:] = 0
        Psi_Qp_left_sb[:] = 0
        Psi_Qp_right_sb[:] = 0
        single_sideband_reconstruction(
            G,
            Qx1d,
            Qy1d,
            Kx,
            Ky,
            C,
            dp_angle_arr[0],
            alpha_max,
            Psi_Qp,
            Psi_Qp_left_sb,
            Psi_Qp_right_sb,
            eps,
            lam,
        )
        crop_pixels = 5

        Psi_Rp[:] = ifft2(Psi_Qp, norm="ortho")
        Psi_Rp_left_sb[:] = ifft2(Psi_Qp_left_sb, norm="ortho")
        Psi_Rp_right_sb[:] = ifft2(Psi_Qp_right_sb, norm="ortho")

        img = np.angle(Psi_Rp_left_sb.get()[crop_pixels:-crop_pixels,crop_pixels:-crop_pixels])
        recon_img.set_data(img)
        recon_img.set_clim(img.min(),img.max())
        recon_fig.canvas.draw()
        recon_fig.canvas.flush_events()

        Gamma = disk_overlap_function(Qx_sel, Qy_sel, Kx, Ky, C, dp_angle_arr[0], alpha_max, lam)
        gg = Gamma.conjugate() * G_sel
        for ax, ggg in zip(overlap_figure_axes,gg):
            ax.set_data(cx_to_hsv_img(ggg.get(), vgamma=0.5))
        for ax, ggg in zip(overlap_figure2_axes,Gamma):
            ax.set_data(cx_to_hsv_img(ggg.get()))

        overlap_figure.canvas.draw()
        overlap_figure2.canvas.draw()
        overlap_figure.canvas.flush_events()
        overlap_figure2.canvas.flush_events()



    def create_ab_function(name, i, is_real, is_magnitude):
        def func1(change):
            update_str = ''
            c_idx = c_indices[i]
            if is_magnitude:
                C_exp[c_idx] = change['new']
                update_str += f"e{change['new']:d} "
            else:
                C_gui[i] = change['new']
            multiplier = 10**C_exp[c_idx]
            if is_real:
                C[i] = C_gui[i] * multiplier
                update_str += f'{C_gui[i]}, {C[i]}'
            else:
                ab_idx = ab_indices[c_idx]
                cx_val = multiplier * C_gui[ab_idx] * np.exp(1j * np.pi / 180 * C_gui[ab_idx+1])
                C[ab_idx] = np.real(cx_val)
                C[ab_idx+1] = np.imag(cx_val)
                update_str += f'{C_gui[i]}, {cx_val:.3f}'
            update_everything()
            text.value = update_str
        func1.__name__ = name
        return func1


    def create_angle_function():
        def func1(change):
            dp_angle_arr[0] = change['new']/180 * np.pi
            update_everything()
            text.value = f'{dp_angle_arr[0]}'
        func1.__name__ = 'dp_angle_slider_changed'
        return func1
    
    
    def setup_scalar_with_magnitude(c_i, ab_name, init_val, max_val, height_str, ab_i, is_real):
        s = FloatSlider(description=ab_name, value=init_val, min=-max_val, max=max_val, readout_format='.1f',
                            layout=widgets.Layout(width=slider_width, height=height_str))
        s.observe(create_ab_function(f'ab_slider_changed_{ab_i}', ab_i, is_real, False), names='value')
        
        magnitude_text = IntText(
            value=C_exp[c_i],
            description='x 10^',
            disabled=False,
            layout=Layout(width='100px', height='30px'),
            style={'description_width': 'initial'}
        )
        magnitude_text.observe(create_ab_function(f'ab_magnitude_slider_changed_{ab_i}', ab_i, is_real, True), names='value')
        return s, magnitude_text

        
    ab_i = 0
    for k, (ab_name, isreal, maxs, cexp) in enumerate(zip(ab_uhlemann, ab_isreal, C_max, C_exp)):
        sbox_children = []
        if isreal:
            scalar_slider, mag_text = setup_scalar_with_magnitude(k, ab_name, C_gui[ab_i], maxs, '40px', ab_i, isreal)
            sbox_children.append(scalar_slider)
            ab_i += 1
        else:
            scalar_slider, mag_text = setup_scalar_with_magnitude(k, ab_name, C_gui[ab_i], maxs, '10px', ab_i, isreal)
            sbox_children.append(scalar_slider)
            ab_i += 1
            
            angle_slider = FloatSlider(description='angle', value=C_gui[ab_i], min=-180, max=180, step=1.0, readout_format='.0f',
                            layout=widgets.Layout(width=slider_width, height='25px'))
            angle_slider.observe(create_ab_function(f'ab_slider_changed_{ab_i}', ab_i, False, False), names='value')
            sbox_children.append(angle_slider)
            ab_i += 1
        sliders_box = VBox(sbox_children)
        sliders.append(HBox([sliders_box, mag_text], 
                            layout=Layout(align_items='center', border_bottom='1px solid rgb(200,200,200)')))


    sdp = FloatSlider(description='dp angle', value=dp_angle_arr[0]*180/np.pi, min=-180, max=180, readout_format='.1f',
                      layout=widgets.Layout(width=slider_width, height='45px'))
    sdp.observe(create_angle_function(), names='value')
    sliders.append(sdp)

    Cslider_box.children = sliders + [text]
    
    return HBox([Cslider_box, 
                 VBox([
                     HBox([plot_box, recon_fig.canvas]), 
                     plot_box2
                 ], layout=Layout(width='50%'))
                ])
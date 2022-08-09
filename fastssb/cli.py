import numpy as np
import cupy as cp
import cupyx.scipy.fft as cpfft
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable

import time
from argparse import ArgumentParser
import os
from pathlib import Path

from . import data4d as d4
from .optics import wavelength, get_qx_qy_1D, disk_overlap_function, single_sideband_reconstruction
from . import plotting

import tifffile


def plot_virtual_images(d, metadata, radius, scan_number):
    abf = d.virtual_annular_image(radius/2, radius, d.frame_dimensions/2)
    bf = d.virtual_annular_image(0, radius/2, d.frame_dimensions/2)
    eabf = abf - bf
    adf = d.virtual_annular_image(radius, d.frame_dimensions[0]/2, d.frame_dimensions/2)

    bf[bf==0] = bf.mean()
    abf[abf==0] = abf.mean()

    fig, ax = plt.subplots(dpi=150)
    im = ax.imshow(abf, cmap= plt.cm.get_cmap('bone'))
    ax.set_title(f'Scan {scan_number} ABF')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_artist(ScaleBar(metadata.dr[0]/10,'nm'))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    fig, ax = plt.subplots(dpi=150)
    im = ax.imshow(bf, cmap= plt.cm.get_cmap('bone'))
    ax.set_title(f'Scan {scan_number} BF')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_artist(ScaleBar(metadata.dr[0]/10,'nm'))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    fig, ax = plt.subplots(dpi=150)
    im = ax.imshow(adf, cmap= plt.cm.get_cmap('bone'))
    ax.set_title(f'Scan {scan_number} ADF')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_artist(ScaleBar(metadata.dr[0]/10,'nm'))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()


def main():
    parser = ArgumentParser(prog='fastssb')
    parser.add_argument(
        'sparse_file', 
        type=Path,
        help='Input sparse 4D STEM data set')
    parser.add_argument(
        'adf_file', 
        type=Path,
        help='ADF file containing metadata')
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path('./'),
        help="Directory to save results",
    )
    args = parser.parse_args()

    # data_dir = 'C:\\Jeffrey\\Code\\fastssb_dev\\realtime_ptycho_example_data'
    scan_number = 147

    # base_path = Path(data_dir)
    # adfpath = base_path
    # sparse_path = base_path  
    results_path = args.out_dir

    if not results_path.exists():
        results_path.mkdir()
        
    # filename4d = sparse_path / f'data_scan{scan_number}_th4.0_electrons.h5'
    # filenameadf = adfpath / f'scan{scan_number}.dm4'
    filename4d = args.sparse_file
    filenameadf = args.adf_file

    alpha_max_factor = 1.2
    alpha_max_factor = 1.05

    print('1: data loading')
    # d = Sparse4DData.from_4Dcamera_file(filename4d)
    d = d4.Sparse4DData.from_4Dcamera_file(filename4d, (5,292), 192)  # use stempy instead????
    metadata = d4.Metadata4D.from_dm4_file(filenameadf)

    metadata.alpha_rad = 25e-3
    metadata.rotation_deg = 0
    metadata.wavelength =  wavelength(metadata.E_ev)  

    # center, radius = d.determine_center_and_radius(manual=False, size=200) 
    center, radius = d.determine_center_and_radius(manual=False, size=70)  # replace with stempy function for finding center of mass
    print(f'center: {center}')
    print(f'radius: {radius}')
    print('2: cropping')
    d.crop_symmetric_center_(center, radius*alpha_max_factor)
    print('3: sum diffraction pattern')
    s = d.sum_diffraction() # stempy
    print('4: plotting')

    f,ax = plt.subplots(1,2,figsize=(8,4))
    imax = ax[0].imshow(s)
    ax[0].set_title(f'Scan {scan_number} sum after cropping')
    imax = ax[1].imshow(np.log10(s+1))
    ax[1].set_title(f'Scan {scan_number} log10(sum) after cropping')
    plt.colorbar(imax)
    plt.tight_layout()

    if False:
        plot_virtual_images(d, metadata, radius, scan_number)


    dwell_time = 1/87e3
    detector_to_real_fluence_80kv = 1 

    fluence = d.fluence(metadata.dr[0]) * detector_to_real_fluence_80kv
    flux = d.flux(metadata.dr[0], dwell_time) * detector_to_real_fluence_80kv

    print(f"E               = {metadata.E_ev/1e3}             keV")
    print(f"λ               = {metadata.wavelength * 1e2:2.2}   pm")
    print(f"dR              = {metadata.dr} Å")  # real space pixel size
    print(f"scan       size = {d.scan_dimensions}")
    print(f"detector   size = {d.frame_dimensions}")
    print(f"scan       FOV  = {d.scan_dimensions*metadata.dr/10} nm")
    print(f"fluence         ~ {fluence} e/Å^2")
    print(f"flux            ~ {flux} e/Å^2/s")


    dssb = d
    metadata.k_max = metadata.alpha_rad * alpha_max_factor / metadata.wavelength
    s = dssb.sum_diffraction()

    if False:
        f,ax = plt.subplots(figsize=(4,4))
        imax = ax.imshow(s)
        ax.set_title('Sum after cropping for SSB')
        plt.colorbar(imax)

    slic = np.s_[:,:]
    data = dssb.slice(slic)

    # ssb_size = np.array([15,15])
    ssb_size = np.array([25,25])
    bin_factor = int(np.min(np.floor(data.frame_dimensions/ssb_size)))
    radius2 = radius/bin_factor
    meta = metadata
    verbose = True

    start = time.perf_counter()
    dc = d4.sparse_to_dense_datacube(data.indices, data.counts, data.scan_dimensions, data.frame_dimensions, data.frame_dimensions/2, data.frame_dimensions[0]/2, data.frame_dimensions[0]/2, binning=bin_factor, fftshift=False)
    print(f"Bin by {bin_factor} for ssb took {time.perf_counter() - start}s")

    rmax = dc.shape[-1] // 2
    alpha_max = rmax / radius2 * meta.alpha_rad

    r_min = meta.wavelength / (2 * alpha_max)
    r_min = [r_min, r_min]
    k_max = [alpha_max / meta.wavelength, alpha_max / meta.wavelength]
    r_min1 = np.array(r_min)
    dxy1 = np.array(meta.dr)

    M = cp.array(dc).astype(cp.float32)
    xp = cp.get_array_module(M)
    ny, nx, nky, nkx = M.shape

    Qx1d, Qy1d = get_qx_qy_1D([nx, ny], meta.dr, M.dtype, fft_shifted=False)

    start = time.perf_counter()
    G = cpfft.fft2(M, axes=(0, 1), overwrite_x=True)
    G /= cp.sqrt(np.prod(G.shape[:2]))
    print(f"FFT along scan coordinate took {time.perf_counter() - start}s")


    manual_frequencies = None  # [[20, 62, 490], [454, 12, 57]]

    Gabs = xp.log10(xp.sum(xp.abs(G), (2, 3)))
    sh = np.array(Gabs.shape)
    mask = ~np.array(np.fft.fftshift(d4.sector_mask(sh, sh // 2, 5, (0, 360))))
    mask[:,-1] = 0
    mask[:,0] = 0
    mask[:,1] = 0

    gg = Gabs.get()
    gg[~mask] = gg.mean()


    show_mask = False
    if show_mask:
        fig, ax = plt.subplots(1,2,figsize=(10,5))
        ax[0].imshow(np.fft.fftshift(mask))
        ax[0].set_title('FFT mask')
        ax[1].imshow(np.fft.fftshift(gg), cmap=plt.cm.get_cmap('inferno'))
        ax[1].set_title('Masked absolute values of G')


    Gabs = xp.sum(xp.abs(G), (2, 3))
    sh = np.array(Gabs.shape)

    n_fit=16
    meta.rotation_deg = 0.0
    best_angle = meta.rotation_deg
    aberrations = xp.zeros((12))

    gg = Gabs.get() * mask
    gg[gg==0] = gg.mean()

    inds = xp.argsort((gg).ravel()) 
    strongest_object_frequencies = np.unravel_index(inds[-1 - n_fit:-1], G.shape[:2])
    G_max = G[strongest_object_frequencies]

    r_min1 = np.array(r_min)
    dxy1 = np.array(meta.dr)

    r_min1 *= 1
    dxy1 *= 1.0
    Kx, Ky = get_qx_qy_1D([nkx, nky], r_min1, G[0, 0, 0, 0].real.dtype, fft_shifted=True)
    print('strongest object frequencies')
    print(strongest_object_frequencies[0])
    print(strongest_object_frequencies[1])
    print(Kx)
    print(Ky)
    print([nx, ny], dxy1)
    Qx1d, Qy1d = get_qx_qy_1D([nx, ny], dxy1, G[0, 0, 0, 0].real.dtype, fft_shifted=False)
    Qy_max1d = Qy1d[strongest_object_frequencies[0]]
    Qx_max1d = Qx1d[strongest_object_frequencies[1]]
    print(Qx1d.max())
    print(Qy1d.max())
    print(Qy_max1d)
    print(Qx_max1d)

    Gamma = disk_overlap_function(Qx_max1d, Qy_max1d, Kx, Ky, aberrations, best_angle, meta.alpha_rad, meta.wavelength)

    fig, ax = plt.subplots(1,3,figsize=(19,6))
    im = ax[0].imshow(np.log10(np.fft.fftshift(gg)+1), cmap= plt.cm.get_cmap('bone'))
    ax[0].set_title(f'Scan {1} fft')
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    im = ax[1].imshow(plotting.imsave(plotting.mosaic(G_max.get() * Gamma.get())), cmap= plt.cm.get_cmap('bone'))
    ax[1].set_title(f'Scan {1} double overlap')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    divider = make_axes_locatable(ax[1])

    im = ax[2].imshow(plotting.imsave(plotting.mosaic(G_max.get())), cmap= plt.cm.get_cmap('bone'))
    ax[2].set_title(f'Scan {1} double overlap')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    divider = make_axes_locatable(ax[1])

    plt.tight_layout()
    # fig.savefig(results_path /f'scan{1}_fft.png')


    aberrations[0] = -143.0
    print(f'defocus: {aberrations[0]}')

    Psi_Qp = cp.zeros((ny, nx), dtype=G.dtype)
    Psi_Qp_left_sb = cp.zeros((ny, nx), dtype=np.complex64)
    Psi_Qp_right_sb = cp.zeros((ny, nx), dtype=np.complex64)
    Psi_Rp = cp.zeros((ny, nx), dtype=G.dtype)
    Psi_Rp_left_sb = cp.zeros((ny, nx), dtype=np.complex64)
    Psi_Rp_right_sb = cp.zeros((ny, nx), dtype=np.complex64)

    start = time.perf_counter()
    eps = 1e-3
    single_sideband_reconstruction( # want this function
        G,
        Qx1d,
        Qy1d,
        Kx,
        Ky,
        aberrations,
        best_angle,
        meta.alpha_rad,
        Psi_Qp,
        Psi_Qp_left_sb,
        Psi_Qp_right_sb,
        eps,
        meta.wavelength,
    )

    Psi_Rp_left_sb = cpfft.ifft2(Psi_Qp_left_sb, norm="ortho")
    Psi_Rp_right_sb = cpfft.ifft2(Psi_Qp_right_sb, norm="ortho")
    Psi_Rp = cpfft.ifft2(Psi_Qp, norm="ortho")

    ssb_defocal = Psi_Rp.get()
    ssb_defocal_right = Psi_Rp_right_sb.get()
    ssb_defocal_left = Psi_Rp_left_sb.get()

    print(f"SSB took {time.perf_counter() - start}")

    fig, ax = plt.subplots(dpi=100) # 300
    im1 = ax.imshow(np.angle(ssb_defocal_right), cmap= plt.cm.get_cmap('bone'))
    ax.set_title(f'Scan {scan_number} SSB ptychography')
    ax.set_xticks([])
    ax.set_yticks([])
    # fig.colorbar(im1, ax=ax)
    ax.add_artist(ScaleBar(metadata.dr[0]/10,'nm'))

    plt.show()

    tifffile.imwrite(results_path /f'scan{scan_number}_ssb_ptycho_best_right.tif',np.angle(ssb_defocal_right).astype('float32'), imagej=True, resolution=(1./(metadata.dr[0]/10), 1./(metadata.dr[1]/10)), metadata={'spacing': 1 / 10, 'unit': 'nm', 'axes': 'YX'})


if __name__ == '__main__':
    main()

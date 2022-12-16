import math
import numpy as np
import cupy as cp
import cupyx
import cupyx.scipy.fft as cpfft
import cupyx.scipy.ndimage
import scipy
import scipy.fft as spfft
import scipy.ndimage
import matplotlib.pyplot as plt
from ncempy.io.dm import fileDM
import h5py
from .optics import wavelength

from numba import jit, cuda

import time


class Metadata4D:
    def __init__(self):
        self.dr = None
        self.k_max = None
        self.alpha_rad = None
        self.rotation_deg = None
        self.E_ev = None
        self.wavelength = None


    @staticmethod
    def from_dm4_file(filename):
        m = Metadata4D()

        with fileDM(filename) as f:
            m.E_ev = f.allTags['.ImageList.2.ImageTags.Microscope Info.Voltage']
            m.dr = np.array(f.scale[-2:]) * 10
        m.wavelength = wavelength(m.E_ev)
        return m


class MetadataEMPAD:
    def __init__(self):
        self.dr = None
        self.alpha_rad = None
        self.rotation_deg = None
        self.E_ev = None
        self.wavelength = None
        self.nominal_mag = None


def locate_BF_disk(pacbed, threshold=0.1, fix_4Dcam=False):
    from skimage.filters import gaussian
    pacbed_blur = (gaussian(pacbed.astype(np.float32),2) > pacbed.max() * threshold).astype(float)
    
    # the 4D camera has several half-rows of dead pixels - try to mitigate that
    if fix_4Dcam:
        dead_y = [183, 184, 185, 387, 388, 389]
        dead_xlims = [288, 576]
        intensity_guess = np.linspace(1.0, 0.0, dead_xlims[1] - dead_xlims[0])
        for dead_y_val in dead_y:
            pacbed_blur[dead_y, dead_xlims[0]:dead_xlims[1]] = intensity_guess

    radius, center_y, center_x = get_probe_size(pacbed_blur)
    print(f'radius: {radius:.2f} px, center: (x,y) = ({center_x:.2f}, {center_y:.2f})')

    fig,ax = plt.subplots()
    im = ax.imshow(pacbed)
    cb = fig.colorbar(im, ax=ax)
    ax.plot(center_x, center_y, 'g.')

    angles = np.linspace(0, 2*np.pi, 18)
    xcirc = center_x + radius * np.cos(angles)
    ycirc = center_y + radius * np.sin(angles)
    ax.plot(xcirc, ycirc, 'ro')

    plt.show()
    
    return radius, (center_y, center_x)


def get_CoM(ar):
    """
    copied from py4DSTEM

    Finds and returns the center of mass of array ar.
    """
    nx, ny = np.shape(ar)
    ry, rx = np.meshgrid(np.arange(ny), np.arange(nx))
    tot_intens = np.sum(ar)
    xCoM = np.sum(rx * ar) / tot_intens
    yCoM = np.sum(ry * ar) / tot_intens
    return xCoM, yCoM


def get_probe_size(DP, thresh_lower=0.01, thresh_upper=0.99, N=100):
    """
    copied from py4DSTEM


    Gets the center and radius of the probe in the diffraction plane.
    The algorithm is as follows:
    First, create a series of N binary masks, by thresholding the diffraction pattern
    DP with a linspace of N thresholds from thresh_lower to thresh_upper, measured
    relative to the maximum intensity in DP.
    Using the area of each binary mask, calculate the radius r of a circular probe.
    Because the central disk is typically very intense relative to the rest of the DP, r
    should change very little over a wide range of intermediate values of the threshold.
    The range in which r is trustworthy is found by taking the derivative of r(thresh)
    and finding identifying where it is small.  The radius is taken to be the mean of
    these r values. Using the threshold corresponding to this r, a mask is created and
    the CoM of the DP times this mask it taken.  This is taken to be the origin x0,y0.
    Args:
        DP (2D array): the diffraction pattern in which to find the central disk.
            A position averaged, or shift-corrected and averaged, DP works best.
        thresh_lower (float, 0 to 1): the lower limit of threshold values
        thresh_upper (float, 0 to 1): the upper limit of threshold values
        N (int): the number of thresholds / masks to use
    Returns:
        (3-tuple): A 3-tuple containing:
            * **r**: *(float)* the central disk radius, in pixels
            * **x0**: *(float)* the x position of the central disk center
            * **y0**: *(float)* the y position of the central disk center
    """
    thresh_vals = np.linspace(thresh_lower, thresh_upper, N)
    r_vals = np.zeros(N)

    # Get r for each mask
    DPmax = np.max(DP)
    for i in range(len(thresh_vals)):
        thresh = thresh_vals[i]
        mask = DP > DPmax * thresh
        r_vals[i] = np.sqrt(np.sum(mask) / np.pi)

    # Get derivative and determine trustworthy r-values
    dr_dtheta = np.gradient(r_vals)
    mask = (dr_dtheta <= 0) * (dr_dtheta >= 2 * np.median(dr_dtheta))
    r = np.mean(r_vals[mask])

    # Get origin
    thresh = np.mean(thresh_vals[mask])
    mask = DP > DPmax * thresh
    x0, y0 = get_CoM(DP * mask)

    return r, x0, y0


def shift_and_crop(arr4d, radius, center, use_gpu=True):
    frame_radius = int(math.ceil(radius)) + 1
    center_y, center_x = center
    yfrac, y_ic = math.modf(center_y)
    xfrac, x_ic = math.modf(center_x)
    y_ic, x_ic = int(y_ic), int(x_ic)
    
    margin = 1
    precrop_r = frame_radius + margin
    data_precrop = arr4d[:, :, y_ic-precrop_r:y_ic+precrop_r+1, x_ic-precrop_r:x_ic+precrop_r+1]
    
    if use_gpu:
        data_shift = cupyx.scipy.ndimage.shift(cp.array(data_precrop), (0, 0, -yfrac, -xfrac))
        data_crop = cp.ascontiguousarray(data_shift[:, :, margin:-margin, margin:-margin])
    else:
        data_shift = scipy.ndimage.shift(data_precrop, (0, 0, -yfrac, -xfrac))
        data_crop = cp.array(data_shift[:, :, margin:-margin, margin:-margin])
    return data_crop, frame_radius


def sector_mask(shape, centre, radius, angle_range=(0,360)):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """

    x, y = np.ogrid[:shape[0], :shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2 * np.pi

    # convert cartesian --> polar coordinates
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2 * np.pi)

    # circular mask
    circmask = r2 <= radius * radius

    # angular mask
    anglemask = theta <= (tmax - tmin)

    return circmask * anglemask


def load_sparse_4dcam_data(filename4d, crop_to_n=None, start_idx=(0,0)):
    t1 = time.perf_counter()

    with h5py.File(filename4d, 'r') as f0:
        frames = f0['electron_events/frames'][:]
        scan_dimensions = (f0['electron_events/scan_positions'].attrs['Ny'],
                           f0['electron_events/scan_positions'].attrs['Nx'])
        frame_dimensions = np.array((576, 576))

    t2 = time.perf_counter()
    print(f'Frames loaded: {t2-t1:.3f} s')

    b_GB = 1073741824
    b_MB = 1048576
    max_frame_size = 0
    frame_tot = 0
    tot_counts = 0
    print('collecting some statistics')
    for ev in frames:
        frame_tot += ev.nbytes
        tot_counts += ev.shape[0]
        if ev.shape[0] > max_frame_size:
            max_frame_size = ev.shape[0]
    print(f'Frames read: {frame_tot/b_GB:.3f} GB')

    frame_indices = np.arange(scan_dimensions[0] * scan_dimensions[1])
    frame_indices = np.reshape(frame_indices, scan_dimensions)

    # scan_dimensions - there is an extra row that needs to be cropped out
    frame_indices = frame_indices[:, :-1]
    scan_dimensions = (scan_dimensions[0], scan_dimensions[1]-1)
    
    if crop_to_n is not None:
        s0, s1 = start_idx
        if s0 < 0 or s1 < 0 or s0+crop_to_n >= scan_dimensions[0] or s1+crop_to_n >= scan_dimensions[1]:
            raise ValueError(f'Crop indices {start_idx}, {crop_to_n} incompatible with scan dimensions {scan_dimensions}')
        frame_indices = frame_indices[s0:s0+crop_to_n, s1:s1+crop_to_n]
        scan_dimensions = (crop_to_n, crop_to_n)
        print(f'Cropped to {scan_dimensions} frames')

    frames_crop = frames.ravel()[frame_indices.flatten()]
    max_frame_size = 0
    tot_counts_crop = 0
    tot_image = np.zeros(scan_dimensions[0] * scan_dimensions[1])
    for index, ev in enumerate(frames_crop.ravel()):
        tot_counts_crop += ev.shape[0]
        tot_image[index] = ev.shape[0]
        if ev.shape[0] > max_frame_size:
            max_frame_size = ev.shape[0]
    tot_image = tot_image.reshape((scan_dimensions[0], scan_dimensions[1]))
    print(f'Largest frame had {max_frame_size} electrons; total number of electrons: {tot_counts_crop}')

    t2 = time.perf_counter()
    print(f'Time elapsed: {t2-t1:.3f} s')
    return frames_crop, scan_dimensions, frame_dimensions, max_frame_size, tot_counts_crop, tot_image


def get_fluence(sum_electrons, scan_dimensions, dr):
    area = np.prod(scan_dimensions) * dr**2
    return sum_electrons/area


def get_flux(sum_electrons, scan_dimensions, dr, dwell_time):
    fluence = get_fluence(sum_electrons, scan_dimensions, dr)
    flux = fluence / (np.prod(scan_dimensions) * dwell_time)
    return flux


@cuda.jit
def sparse_to_pacbed_kernel(indices1d, pacbed_gpu, frame_dimensions, start, end):
    n = cuda.grid(1)
    MY, MX = frame_dimensions
    if n >= start and n < end:
        idx1d = indices1d[n]
        my = idx1d // MX
        mx = idx1d - my*MX
        cuda.atomic.add(pacbed_gpu, (my, mx), 1)


def calculate_pacbed_from_sparse(frames, frame_dimensions, tot_counts, arr_size_gpu=int(3e8)):
    # 1D indices array (don't need to know scan positions for computing PACBED)
    fr_1D = cupyx.zeros_pinned((tot_counts), dtype=np.int32)
    idx = 0
    for ev in frames.ravel():
        idx_next = idx + ev.shape[0]
        fr_1D[idx:idx_next] = ev
        idx = idx_next

    pacbed_gpu = cp.zeros(frame_dimensions, dtype=np.int32)
        
    if arr_size_gpu <= tot_counts:
        fr_1D_gpu = cp.empty(arr_size_gpu, dtype=cp.int32)

        nbatches = tot_counts // arr_size_gpu
        nleft = tot_counts - nbatches * arr_size_gpu
        threadsperblock = 256
        blockspergrid = math.ceil(arr_size_gpu / threadsperblock)

        for b_idx in range(nbatches):
            offset = b_idx * arr_size_gpu
            fr_1D_gpu.set(fr_1D[offset:offset+arr_size_gpu])
            sparse_to_pacbed_kernel[blockspergrid, threadsperblock](fr_1D_gpu, pacbed_gpu, tuple(frame_dimensions), 0, arr_size_gpu)
            cuda.synchronize()

        if nleft > 0:
            fr_1D_gpu.set(fr_1D[-arr_size_gpu:])
            sparse_to_pacbed_kernel[blockspergrid, threadsperblock](fr_1D_gpu, pacbed_gpu, tuple(frame_dimensions), arr_size_gpu-nleft, arr_size_gpu)
            cuda.synchronize()
    else:
        fr_1D_gpu = cp.array(fr_1D)
        threadsperblock = 256
        blockspergrid = math.ceil(tot_counts / threadsperblock)
        sparse_to_pacbed_kernel[blockspergrid, threadsperblock](fr_1D_gpu, pacbed_gpu, tuple(frame_dimensions), 0, tot_counts)

    return pacbed_gpu.get()


# typically use this for 4D Camera data
def crop_bin_single_CBED(frame_size, bin_factor, center, frame_dimensions, pacbed):
    radius_max_bin_int = frame_size // 2
    center_frame = frame_size // 2
    max_dist_center = (radius_max_bin_int + 0.5 - 1e-3) * bin_factor
    # print(radius_data_int, radius_max_int, radius_max, frame_size, max_dist_center)

    pacbed_binned = np.zeros((frame_size, frame_size), dtype=np.int32)
    pixels_in_bin = np.zeros((frame_size, frame_size), dtype=int)
    center_y, center_x = center
    for iy in range(frame_dimensions[0]):
        for ix in range(frame_dimensions[1]):
            my_center = iy - center_y
            mx_center = ix - center_x
            dist_center = math.sqrt(my_center**2 + mx_center**2)
            if dist_center < max_dist_center:
                mybin = center_frame + int(np.round(my_center/bin_factor))
                mxbin = center_frame + int(np.round(mx_center/bin_factor))
                pacbed_binned[mybin, mxbin] += pacbed[iy, ix]
                pixels_in_bin[mybin, mxbin] += 1
    return pacbed_binned, pixels_in_bin


@cuda.jit
def sparse_to_dense_crop_bin_kernel(indices3d, max_frame_size, dc, 
                                    idx_to_my, idx_to_mx, no_count_indicator, ny_offset, start, end):
    iy, nx = cuda.grid(2)
    NY, NX, _, _ = dc.shape
    if iy >= start and iy < end and nx < NX:
        ny = iy + ny_offset
        if ny < NY:
            i = 0
            done = False
            while i < max_frame_size and not done:
                idx1d = indices3d[iy, nx, i]
                if idx1d == no_count_indicator:
                    done = True
                elif idx_to_my[idx1d] >= 0:
                    cuda.atomic.add(dc, (ny, nx, idx_to_my[idx1d], idx_to_mx[idx1d]), 1)
                i += 1


def crop_bin_sparse_to_dense(frames, frame_size, bin_factor, center, radius, frame_dimensions, scan_dimensions, max_frame_size, arr_size_gpu=int(3e8)):
    t1 = time.perf_counter()
    # figure out where every pixel of the detector should go, ahead of time
    idx_range_1D = np.arange(frame_dimensions[0] * frame_dimensions[1])
    my_range = idx_range_1D // frame_dimensions[1]
    mx_range = idx_range_1D - my_range * frame_dimensions[1]
    center_frame = frame_size // 2
    radius_max_bin_int = frame_size // 2
    max_dist_center = (radius_max_bin_int + 0.5 - 1e-3) * bin_factor
    center_y, center_x = center
    my_center_range = my_range - center_y
    mx_center_range = mx_range - center_x
    dist_center_range = np.sqrt(my_center_range**2 + mx_center_range**2)
    mybin_range = (my_center_range/bin_factor).round().astype(int) + center_frame
    mxbin_range = (mx_center_range/bin_factor).round().astype(int) + center_frame
    replace_mask = dist_center_range >= max_dist_center
    mybin_range[replace_mask] = -1
    mxbin_range[replace_mask] = -1
    mybin_range_gpu = cp.asarray(mybin_range)
    mxbin_range_gpu = cp.asarray(mxbin_range)
    
    print('Converting frames to 3D array')
    idx_t = np.int32
    no_count_indicator = np.iinfo(idx_t).max
    fr_full = cupyx.empty_pinned((frames.ravel().shape[0], max_frame_size), dtype=idx_t)
    fr_full.fill(no_count_indicator)
    for ii, ev in enumerate(frames.ravel()):
        fr_full[ii, :ev.shape[0]] = ev
    fr_full = fr_full.reshape((scan_dimensions[0], scan_dimensions[1], max_frame_size))
    
    print('Converting frames to dense 4D array')
    dc_gpu = cp.zeros((scan_dimensions[0], scan_dimensions[1], frame_size, frame_size), dtype=np.int32)
    if scan_dimensions[0] * scan_dimensions[1] * max_frame_size >= arr_size_gpu:
        assert scan_dimensions[1] * max_frame_size <= arr_size_gpu
        
        # calculate batch sizes
        batch_size_dim0 = arr_size_gpu // (scan_dimensions[1] * max_frame_size)
        # batch_size_dim0 = batch_size_dim0 // 16 * 16
        batch_size_dim0 = batch_size_dim0 // 32 * 32

        nbatches = scan_dimensions[0] // batch_size_dim0
        nleft = scan_dimensions[0] - nbatches * batch_size_dim0

        fr_full_gpu = cp.empty((batch_size_dim0, scan_dimensions[1], max_frame_size), dtype=idx_t)

        threadsperblock = (32, 32)
        blockspergrid = tuple(np.ceil(np.array(fr_full_gpu.shape[:2]) / threadsperblock).astype(int))

        # binning in batches using Numba
        for b_idx in range(nbatches):
            offset = b_idx * batch_size_dim0
            fr_full_gpu.set(fr_full[offset:offset+batch_size_dim0])
            sparse_to_dense_crop_bin_kernel[blockspergrid, threadsperblock](fr_full_gpu, max_frame_size, dc_gpu, 
                                        mybin_range_gpu, mxbin_range_gpu, no_count_indicator, offset, 0, batch_size_dim0)
            cuda.synchronize()
            print('batch at offset:', offset)

        if nleft > 0:
            fr_full_gpu.set(fr_full[-batch_size_dim0:])
            sparse_to_dense_crop_bin_kernel[blockspergrid, threadsperblock](fr_full_gpu, max_frame_size, dc_gpu, 
                                        mybin_range_gpu, mxbin_range_gpu, no_count_indicator, scan_dimensions[0]-batch_size_dim0, batch_size_dim0-nleft, batch_size_dim0)
            cuda.synchronize()
            print('partial batch of', nleft)
    else:
        fr_full_gpu = cp.array(fr_full)
        threadsperblock = (32, 32)
        blockspergrid = tuple(np.ceil(np.array(fr_full_gpu.shape[:2]) / threadsperblock).astype(int))
        sparse_to_dense_crop_bin_kernel[blockspergrid, threadsperblock](fr_full_gpu, max_frame_size, dc_gpu, 
                                        mybin_range_gpu, mxbin_range_gpu, no_count_indicator, 0, 0, scan_dimensions[0])
        
            
    t2 = time.perf_counter()
    print(f'cropping, binning took {t2-t1:.3f} sec')
    return dc_gpu
import math
import numpy as np
import cupy as cp
import torch as th
import sys

from ncempy.io.dm import fileDM
from .optics import wavelength

from skimage.filters import gaussian
from numba import jit, cuda
import h5py

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


class Sparse4DData:

    def __init__(self):
        self.indices = None
        self.counts = None
        self.scan_dimensions = None
        self.frame_dimensions = None

    @staticmethod
    def from_4Dcamera_file(filename, start_idx=None, n_pos=None):
        with h5py.File(filename, 'r') as f0:
            frames = f0['electron_events/frames'][:]
            scan_dimensions = (f0['electron_events/scan_positions'].attrs['Ny'],
                               f0['electron_events/scan_positions'].attrs['Nx'])
            frame_dimensions = np.array((576, 576))

        def unragged_frames_size(frames):
            mm = 0
            for ev in frames:
                if ev.shape[0] > mm:
                    mm = ev.shape[0]
            return mm

        def make_unragged_frames(frames, scan_dimensions):
            unragged_frame_size = unragged_frames_size(frames.ravel())
            print('scan dimensions:', scan_dimensions)
            print('frames size: ', len(frames.ravel()), frames[0].shape)
            fr_full = np.zeros((frames.ravel().shape[0], unragged_frame_size), dtype=np.int32)
            fr_full[:] = np.iinfo(fr_full.dtype).max
            print('filling fr_full array')
            for ii, ev in enumerate(frames.ravel()):
                fr_full[ii, :ev.shape[0]] = ev
            fr_full_4d = cp.asarray(fr_full.reshape((*scan_dimensions, fr_full.shape[1])))
            print('fr_full: ', fr_full.shape, 'fr_full_4d: ', fr_full_4d.shape)

            fr_full_4d = fr_full_4d[:, :-1, :]
            print('fr_full_4d: ', fr_full_4d.shape)
            return fr_full_4d

        def make_unragged_frames_cropped(frames, scan_dimensions, start_idx, n_pos):
            frame_indices = np.arange(scan_dimensions[0] * scan_dimensions[1])
            frame_indices = np.reshape(frame_indices, scan_dimensions)
            s1, s2 = start_idx
            if s1 < 0 or s2 < 0 or s1+n_pos >= scan_dimensions[0] or s2+n_pos >= scan_dimensions[1]:
                sys.exit(f'Crop indices {start_idx}, {n_pos} incompatible with scan dimensions {scan_dimensions}')
            frame_indices = frame_indices[s1:s1+n_pos, s2:s2+n_pos].flatten()
            print(np.reshape(frame_indices, (n_pos, n_pos)))
            frames_crop = frames.ravel()[frame_indices]
            unragged_frame_size = unragged_frames_size(frames_crop)
            fr_full = np.zeros((frames_crop.ravel().shape[0], unragged_frame_size), dtype=np.int32)
            fr_full[:] = np.iinfo(fr_full.dtype).max
            print('filling fr_full array')
            for ii, ev in enumerate(frames_crop.ravel()):
                fr_full[ii, :ev.shape[0]] = ev
            fr_full_4d = cp.asarray(fr_full.reshape((n_pos, n_pos, fr_full.shape[1])))
            return fr_full_4d 

        d = Sparse4DData()
        if start_idx is not None and n_pos is not None:
            d.indices = cp.ascontiguousarray(make_unragged_frames_cropped(frames.ravel(), scan_dimensions, start_idx, n_pos))
        else:
            d.indices = cp.ascontiguousarray(make_unragged_frames(frames.ravel(), scan_dimensions))
        d.scan_dimensions = np.array(d.indices.shape[:2])
        d.frame_dimensions = frame_dimensions
        d.counts = cp.zeros(d.indices.shape, dtype=cp.bool)
        d.counts[d.indices != cp.iinfo(d.indices.dtype).max] = 1

        return d

    def crop_symmetric_center_(self, center, max_radius = None):
        if max_radius is None:
            y_min_radius = np.min([center[0], self.frame_dimensions[0] - center[0]])
            x_min_radius = np.min([center[1], self.frame_dimensions[1] - center[1]])
            max_radius = np.min([y_min_radius, x_min_radius])
        new_frames, new_frame_dimensions = crop_symmetric_around_center(self.indices,
                                                                        self.frame_dimensions,
                                                                        center, max_radius)
        print(f'old frames frame_dimensions: {self.frame_dimensions}')
        print(f'new frames frame_dimensions: {new_frame_dimensions}')
        self.indices = new_frames
        self.counts = cp.zeros(self.indices.shape, dtype=cp.bool)
        self.counts[self.indices != cp.iinfo(self.indices.dtype).max] = 1
        self.frame_dimensions = new_frame_dimensions

    def crop_symmetric_center(self, center, max_radius = None):
        if max_radius is None:
            y_min_radius = np.min([center[0], self.frame_dimensions[0] - center[0]])
            x_min_radius = np.min([center[1], self.frame_dimensions[1] - center[1]])
            max_radius = np.min([y_min_radius, x_min_radius])
        new_frames, new_frame_dimensions = crop_symmetric_around_center(cp.array(self.indices),
                                                                        cp.array(self.frame_dimensions),
                                                                        center, max_radius)
        print(f'old frames frame_dimensions: {self.frame_dimensions}')
        print(f'new frames frame_dimensions: {new_frame_dimensions}')
        res = Sparse4DData()
        res.indices = new_frames
        res.counts = cp.zeros(self.indices.shape, dtype=cp.bool)
        res.counts[self.indices != cp.iinfo(self.indices.dtype).max] = 1
        res.frame_dimensions = new_frame_dimensions
        res.scan_dimensions = self.scan_dimensions.copy()
        return res

    def rotate_(self, angle_rad, center=None):
        if center is None:
            center = self.frame_dimensions // 2
        new_indices = rotate(self.indices, self.frame_dimensions, center, angle_rad)
        self.indices = new_indices

    def rotate(self, angle_rad, center=None):
        if center is None:
            center = self.frame_dimensions // 2
        new_indices = rotate(self.indices, self.frame_dimensions, center, angle_rad)
        res = Sparse4DData()
        res.indices = new_indices
        res.counts = self.counts.copy()
        res.frame_dimensions = self.frame_dimensions
        res.scan_dimensions = self.scan_dimensions.copy()
        return res

    def sum_diffraction(self):
        res = sum_frames(self.indices, self.counts, self.frame_dimensions)
        return res

    @staticmethod
    def _determine_center_and_radius(data , manual=False, size=25):
        sh = np.concatenate([data.scan_dimensions,data.frame_dimensions])
        c = np.zeros((2,))
        c[:] = (sh[-1] // 2, sh[-2] // 2)
        c = cp.array(c)
        radius = cp.ones((1,)) * sh[-1] // 2
        inds = cp.array(data.indices[:size, :size].astype(cp.uint32))
        cts = cp.array(data.counts[:size, :size].astype(cp.uint32))
        dc_subset = sparse_to_dense_datacube_crop(inds,cts, (size,size), data.frame_dimensions, c, radius, bin=2)
        dcs = cp.sum(dc_subset, (0, 1))
        m1 = dcs.get()
        m = (gaussian(m1.astype(cp.float32),2) > m1.max() * 3e-1).astype(cp.float)
        r, y0, x0 = get_probe_size(m)
        return 2 * np.array([y0,x0]), r*2

    def determine_center_and_radius(self, manual=False, size=25):
        return Sparse4DData._determine_center_and_radius(self, manual, size=size)

    def to_dense(self, bin_factor):
        dense = sparse_to_dense_datacube_crop_gain_mask(self.indices, self.counts.astype(cp.int16), self.scan_dimensions,
                                                self.frame_dimensions, self.frame_dimensions/2, self.frame_dimensions[0]/2,
                                                self.frame_dimensions[0]/2, binning=bin_factor, fftshift=False)
        return dense

    @staticmethod
    def from_dense(dense, make_float = False):
        res = Sparse4DData()
        res.frame_dimensions = np.array(dense.shape[-2:])
        res.scan_dimensions = np.array(dense.shape[:2])

        inds = np.prod(res.frame_dimensions)
        if inds > 2**31:
            dtype = cp.int64
        elif inds > 2**15:
            dtype = cp.int32
        elif inds > 2**8:
            dtype = cp.int16
        else:
            dtype = cp.uint8

        nonzeros = cp.sum((dense > 0),(2,3))
        nonzeros = cp.max(nonzeros)

        bits_counts = np.log2(dense.max())
        if make_float:
            dtype_counts = cp.float32
        else:
            if bits_counts > np.log2(2**31-1):
                dtype_counts = cp.int64
            elif bits_counts > np.log2(2**15-1):
                dtype_counts = cp.int32
            elif bits_counts > 8:
                dtype_counts = cp.int16
            else:
                dtype_counts = cp.uint8

        threadsperblock = (16, 16)
        blockspergrid = tuple(np.ceil(res.scan_dimensions / threadsperblock).astype(np.int))
        dense = cp.array(dense)
        indices = cp.zeros((*dense.shape[:2], nonzeros), dtype=dtype)
        indices[:] = cp.iinfo(dtype).max
        counts = cp.zeros((*dense.shape[:2], nonzeros), dtype=dtype_counts)
        dense_to_sparse_kernel[blockspergrid, threadsperblock](dense, indices, counts, cp.array(res.frame_dimensions))

        res.indices = indices.get()
        res.counts = counts.get()

        print(f'frame_dimensions: {res.frame_dimensions}')
        print(f'scan_dimensions : {res.scan_dimensions}')
        print(f'Using dtype: {dtype} for indices')
        print(f'Using dtype: {dtype_counts} for counts')
        return res

    @staticmethod
    def rebin(sparse_data, bin_factor : int):
        dense = sparse_to_dense_datacube_crop_gain_mask(sparse_data.indices, sparse_data.counts.astype(cp.int16), sparse_data.scan_dimensions,
                                                sparse_data.frame_dimensions, sparse_data.frame_dimensions/2, sparse_data.frame_dimensions[0]/2,
                                                sparse_data.frame_dimensions[0]/2, binning=bin_factor, fftshift=False)
        sparse = Sparse4DData.from_dense(dense)
        return sparse

    @staticmethod
    def fftshift(sparse_data):
        indices = sparse_data.indices
        scan_dimensions = sparse_data.scan_dimensions
        frame_dimensions = sparse_data.frame_dimensions
        center_frame = frame_dimensions / 2
        radius_data = frame_dimensions[0] / 2

        threadsperblock = (16, 16)
        blockspergrid = tuple(np.ceil(np.array(indices.shape[:2]) / threadsperblock).astype(np.int))

        no_count_indicator = np.iinfo(indices.dtype).max
        inds = cp.array(indices)
        fftshift_kernel[blockspergrid, threadsperblock](inds, center_frame, scan_dimensions, no_count_indicator)
        sparse_data.indices = inds.get()
        return sparse_data

    @staticmethod
    def fftshift_and_pad_to(sparse_data, pad_to_frame_dimensions):
        indices = sparse_data.indices
        scan_dimensions = sparse_data.scan_dimensions
        frame_dimensions = sparse_data.frame_dimensions
        center_frame = frame_dimensions / 2

        threadsperblock = (16, 16)
        blockspergrid = tuple(np.ceil(np.array(indices.shape[:2]) / threadsperblock).astype(np.int))

        no_count_indicator_old = np.iinfo(indices.dtype).max

        inds = np.prod(pad_to_frame_dimensions)
        if inds > 2**15:
            dtype = cp.int64
        elif inds > 2**15:
            dtype = cp.int32
        elif inds > 2**8:
            dtype = cp.int16
        else:
            dtype = cp.uint8

        no_count_indicator_new = cp.iinfo(dtype).max

        inds = cp.array(indices, dtype=dtype)
        fftshift_pad_kernel[blockspergrid, threadsperblock](inds, center_frame, scan_dimensions,
                                                            cp.array(pad_to_frame_dimensions), no_count_indicator_old,
                                                            no_count_indicator_new)
        sparse_data.indices = inds.get()
        sparse_data.frame_dimensions = np.array(pad_to_frame_dimensions)
        return sparse_data

    def fftshift_(self):
        return Sparse4DData.fftshift(self)

    def fftshift_and_pad_to_(self, pad_to_frame_dimensions):
        return Sparse4DData.fftshift_and_pad_to(self, pad_to_frame_dimensions)

    def bin(self, binning_factor):
        res = Sparse4DData.rebin(self, binning_factor)
        return res

    def virtual_annular_image(self, inner_radius, outer_radius, center):
        img = cp.zeros(tuple(self.scan_dimensions), dtype=np.uint32)
        no_count_indicator = np.iinfo(self.indices.dtype).max
        threadsperblock = (16, 16)
        blockspergrid = tuple(np.ceil(np.array(self.indices.shape[:2]) / threadsperblock).astype(np.int))
        virtual_annular_image_kernel[blockspergrid, threadsperblock](img, cp.array(self.indices), cp.array(self.counts.astype(np.uint32)),
                                                                     inner_radius, outer_radius, cp.array(center),
                                                                     cp.array(self.frame_dimensions), no_count_indicator)
        return img.get()

    def fluence(self, dr):
        sum_electrons = self.counts.sum()
        area = np.prod(self.scan_dimensions) * dr**2
        return sum_electrons/area

    def flux(self, dr, dwell_time):
        fluence = self.fluence(dr)
        flux = fluence / (np.prod(self.scan_dimensions) * dwell_time)
        return flux

    def slice(self, slice):
        res = Sparse4DData()
        res.indices = cp.ascontiguousarray(self.indices[slice])
        res.counts = cp.ascontiguousarray(self.counts[slice])
        res.scan_dimensions = np.array(res.counts.shape[:2])
        res.frame_dimensions = self.frame_dimensions.copy()
        return res

    def center_of_mass(self):
        qx, qy = np.meshgrid(np.arange(self.scan_dimensions[0]),np.arange(self.scan_dimensions[1]))
        comx = cp.zeros(self.scan_dimensions, dtype=cp.float32)
        comy = cp.zeros(self.scan_dimensions, dtype=cp.float32)

        no_count_indicator = np.iinfo(self.indices.dtype).max

        mass = cp.sum(self.counts,2)

        threadsperblock = (16, 16)
        blockspergrid = tuple(np.ceil(np.array(self.indices.shape[:2]) / threadsperblock).astype(np.int))

        qx = cp.array(qx).astype(cp.float32)
        qy = cp.array(qy).astype(cp.float32)
        center_of_mass_kernel[blockspergrid, threadsperblock](comx, comy, self.indices, self.counts.astype(cp.uint32),
                                                              cp.array(self.frame_dimensions), no_count_indicator, qx, qy)
        comy = comy
        comx = comx
        comx /= mass + 1e-6
        comy /= mass + 1e-6
        comy[comy==0] = cp.mean(comy[comy!=0])
        comx[comx==0] = cp.mean(comx[comx!=0])
        comx -= cp.mean(comx)
        comy -= cp.mean(comy)
        return comy, comx


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


@cuda.jit
def center_of_mass_kernel(comx, comy, indices, counts, frame_dimensions, no_count_indicator, qx, qy):
    ny, nx = cuda.grid(2)
    NY, NX, _ = indices.shape
    MY, MX = frame_dimensions
    if ny < NY and nx < NX:
        for i in range(indices[ny, nx].shape[0]):
            idx1d = indices[ny, nx, i]
            my = idx1d // MX
            mx = idx1d - my * MX
            if idx1d != no_count_indicator:
                cuda.atomic.add(comy, (ny, nx), counts[ny, nx, i] * qy[my, mx])
                cuda.atomic.add(comx, (ny, nx), counts[ny, nx, i] * qx[my, mx])


@cuda.jit
def sparse_to_dense_datacube_kernel_crop(dc, indices, counts, frame_dimensions, bin, start, end, no_count_indicator):
    ny, nx = cuda.grid(2)
    NY, NX, MYBIN, MXBIN = dc.shape
    MY, MX = frame_dimensions
    if ny < NY and nx < NX:
        for i in range(indices[ny, nx].shape[0]):
            idx1d = indices[ny, nx, i]
            my = idx1d // MX
            mx = idx1d - my * MX
            if my >= start[0] and mx >= start[1] and my < end[0] and mx < end[1]:
                mybin = (my - start[0]) // bin
                mxbin = (mx - start[1]) // bin
                if idx1d != no_count_indicator:
                    cuda.atomic.add(dc, (ny, nx, mybin, mxbin), counts[ny, nx, i])


def sparse_to_dense_datacube_crop(indices, counts, scan_dimensions, frame_dimensions, center, radius, bin=1):
    radius = int(np.ceil(radius / bin) * bin)
    start = center - radius
    end = center + radius
    frame_size = 2 * radius // bin

    xp = cp.get_array_module(indices)
    print('attempting to allocate array: ', (*scan_dimensions, frame_size, frame_size), indices.dtype)
    dc = cp.zeros((*scan_dimensions, frame_size, frame_size), dtype=indices.dtype)

    threadsperblock = (16, 16)
    blockspergrid = tuple(np.ceil(np.array(indices.shape[:2]) / threadsperblock).astype(np.int))

    no_count_indicator = np.iinfo(indices.dtype).max

    sparse_to_dense_datacube_kernel_crop[blockspergrid, threadsperblock](dc, indices, counts, cp.array(frame_dimensions), bin,
                                                                         start, end, no_count_indicator)
    return dc


@cuda.jit
def sparse_to_dense_datacube_crop_gain_mask_kernel(dc, frames, counts,
                                                   frame_dimensions,
                                                   center_frame, center_data,
                                                   radius_data_int, binning,
                                                   fftshift):
    ny, nx = cuda.grid(2)
    NY, NX, MYBIN, MXBIN = dc.shape
    MY, MX = frame_dimensions
    if ny < NY and nx < NX:
        for i in range(frames[ny, nx].shape[0]):
            idx1d = frames[ny, nx, i]
            my = idx1d // MX
            mx = idx1d - my * MX
            my_center = my - center_data[0]
            mx_center = mx - center_data[1]
            dist_center = math.sqrt(my_center ** 2 + mx_center ** 2)
            if dist_center < radius_data_int:
                mybin = int(center_frame[0] + my_center // binning)
                mxbin = int(center_frame[1] + mx_center // binning)
                if fftshift:
                    mybin = (mybin - center_frame[0]) % (center_frame[0] * 2)
                    mxbin = (mxbin - center_frame[1]) % (center_frame[1] * 2)
                if (mxbin >= 0 and mybin >= 0):
                    cuda.atomic.add(dc, (ny, nx, mybin, mxbin), counts[ny, nx, i])


def sparse_to_dense_datacube_crop_gain_mask(indices, counts, scan_dimensions, frame_dimensions, center_data, radius_data,
                                            radius_max, binning=1, fftshift=False):
    radius_data_int = int(np.ceil(radius_data / binning) * binning)
    radius_max_int = int(np.ceil(radius_max / binning) * binning)
    frame_size = 2 * radius_max_int // binning

    print(f'radius_data_int : {radius_data_int} ')
    print(f'radius_max_int  : {radius_max_int} ')
    print(f'Dense frame size: {frame_size}x {frame_size}')

    stream = th.cuda.current_stream().cuda_stream

    dc0 = np.zeros((scan_dimensions[0],scan_dimensions[1], frame_size, frame_size), dtype=np.uint8)
    dc = th.zeros((scan_dimensions[0]//2,scan_dimensions[1], frame_size, frame_size), dtype=th.float32)

    center_frame = th.tensor([frame_size // 2, frame_size // 2])
    fd = th.as_tensor(frame_dimensions)
    center = th.as_tensor(center_data)
    inds = th.as_tensor(indices[:scan_dimensions[0]//2,...])
    cts = th.as_tensor(counts[:scan_dimensions[0]//2,...].astype(np.float32), dtype=th.float32)

    threadsperblock = (16, 16)
    blockspergrid = tuple(np.ceil(np.array(indices.shape[:2]) / threadsperblock).astype(np.int))
    print('sparse_to_dense_datacube_crop_gain_mask dtypes:',dc.dtype, inds.dtype, cts.dtype, frame_dimensions.dtype)

    sparse_to_dense_datacube_crop_gain_mask_kernel[blockspergrid, threadsperblock, stream](dc, inds, cts, fd,
                                                                                   center_frame, center,
                                                                                   radius_data_int, binning,
                                                                                   fftshift)

    dc0[:scan_dimensions[0]//2,...] = dc.cpu().type(th.uint8).numpy()

    dc[:] = 0
    inds = th.as_tensor(indices[scan_dimensions[0]//2:,...])
    cts = th.as_tensor(counts[scan_dimensions[0]//2:,...].astype(np.float32), dtype=th.float32)

    sparse_to_dense_datacube_crop_gain_mask_kernel[blockspergrid, threadsperblock, stream](dc, inds, cts, fd,
                                                                                   center_frame, center,
                                                                                   radius_data_int, binning,
                                                                                   fftshift)
    dc0[scan_dimensions[0]//2:,...] = dc.cpu().type(th.uint8).numpy()
    cuda.select_device(0)
    return dc0


def sparse_to_dense_datacube(indices, counts, scan_dimensions, frame_dimensions, center_data, radius_data,
                                            radius_max, binning=1, fftshift=False):
    radius_data_int = int(np.ceil(radius_data / binning) * binning)
    radius_max_int = int(np.ceil(radius_max / binning) * binning)
    frame_size = 2 * radius_max_int // binning

    print(f'radius_data_int : {radius_data_int} ')
    print(f'radius_max_int  : {radius_max_int} ')
    print(f'dense frame size: {frame_size}x {frame_size}')

    dc = cp.zeros((scan_dimensions[0],scan_dimensions[1], frame_size, frame_size), dtype=cp.float32)

    center_frame = cp.array(([frame_size // 2, frame_size // 2]))
    fd = cp.array(frame_dimensions)
    center = cp.array(center_data)

    threadsperblock = (16, 16)
    blockspergrid = tuple(np.ceil(np.array(indices.shape[:2]) / threadsperblock).astype(np.int))

    sparse_to_dense_datacube_crop_gain_mask_kernel[blockspergrid, threadsperblock](dc, indices, counts, fd,
        center_frame, center, radius_data_int, binning, fftshift)
    return dc


@cuda.jit
def fftshift_kernel(indices, center_frame, scan_dimensions, no_count_indicator):
    ny, nx = cuda.grid(2)
    NY, NX = scan_dimensions
    MY = center_frame[0] * 2
    MX = center_frame[1] * 2
    if ny < NY and nx < NX:
        for i in range(indices[ny, nx].shape[0]):
            idx1d = indices[ny, nx, i]
            my = idx1d // MX
            mx = idx1d - my * MX
            mysh = (my - center_frame[0]) % (center_frame[0] * 2)
            mxsh = (mx - center_frame[1]) % (center_frame[1] * 2)
            if idx1d != no_count_indicator:
                indices[ny, nx, i] = mysh * MX + mxsh


@cuda.jit
def fftshift_pad_kernel(indices, center_frame, scan_dimensions, new_frame_dimensions, no_count_indicator_old, no_count_indicator_new):
    ny, nx = cuda.grid(2)
    NY, NX = scan_dimensions
    MX = center_frame[1] * 2
    if ny < NY and nx < NX:
        for i in range(indices[ny, nx].shape[0]):
            idx1d = indices[ny, nx, i]
            my = idx1d // MX
            mx = idx1d - my * MX
            mysh = (my - center_frame[0]) % (new_frame_dimensions[0])
            mxsh = (mx - center_frame[1]) % (new_frame_dimensions[1])
            if idx1d != no_count_indicator_old:
                indices[ny, nx, i] = mysh * new_frame_dimensions[1] + mxsh
            else:
                indices[ny, nx, i] = no_count_indicator_new


@cuda.jit
def virtual_annular_image_kernel(img, indices, counts, radius_inner, radius_outer, center_frame, frame_dimensions, no_count_indicator):
    ny, nx = cuda.grid(2)
    NY, NX, _ = indices.shape
    MY, MX = frame_dimensions
    if ny < NY and nx < NX:
        for i in range(indices[ny, nx].shape[0]):
            idx1d = indices[ny, nx, i]
            my = idx1d // MX
            mx = idx1d - my * MX
            my_center = my - center_frame[0]
            mx_center = mx - center_frame[1]
            radius = math.sqrt(my_center ** 2 + mx_center ** 2)
            if radius < radius_outer and radius >= radius_inner and idx1d != no_count_indicator:
                cuda.atomic.add(img, (ny,nx), counts[ny, nx, i])


@cuda.jit
def crop_symmetric_around_center_kernel(new_frames, old_frames, center_frame, old_frame_dimensions, center_data, radius_data_int):
    ny, nx = cuda.grid(2)
    NY, NX, _ = old_frames.shape
    MY, MX = old_frame_dimensions
    MXnew = center_frame[1] * 2
    if ny < NY and nx < NX:
        k = 0
        for i in range(old_frames[ny, nx].shape[0]):
            idx1d = old_frames[ny, nx, i]
            my = idx1d // MX
            mx = idx1d - my * MX
            my_center = my - center_data[0]
            mx_center = mx - center_data[1]
            dist_center = math.sqrt(my_center ** 2 + mx_center ** 2)
            if dist_center < radius_data_int:
                mybin = int(center_frame[0] + my_center)
                mxbin = int(center_frame[1] + mx_center)
                new_frames[ny, nx, k] = mybin * MXnew + mxbin
                k += 1


def crop_symmetric_around_center(old_frames, old_frame_dimensions, center_data, max_radius):
    max_radius_int = int(max_radius)
    frame_size = 2 * max_radius_int
    center_frame = cp.array([frame_size // 2, frame_size // 2])
    new_frame_dimensions = np.array([frame_size,frame_size])

    threadsperblock = (16, 16)
    blockspergrid = tuple(np.ceil(np.array(old_frames.shape[:2]) / threadsperblock).astype(np.int))

    new_frames = cp.zeros_like(old_frames)
    new_frames[:] = cp.iinfo(new_frames.dtype).max

    crop_symmetric_around_center_kernel[blockspergrid, threadsperblock](new_frames, old_frames, center_frame,
                                                                        cp.array(old_frame_dimensions),
                                                                        cp.array(center_data), max_radius_int)

    max_counts = cp.max(cp.sum(new_frames > 0, 2).ravel())
    res = cp.ascontiguousarray(new_frames[:,:,:max_counts])
    return res, new_frame_dimensions


@cuda.jit
def rotate_kernel(frames, center_frame, old_frame_dimensions, center_data, no_count_indicator, angle_rad):
    ny, nx = cuda.grid(2)
    NY, NX, _ = frames.shape
    MY, MX = old_frame_dimensions
    MXnew = center_frame[1] * 2
    if ny < NY and nx < NX:
        for i in range(frames[ny, nx].shape[0]):
            idx1d = frames[ny, nx, i]
            if idx1d != no_count_indicator:
                my = idx1d // MX
                mx = idx1d - my * MX
                my_center = my - center_data[0]
                mx_center = mx - center_data[1]
                #rotate
                mx_center_rot = round(mx_center * math.cos(angle_rad) - my_center * math.sin(angle_rad))
                my_center_rot = round(mx_center * math.sin(angle_rad) + my_center * math.cos(angle_rad))
                mybin = int(center_frame[0] + my_center_rot)
                mxbin = int(center_frame[1] + mx_center_rot)
                frames[ny, nx, i] = mybin * MXnew + mxbin


def rotate(frames, old_frame_dimensions, center, angle_rad):
    threadsperblock = (16, 16)
    blockspergrid = tuple(np.ceil(np.array(frames.shape[:2]) / threadsperblock).astype(np.int))
    no_count_indicator = np.iinfo(frames.dtype).max
    new_frames= cp.array(frames)
    rotate_kernel[blockspergrid, threadsperblock](new_frames, center, cp.array(old_frame_dimensions), cp.array(center),
                                                  no_count_indicator, angle_rad)

    return new_frames.get()


@cuda.jit
def sum_kernel(indices, counts, frame_dimensions, sum, no_count_indicator):
    ny, nx = cuda.grid(2)
    NY, NX, _ = indices.shape
    MY, MX = frame_dimensions
    if ny < NY and nx < NX:
        for i in range(indices[ny, nx].shape[0]):
            idx1d = indices[ny, nx, i]
            my = idx1d // MX
            mx = idx1d - my * MX
            if idx1d != no_count_indicator:
                cuda.atomic.add(sum, (my, mx), counts[ny, nx, i])


def sum_frames(frames, counts, frame_dimensions):
    threadsperblock = (16, 16)
    blockspergrid = tuple(np.ceil(np.array(frames.shape[:2]) / threadsperblock).astype(np.int))

    sum = cp.zeros(frame_dimensions)
    no_count_indicator = np.iinfo(frames.dtype).max
    frames1 = cp.array(frames)
    counts1 = cp.array(counts)
    sum_kernel[blockspergrid, threadsperblock](frames1, counts1, cp.array(frame_dimensions), sum, no_count_indicator)
    return sum.get()


@cuda.jit
def rebin_kernel(indices, counts, new_frame_center, old_indices, old_counts, old_frame_center, no_count_indicator,
                 bin_factor):
    ny, nx = cuda.grid(2)
    NY, NX, _ = indices.shape
    MY = old_frame_center[0] * 2
    MX = old_frame_center[1] * 2
    MXnew = new_frame_center[1] * 2
    if ny < NY and nx < NX:
        k = 0
        for i in range(old_indices[ny, nx].shape[0]):
            idx1d = old_indices[ny, nx, i]
            my = idx1d // MX
            mx = idx1d - my * MX
            my_center = my - old_frame_center[0]
            mx_center = mx - old_frame_center[1]
            if idx1d != no_count_indicator:
                mybin = int(new_frame_center[0] + my_center // bin_factor)
                mxbin = int(new_frame_center[1] + mx_center // bin_factor)
                indices[ny, nx, k] = mybin * MXnew + mxbin
                counts[ny, nx, k] = old_counts[ny, nx, i]
                k += 1


def dense_to_sparse_kernel(dense, indices, counts, frame_dimensions):
    ny, nx = cuda.grid(2)
    NY, NX, MYBIN, MXBIN = dense.shape
    MY, MX = frame_dimensions
    if ny < NY and nx < NX:
        k = 0
        for mx in range(MX):
            for my in range(MY):
                idx1d = my * MX + mx
                if dense[ny,nx,my,mx] > 0:
                    indices[ny,nx,k] = idx1d
                    counts[ny,nx,k] = dense[ny,nx,my,mx]
                    k += 1

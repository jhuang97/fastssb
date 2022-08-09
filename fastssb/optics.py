import math
import cmath as cm
import numpy as np
import cupy as cp
import torch as th
from numba import cuda

def wavelength(E_eV):
    emass = 510.99906;  # electron rest mass in keV
    hc = 12.3984244;  # h*c
    lam = hc / math.sqrt(E_eV * 1e-3 * (2 * emass + E_eV * 1e-3))  # in Angstrom
    return lam


def DOF(alpha, E_eV):
    E0 = E_eV

    # Calculate wavelength and electron interaction parameter
    m = 9.109383 * 10 ** -31
    e = 1.602177 * 10 ** -19
    c = 299792458
    h = 6.62607 * 10 ** -34

    lam = h / math.sqrt(2 * m * e * E0) / math.sqrt(1 + e * E0 / 2 / m / c ** 2) * 10 ** 10
    DOF = 2 * lam / alpha ** 2
    return DOF


def get_qx_qy_1D(M, dx, dtype, fft_shifted=False):
    qxa = cp.fft.fftfreq(M[0], dx[0]).astype(dtype)
    qya = cp.fft.fftfreq(M[1], dx[1]).astype(dtype)
    if fft_shifted:
        qxa = cp.fft.fftshift(qxa)
        qya = cp.fft.fftshift(qya)
    return qxa, qya


def get_qx_qy_2D(M, dx, dtype, fft_shifted=False):
    qxa = cp.fft.fftfreq(M[0], dx[0]).astype(dtype)
    qya = cp.fft.fftfreq(M[1], dx[1]).astype(dtype)
    [qxn, qyn] = cp.meshgrid(qxa, qya)
    if fft_shifted:
        qxn = cp.fft.fftshift(qxn)
        qyn = cp.fft.fftshift(qyn)
    return qxn, qyn

def get_qx_qy_1D_th(M, dx, dtype, fft_shifted=False):
    qxa = np.fft.fftfreq(M[0], dx[0]).astype(dtype)
    qya = np.fft.fftfreq(M[1], dx[1]).astype(dtype)
    if fft_shifted:
        qxa = np.fft.fftshift(qxa)
        qya = np.fft.fftshift(qya)
    return th.as_tensor(np.stack([qxa, qya]))

def get_qx_qy_2D_th(M, dx, dtype, fft_shifted=False):
    qxa = np.fft.fftfreq(M[0], dx[0]).astype(dtype)
    qya = np.fft.fftfreq(M[1], dx[1]).astype(dtype)
    [qxn, qyn] = np.meshgrid(qxa, qya)
    if fft_shifted:
        qxn = np.fft.fftshift(qxn)
        qyn = np.fft.fftshift(qyn)
    return th.as_tensor(np.stack([qxn, qyn]))


def disk_overlap_function(Qx_all, Qy_all, Kx_all, Ky_all, aberrations, theta_rot, alpha, lam):
    n_batch = Qx_all.shape[0]
    xp = cp.get_array_module(Qx_all)
    Gamma = xp.zeros((n_batch,) + (Ky_all.shape[0], Kx_all.shape[0]), dtype=xp.complex64)
    gs = Gamma.shape
    threadsperblock = 2 ** 8
    blockspergrid = math.ceil(np.prod(gs) / threadsperblock)
    strides = cp.array((np.array(Gamma.strides) / (Gamma.nbytes / Gamma.size)).astype(np.int))
    disk_overlap_kernel[blockspergrid, threadsperblock](Gamma, strides, Qx_all, Qy_all, Kx_all, Ky_all, aberrations,
                                                        theta_rot, alpha, lam)
    cp.cuda.Device().synchronize()
    return Gamma


@cuda.jit
def disk_overlap_kernel(Γ, strides, Qx_all, Qy_all, Kx_all, Ky_all, aberrations, theta_rot, alpha, lam):
    def aperture2(qx, qy, lam, alpha_max):
        qx2 = qx ** 2
        qy2 = qy ** 2
        q = math.sqrt(qx2 + qy2)
        ktheta = math.asin(q * lam)
        return ktheta < alpha_max

    def chi3(qy, qx, lam, C):
        """
        Zernike polynomials in the cartesian coordinate system
        :param qx:
        :param qy:
        :param lam: wavelength in Angstrom
        :param C:   (12 ,)
        :return:
        """

        u = qx * lam
        v = qy * lam
        u2 = u ** 2
        u3 = u ** 3
        u4 = u ** 4
        # u5 = u ** 5

        v2 = v ** 2
        v3 = v ** 3
        v4 = v ** 4
        # v5 = v ** 5

        # aberr = Param()
        # aberr.C1 = C[0]
        # aberr.C12a = C[1]
        # aberr.C12b = C[2]
        # aberr.C21a = C[3]
        # aberr.C21b = C[4]
        # aberr.C23a = C[5]
        # aberr.C23b = C[6]
        # aberr.C3 = C[7]
        # aberr.C32a = C[8]
        # aberr.C32b = C[9]
        # aberr.C34a = C[10]
        # aberr.C34b = C[11]

        chi = 0

        # r-2 = x-2 +y-2.
        chi += 1 / 2 * C[0] * (u2 + v2)  # r^2
        # r-2 cos(2*phi) = x"2 -y-2.
        # r-2 sin(2*phi) = 2*x*y.
        chi += 1 / 2 * (C[1] * (u2 - v2) + 2 * C[2] * u * v)  # r^2 cos(2 phi) + r^2 sin(2 phi)
        # r-3 cos(3*phi) = x-3 -3*x*y'2. r"3 sin(3*phi) = 3*y*x-2 -y-3.
        chi += 1 / 3 * (C[5] * (u3 - 3 * u * v2) + C[6] * (3 * u2 * v - v3))  # r^3 cos(3phi) + r^3 sin(3 phi)
        # r-3 cos(phi) = x-3 +x*y-2.
        # r-3 sin(phi) = y*x-2 +y-3.
        chi += 1 / 3 * (C[3] * (u3 + u * v2) + C[4] * (v3 + u2 * v))  # r^3 cos(phi) + r^3 sin(phi)
        # r-4 = x-4 +2*x-2*y-2 +y-4.
        chi += 1 / 4 * C[7] * (u4 + v4 + 2 * u2 * v2)  # r^4
        # r-4 cos(4*phi) = x-4 -6*x-2*y-2 +y-4.
        chi += 1 / 4 * C[10] * (u4 - 6 * u2 * v2 + v4)  # r^4 cos(4 phi)
        # r-4 sin(4*phi) = 4*x-3*y -4*x*y-3.
        chi += 1 / 4 * C[11] * (4 * u3 * v - 4 * u * v3)  # r^4 sin(4 phi)
        # r-4 cos(2*phi) = x-4 -y-4.
        chi += 1 / 4 * C[8] * (u4 - v4)
        # r-4 sin(2*phi) = 2*x-3*y +2*x*y-3.
        chi += 1 / 4 * C[9] * (2 * u3 * v + 2 * u * v3)
        # r-5 cos(phi) = x-5 +2*x-3*y-2 +x*y-4.
        # r-5 sin(phi) = y*x"4 +2*x-2*y-3 +y-5.
        # r-5 cos(3*phi) = x-5 -2*x-3*y-2 -3*x*y-4.
        # r-5 sin(3*phi) = 3*y*x-4 +2*x-2*y-3 -y-5.
        # r-5 cos(5*phi) = x-5 -10*x-3*y-2 +5*x*y-4.
        # r-5 sin(5*phi) = 5*y*x-4 -10*x-2*y-3 +y-5.

        chi *= 2 * np.pi / lam

        return chi

    gs = Γ.shape
    N = gs[0] * gs[1] * gs[2]
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = n // strides[0]
    iky = (n - j * strides[0]) // strides[1]
    ikx = (n - (j * strides[0] + iky * strides[1])) // strides[2]

    if n < N:
        Qx = Qx_all[j]
        Qy = Qy_all[j]
        Kx = Kx_all[ikx]
        Ky = Ky_all[iky]

        Qx_rot = Qx * math.cos(theta_rot) - Qy * math.sin(theta_rot)
        Qy_rot = Qx * math.sin(theta_rot) + Qy * math.cos(theta_rot)

        Qx = Qx_rot
        Qy = Qy_rot

        chi = chi3(Ky, Kx, lam, aberrations)
        A = aperture2(Ky, Kx, lam, alpha) * cm.exp(-1j * chi)
        chi = chi3(Ky + Qy, Kx + Qx, lam, aberrations)
        Ap = aperture2(Ky + Qy, Kx + Qx, lam, alpha) * cm.exp(-1j * chi)
        chi = chi3(Ky - Qy, Kx - Qx, lam, aberrations)
        Am = aperture2(Ky - Qy, Kx - Qx, lam, alpha) * cm.exp(-1j * chi)

        Γ[j, iky, ikx] = A.conjugate() * Am - A * Ap.conjugate()


def single_sideband_reconstruction(G, Qx_all, Qy_all, Kx_all, Ky_all, aberrations, theta_rot, alpha_rad,
                                   Ψ_Qp, Ψ_Qp_left_sb, Ψ_Qp_right_sb, eps, lam):
    threadsperblock = 2 ** 8
    blockspergrid = math.ceil(np.prod(G.shape) / threadsperblock)
    strides = cp.array((np.array(G.strides) / (G.nbytes / G.size)).astype(np.int))
    scale = 1
    single_sideband_kernel[blockspergrid, threadsperblock](G, strides, Qx_all, Qy_all, Kx_all, Ky_all, aberrations,
                                                           theta_rot, alpha_rad, Ψ_Qp, Ψ_Qp_left_sb,
                                                           Ψ_Qp_right_sb, eps, lam, scale)
    cp.cuda.Device(Ψ_Qp.device).synchronize()


@cuda.jit
def single_sideband_kernel(G, strides, Qx_all, Qy_all, Kx_all, Ky_all, aberrations, theta_rot, alpha,
                           Ψ_Qp, Ψ_Qp_left_sb, Ψ_Qp_right_sb, eps, lam, scale):
    def aperture2(qx, qy, lam, alpha_max, scale):
        qx2 = qx ** 2
        qy2 = qy ** 2
        q = math.sqrt(qx2 + qy2)
        ktheta = math.asin(q * lam)
        return (ktheta < alpha_max) * scale

    def chi3(qy, qx, lam, C):
        """
        Zernike polynomials in the cartesian coordinate system
        :param qx:
        :param qy:
        :param lam: wavelength in Angstrom
        :param C:   (12 ,)
        :return:
        """

        u = qx * lam
        v = qy * lam
        u2 = u ** 2
        u3 = u ** 3
        u4 = u ** 4
        # u5 = u ** 5

        v2 = v ** 2
        v3 = v ** 3
        v4 = v ** 4
        # v5 = v ** 5

        # aberr = Param()
        # aberr.C1 = C[0]
        # aberr.C12a = C[1]
        # aberr.C12b = C[2]
        # aberr.C21a = C[3]
        # aberr.C21b = C[4]
        # aberr.C23a = C[5]
        # aberr.C23b = C[6]
        # aberr.C3 = C[7]
        # aberr.C32a = C[8]
        # aberr.C32b = C[9]
        # aberr.C34a = C[10]
        # aberr.C34b = C[11]

        chi = 0

        # r-2 = x-2 +y-2.
        chi += 1 / 2 * C[0] * (u2 + v2)  # r^2
        # r-2 cos(2*phi) = x"2 -y-2.
        # r-2 sin(2*phi) = 2*x*y.
        chi += 1 / 2 * (C[1] * (u2 - v2) + 2 * C[2] * u * v)  # r^2 cos(2 phi) + r^2 sin(2 phi)
        # r-3 cos(3*phi) = x-3 -3*x*y'2. r"3 sin(3*phi) = 3*y*x-2 -y-3.
        chi += 1 / 3 * (C[5] * (u3 - 3 * u * v2) + C[6] * (3 * u2 * v - v3))  # r^3 cos(3phi) + r^3 sin(3 phi)
        # r-3 cos(phi) = x-3 +x*y-2.
        # r-3 sin(phi) = y*x-2 +y-3.
        chi += 1 / 3 * (C[3] * (u3 + u * v2) + C[4] * (v3 + u2 * v))  # r^3 cos(phi) + r^3 sin(phi)
        # r-4 = x-4 +2*x-2*y-2 +y-4.
        chi += 1 / 4 * C[7] * (u4 + v4 + 2 * u2 * v2)  # r^4
        # r-4 cos(4*phi) = x-4 -6*x-2*y-2 +y-4.
        chi += 1 / 4 * C[10] * (u4 - 6 * u2 * v2 + v4)  # r^4 cos(4 phi)
        # r-4 sin(4*phi) = 4*x-3*y -4*x*y-3.
        chi += 1 / 4 * C[11] * (4 * u3 * v - 4 * u * v3)  # r^4 sin(4 phi)
        # r-4 cos(2*phi) = x-4 -y-4.
        chi += 1 / 4 * C[8] * (u4 - v4)
        # r-4 sin(2*phi) = 2*x-3*y +2*x*y-3.
        chi += 1 / 4 * C[9] * (2 * u3 * v + 2 * u * v3)
        # r-5 cos(phi) = x-5 +2*x-3*y-2 +x*y-4.
        # r-5 sin(phi) = y*x"4 +2*x-2*y-3 +y-5.
        # r-5 cos(3*phi) = x-5 -2*x-3*y-2 -3*x*y-4.
        # r-5 sin(3*phi) = 3*y*x-4 +2*x-2*y-3 -y-5.
        # r-5 cos(5*phi) = x-5 -10*x-3*y-2 +5*x*y-4.
        # r-5 sin(5*phi) = 5*y*x-4 -10*x-2*y-3 +y-5.

        chi *= 2 * np.pi / lam

        return chi

    gs = G.shape
    N = gs[0] * gs[1] * gs[2] * gs[3]
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iqy = n // strides[0]
    iqx = (n - iqy * strides[0]) // strides[1]
    iky = (n - (iqy * strides[0] + iqx * strides[1])) // strides[2]
    ikx = (n - (iqy * strides[0] + iqx * strides[1] + iky * strides[2])) // strides[3]

    if n < N:

        Qx = Qx_all[iqx]
        Qy = Qy_all[iqy]
        Kx = Kx_all[ikx]
        Ky = Ky_all[iky]

        Qx_rot = Qx * math.cos(theta_rot) - Qy * math.sin(theta_rot)
        Qy_rot = Qx * math.sin(theta_rot) + Qy * math.cos(theta_rot)

        Qx = Qx_rot
        Qy = Qy_rot

        A = aperture2(Ky, Kx, lam, alpha, scale) * cm.exp(-1j * chi3(Ky, Kx, lam, aberrations))
        chi_KplusQ = chi3(Ky + Qy, Kx + Qx, lam, aberrations)
        A_KplusQ = aperture2(Ky + Qy, Kx + Qx, lam, alpha, scale) * cm.exp(-1j * chi_KplusQ)
        chi_KminusQ = chi3(Ky - Qy, Kx - Qx, lam, aberrations)
        A_KminusQ = aperture2(Ky - Qy, Kx - Qx, lam, alpha, scale) * cm.exp(-1j * chi_KminusQ)

        Γ = A.conjugate() * A_KminusQ - A * A_KplusQ.conjugate()

        Kplus = math.sqrt((Kx + Qx) ** 2 + (Ky + Qy) ** 2)
        Kminus = math.sqrt((Kx - Qx) ** 2 + (Ky - Qy) ** 2)
        K = math.sqrt(Kx ** 2 + Ky ** 2)
        bright_field = K < alpha / lam
        double_overlap1 = (Kplus < alpha / lam) * bright_field * (Kminus > alpha / lam)
        double_overlap2 = (Kplus > alpha / lam) * bright_field * (Kminus < alpha / lam)

        Γ_abs = abs(Γ)
        take = Γ_abs > eps and bright_field
        if take:
            val = G[iqy, iqx, iky, ikx] * Γ.conjugate()
            cuda.atomic.add(Ψ_Qp.real, (iqy, iqx), val.real)
            cuda.atomic.add(Ψ_Qp.imag, (iqy, iqx), val.imag)
        if double_overlap1:
            val = G[iqy, iqx, iky, ikx] * Γ.conjugate()
            cuda.atomic.add(Ψ_Qp_left_sb.real, (iqy, iqx), val.real)
            cuda.atomic.add(Ψ_Qp_left_sb.imag, (iqy, iqx), val.imag)
        if double_overlap2:
            val = G[iqy, iqx, iky, ikx] * Γ.conjugate()
            cuda.atomic.add(Ψ_Qp_right_sb.real, (iqy, iqx), val.real)
            cuda.atomic.add(Ψ_Qp_right_sb.imag, (iqy, iqx), val.imag)
        if iqx == 0 and iqy == 0:
            val = abs(G[iqy, iqx, iky, ikx]) + 1j * 0
            cuda.atomic.add(Ψ_Qp.real, (iqy, iqx), val.real)
            cuda.atomic.add(Ψ_Qp_left_sb.real, (iqy, iqx), val.real)
            cuda.atomic.add(Ψ_Qp_right_sb.real, (iqy, iqx), val.real)
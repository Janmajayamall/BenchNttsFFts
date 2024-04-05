import numpy as np
import math
np.random.seed(0)
def schoolbook_negacylic_mul(a: np.ndarray, b: np.ndarray):
    n = a.shape[-1]
    c = np.zeros(shape=n, dtype=a.dtype)
    for i in range(n):
        for j in range(0, i+1):
            c[i] += a[j] * b[i - j]
        for j in range(0, n-(i+1)):
            c[i] -= a[j+i+1] * b[n - 1 - j]
    return c

def negacylic_mul_size_2n_fft(p1: np.ndarray, p2: np.ndarray):
    p1_floats = p1.astype(np.float64)
    p2_floats = p2.astype(np.float64)

    print(p1_floats, p1)

    p1_extended = np.concatenate([p1_floats, -p1_floats])
    p2_extended = np.concatenate([p2_floats, -p2_floats])

    p1_forward = np.fft.fft(p1_extended)
    p2_forward = np.fft.fft(p2_extended)

    out_forward = np.multiply(p1_forward, p2_forward)
    out_extended = np.fft.ifft(out_forward)

    out = (np.round(out_extended.real / np.float64(2))).astype(np.int32).astype(np.uint32)

    # out = out.astype(p1.dtype)
    # out = np.array([math.fmod(i, np.float64(1<<32)) for i in out.tolist()], dtype=np.uint32)
    return out[:p1.shape[-1]]

def powers_of_omega(N: int, powers: int, C: np.dtype) -> np.array:
    omega = np.exp(C(2 * math.pi * 1j)/C(N))
    return np.array(object=[omega ** i for i in range(powers)], dtype=omega.dtype)


def negacyclic_mul_folding_twisting_trick(p1: np.ndarray, p2: np.ndarray, C_type: np.dtype, F_type: np.dtype, Int_type: np.dtype, Uint_type: np.dtype) -> np.ndarray:
    N = len(p1)
    powers_omega = powers_of_omega(N=2*N, powers=N//2, C=C_type)
    powers_omega_inv = np.array(object=[C_type(1)/i for i in powers_omega], dtype=C_type)

    # fold
    p1_fold = np.zeros(shape=N//2, dtype=C_type)
    p1_fold.real = p1.astype(dtype=F_type)[:N//2]
    p1_fold.imag = p1.astype(dtype=F_type)[N//2:]
    p2_fold = np.zeros(shape=N//2, dtype=C_type)
    p2_fold.real = p2.astype(dtype=F_type)[:N//2]
    p2_fold.imag = p2.astype(dtype=F_type)[N//2:]

    # twist
    p1_fold_twisted = np.multiply(p1_fold, powers_omega)
    p2_fold_twisted = np.multiply(p2_fold, powers_omega)

    # FFT
    p1_forward = np.fft.fft(p1_fold_twisted)
    p2_forward = np.fft.fft(p2_fold_twisted)
    out_forward = np.multiply(p1_forward, p2_forward)
    out_fold_twisted = np.fft.ifft(out_forward)

    # untwist
    out_fold = np.multiply(out_fold_twisted, powers_omega_inv)

    # unfold
    out = (np.round(np.concatenate([out_fold.real, out_fold.imag], dtype=F_type))).astype(Int_type).astype(Uint_type)

    return out

def test():
    N = 8
    a = np.random.randint(low=0, high=1<<31, size=N, dtype=np.uint32)
    b = np.random.randint(low=0, high=1<<31, size=N, dtype=np.uint32)

    out_expected = schoolbook_negacylic_mul(a=a, b=b)

    # out = negacylic_mul_size_2n_fft(p1=a, p2=b)
    out = negacyclic_mul_folding_twisting_trick(p1=a, p2=b, C_type=np.complex128, F_type=np.float64, Int_type=np.int32, Uint_type=np.uint32)
    # print(out)
    print(out - out_expected)

test()
# print((np.float64(1<<61)).astype(np.uint64))
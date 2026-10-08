"""
optimise_acetone.py

optimise acetone at PBE0-D3(BJ)/def2-TZVPD, DF), check it is a true minimum, and print the
geometry ready to use with water clusters 

"""

import numpy as np
from pyscf import gto
from pyscf.geomopt import geometric_solver
from pyscf.hessian.thermo import harmonic_analysis
from gpu4pyscf import dft as gdft

"""
SETTINGS
"""

XC, DISP, BASIS = "PBE0", "d3bj", "def2-TZVPD"
CONV = {"convergence_energy": 1e-7, "convergence_grms": 1e-5,
        "convergence_gmax": 3e-5, "convergence_drms": 4e-5,
        "convergence_dmax": 6e-5}

# starting acetone geometry

labels = ['O1', 'C1', 'C2', 'C3', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6']
start = np.array([
    [ 0.32008, -0.61272, -1.49255],
    [ 0.13085,  0.22899, -0.62760],
    [ 1.28820,  0.77569,  0.15033],
    [-1.25828,  0.71331, -0.34608],
    [ 1.35357,  1.87359,  0.00010],
    [ 1.14584,  0.56262,  1.23041],
    [ 2.24072,  0.31139, -0.18357],
    [-1.99275,  0.20769, -1.00885],
    [-1.31418,  1.80824, -0.51996],
    [-1.52191,  0.49728,  0.71036],
])


def make_mf(mol):
    mf = gdft.RKS(mol).density_fit()
    mf.grids.level=5
    mf.xc, mf.disp, mf.conv_tol = XC, DISP, 1e-9
    return mf


"""
MAIN
"""

mol = gto.M(atom=[(l[0], tuple(c)) for l, c in zip(labels, start)],
            basis=BASIS, unit="Angstrom", verbose=3)

conv_ok, mol_opt = geometric_solver.kernel(make_mf(mol), maxsteps=200, **CONV)
print("converged:", conv_ok)

# frequency check - a true minimum has no imaginary modes
mf = make_mf(mol_opt)
mf.kernel()


freqs = harmonic_analysis(mol_opt, mf.Hessian().kernel())["freq_wavenumber"]
n_imag = int(np.sum(np.iscomplex(freqs) | (np.real(freqs) < 0)))
print("frequencies (cm-1):", np.round(np.real(freqs), 1))
print("imaginary modes:", n_imag)


xyz = mol_opt.atom_coords(unit="Ang")


np.save("acetone_opt.npy", xyz)

print("\nacetone = {")
for l, c in zip(labels, xyz):
    print(f"    '{l}': np.array([{c[0]:.5f}, {c[1]:.5f}, {c[2]:.5f}]),")
print("}")

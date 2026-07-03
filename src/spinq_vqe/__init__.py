"""
spinq_vqe
=========
Variational Quantum Simulation of Antiferromagnetic Hamiltonians.

Part of the ARPA Quantum Logical Systems (QONDRA) research program.

Modules
-------
kagome       : Kagome lattice graph builder and Heisenberg Hamiltonian constructor
ansatz       : HVA (primary), HEA, and MERA variational ansatze
vqe          : VQE runners — COBYLA (gradient-free, primary) + Adam (diagnostic)
entanglement : Von Neumann entropy and mutual information from VQE wavefunctions
surrogate    : MLP surrogate for spin Hall angle prediction from materials descriptors
qaoa         : QAOA circuit and optimizer for k-from-N spintronic material selection
utils        : Plotting helpers with consistent pastel palette
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spinq-vqe")
except PackageNotFoundError:
    __version__ = "dev"

__author__ = "ARPA Quantum Logical Systems (QONDRA)"
__all__ = [
    "kagome",
    "ansatz",
    "vqe",
    "entanglement",
    "surrogate",
    "qaoa",
    "utils",
]

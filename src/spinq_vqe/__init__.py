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
surrogate    : MLP surrogate for Materials Project spin Hall angle data (B2)
qaoa         : QAOA circuit and optimizer for SOC material composition problem (B2)
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

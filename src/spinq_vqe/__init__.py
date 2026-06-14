"""
spinq_vqe
=========
Variational Quantum Simulation of Antiferromagnetic Hamiltonians.

Part of the ARPA Quantum Logical Systems (QONDRA) research program.

Modules
-------
kagome          : Kagome lattice graph builder and Hamiltonian constructor
ansatz          : HEA and MERA-inspired variational ansatze
vqe             : VQE runner with JAX JIT and optimizer support
entanglement    : Von Neumann entropy and mutual information from VQE wavefunctions
surrogate       : MLP surrogate for Materials Project spin Hall angle data
qaoa            : QAOA circuit and optimizer for SOC material composition problem
utils           : Plotting helpers, Pauli string utilities
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

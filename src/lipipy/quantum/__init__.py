"""Layer 2: quantum computing, implemented from scratch on NumPy.

Simulation cost is explicit. A dense statevector for q qubits needs 16·2^q bytes
(complex128): 26 qubits ≈ 1 GiB, 30 qubits ≈ 16 GiB. Simulators refuse requests
above ``max_qubits`` with ``ResourceLimitError`` before allocating, and point to the
matrix-product-state simulator (memory O(q χ^2)). Parity tests against Qiskit /
PennyLane run when ``lipipy[quantum]`` is installed.

Catalog. ``[x]`` = implemented, ``[ ]`` = planned.

Foundations
-----------
[ ] Qubit states, Dirac notation helpers, Bloch-sphere coordinates
[ ] Gates: Pauli X/Y/Z, H, S, T, rotations RX/RY/RZ, U3, CNOT, CZ, SWAP, Toffoli
[ ] Tensor products, partial trace, fidelity, entanglement entropy

Circuits and simulators
-----------------------
[ ] ``Circuit`` builder (fluent API) with parameters for variational circuits
[ ] Statevector simulator (apply k-qubit gate in O(2^q) time without building 2^q x 2^q matrices)
[ ] Measurement, sampling shots, expectation values of Pauli strings
[ ] Density-matrix simulator with noise channels (depolarizing, amplitude/phase damping)
[ ] Matrix-product-state (tensor network) simulator
[ ] Clifford / stabilizer simulator (Gottesman-Knill, polynomial time)

Algorithms
----------
[ ] Deutsch-Jozsa, Bernstein-Vazirani, Simon
[ ] Quantum Fourier transform, phase estimation
[ ] Grover search, amplitude amplification and estimation
[ ] Shor's algorithm (small-number demonstration)
[ ] Quantum teleportation, superdense coding, BB84 key distribution

Variational and quantum machine learning
----------------------------------------
[ ] Parameter-shift gradients
[ ] VQE (variational quantum eigensolver) with Pauli Hamiltonians
[ ] QAOA for MaxCut and other QUBO problems
[ ] Quantum kernels and variational classifiers

Error correction
----------------
[ ] Bit-flip and phase-flip codes, Shor 9-qubit code, Steane code
[ ] Surface code basics (syndrome extraction, decoding toy examples)
"""

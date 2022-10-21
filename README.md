# VF2 Partial Layout Plugin

This repo contains an experimental transpiler pass VF2PartialLayout which
is for performing the layout stage of compilation (ie initial qubit selection).
This pass  leverages the vf2 implementation in rustworkx to find a layout using
subgraph isomorphsim of a partial interaction graph and the coupling map. Unlike
VF2Layout (and VF2PostLayout) it gives up on finding a perfect layout for the
full interaction graph and instead builds partial interaction graphs over the
circuit and tries to find the most complete interaction graph that has a
matching subgraph. The best performing qubits are then matched for this subgraph
(per the normal vf2 layout algorithm) and that is used as the core of the initial
layout.

## Installing

Right now the package is not published on pypi but you can install it with pip
still by running:

```
pip install git+https://github.com/mtreinish/vf2_partial_layout
```

## Usage

Once you have it installed you can use the `VF2PartialLayout` method by
specifying ``layout_method="vf2_partial"`` when calling transpile. For example:

```python3
from qiskit.circuit.library import QFT
from qiskit import transpile
from qiskit.providers.fake_provider import FakeAuckland
qc = QFT(4)
qc.measure_all()

tqc = transpile(qc, FakeAuckland(), layout_method='vf2_partial')
```


Note this pass is still experimental and also known to be quite memory hungry
in its current form. The eventual goal is to have this pass as part of
qiskit-terra after we prove it's effectiveness and decrease the memory overhead.

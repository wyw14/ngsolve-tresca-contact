The task is an application extension of NGSolve's three-dimensional P1 elasticity and contact modeling facilities. It uses synthetic loading paths on explicitly constructed tetrahedral meshes and a fixed-pair nodal Tresca interface model.

Upstream: [NGSolve](https://github.com/NGSolve/ngsolve), version `v6.2.2608`, source commit `37cb7f9049921c87deec9250fb444754d1e31ab7`, licensed under LGPL-2.1. The runtime distribution is NGSolve `6.2.2608`.

Relevant upstream components are `ngsolve.VectorH1`, symmetric-gradient elasticity bilinear forms, native mesh and vertex numbering, and `ContactBoundary` in `comp/contact.hpp`, `comp/contact.cpp` and `comp/python_comp.cpp`. The official contact tutorial is `docs/i-tutorials/unit-6.2-contact/contact.ipynb`; native normal-contact assembly tests are in `tests/pytest/test_contact.py`. The prescribed Tresca history and output API are defined by this task.

# Incremental Tresca contact of two elastic solids

Extend `tresca_contact.solve_history` to compute quasistatic loading histories for two tetrahedral elastic bodies joined by a unilateral frictional interface. Return the displacement, interface force, elastic energy and accumulated frictional dissipation at every prescribed increment.

## Discrete physical model

Use the supplied straight tetrahedral meshes with continuous vector P1 displacement and isotropic small-strain elasticity. For body `b`, the Lamé constants are `mu_b = E_b/(2(1+nu_b))` and `lambda_b = E_b*nu_b/((1+nu_b)(1-2*nu_b))`. Its strain is the symmetric displacement gradient and its energy is

`U_b(u) = integral [mu_b*epsilon(u):epsilon(u) + lambda_b/2*(div u)^2] dV`.

The meshes, contact pairs and contact areas stay fixed throughout the history. This is a discrete node-to-node, small-sliding interface model. There are no body forces, applied free-surface tractions or inertia. Vertex displacements listed in `fixed_vertices` are prescribed by `boundary_displacements`; all remaining displacement degrees of freedom satisfy elastic equilibrium with the interface forces. Assemble the P1 elasticity operator using NGSolve on the supplied meshes.

For a paired upper/lower vertex `i`, let `n` be the supplied unit normal, pointing from the lower body toward the upper body. The undeformed pair separation has no component in the tangent plane. Define

`g0_i = n dot (x_upper_i - x_lower_i)`,

`r_i^s = u_upper_i^s - u_lower_i^s`,

`g_i^s = g0_i + n dot r_i^s`,

`d_i^s = (I - n*n^T)(r_i^s - r_i^(s-1))`.

Before the first increment, all displacements and `r_i^(-1)` are zero. Each increment uses the displacement from the immediately preceding increment, including when a pair was open.

The force **on the upper vertex** is `f_i^s = p_i^s*n + t_i^s`; the lower vertex receives its negative. The normal law is `g_i^s >= 0`, `p_i^s >= 0`, `p_i^s*g_i^s = 0`. An open pair (`g_i^s > 0`) has `p_i^s = 0` and `t_i^s = 0`. At a closed pair (`g_i^s = 0`), `t_i^s` is tangential and obeys the given Tresca law

`t_i^s in -T_i * partial ||d_i^s||_2`, with `T_i = contact_areas[i] * tresca_thresholds[i]`.

Thus nonzero tangential increment gives `t_i^s = -T_i*d_i^s/||d_i^s||_2`, while a zero increment permits any tangential force of norm at most `T_i`. `tresca_thresholds` is a prescribed stress bound, independent of the normal pressure. It can be zero. The problem domain guarantees a unique physical incremental displacement and interface force; an active-set label at zero pressure is not part of the output.

The cumulative dissipation after increment `s` is

`D^s = sum_(j=0..s) sum_(i: g_i^j=0) T_i * ||d_i^j||_2`.

The requested elastic energy is `U_upper(u^s) + U_lower(u^s)`.

## Python interface

Implement the package at `/app/src/tresca_contact`; additional modules may be placed within `/app/src`. NGSolve, NumPy, SciPy, mpmath and gmpy2 are installed. Each scientific call has 180 seconds on 2 CPU and 4 GiB memory. Do not modify libraries or execution facilities outside the workspace.

Implement:

```python
solve_history(upper_mesh, lower_mesh, young_moduli, poisson_ratios,
              fixed_vertices, boundary_displacements, contact_pairs,
              normal, contact_areas, tresca_thresholds)
```

Inputs use SI units and real arrays:

| Argument | Meaning |
|---|---|
| `upper_mesh`, `lower_mesh` | Native `ngsolve.Mesh`, dimension 3, straight nondegenerate tetrahedra. Nodal ordering is the native vertex number `vertex.nr`. |
| `young_moduli`, `poisson_ratios` | Length-2 arrays in upper/lower order; Young's moduli in Pa. |
| `fixed_vertices` | A pair of nonempty 1D integer arrays of distinct prescribed vertex numbers, one array per body. |
| `boundary_displacements` | A pair of arrays `(S, Nfixed_b, 3)` in m, following the corresponding `fixed_vertices` order. These are total displacements, not increments. |
| `contact_pairs` | Integer array `(Nc, 2)` of upper/lower native vertex numbers. Each column is distinct; contact vertices are not prescribed vertices. |
| `normal` | A unit 3-vector common to all contact pairs. |
| `contact_areas` | Positive `(Nc,)` array in m², supplying the nodal interface weights. |
| `tresca_thresholds` | Nonnegative `(Nc,)` array in Pa. |

Return a dictionary with exactly these float-array entries, in the original vertex/contact/increment ordering:

| Key | Shape | Units |
|---|---|---|
| `upper_displacements` | `(S, Nupper, 3)` | m |
| `lower_displacements` | `(S, Nlower, 3)` | m |
| `contact_forces` | `(S, Nc, 3)` | N, global Cartesian force on the upper vertex |
| `cumulative_dissipation` | `(S,)` | J |
| `elastic_energies` | `(S,)` | J |

Calls are independent and must not mutate their inputs.

## Supported domain and numerical accuracy

There are 1–24 increments, 1–6 contact pairs and at most 64 vertices per body. Both bodies are connected, and their prescribed vertices remove all rigid motions. In canonical vertex Cartesian coordinates, the reduced free-displacement elasticity matrix is positive definite with spectral condition number at most `1e6`. All contact constraints use distinct free vertices. The physical incremental solution exists and is unique at every increment; a zero force/gap may have equivalent contact classifications.

Let `L` be the larger body diameter, using Euclidean distance between its vertices. Vertex position norms are at most `100*L`, `1e-5 <= L <= 1` m, `1e6 <= E_b <= 1e12` Pa, `max(E_b)/min(E_b) <= 100`, and `-0.2 <= nu_b <= 0.45`. The bodies are initially unpenetrated: `0 <= g0_i <= 1e-3*L`. All prescribed displacement-vector norms are at most `1e-3*L`. Contact-area weights are between `1e-6*L^2` and `L^2`; Tresca thresholds are between zero and `1e-2*min(E_b)` Pa. Mesh geometry and contact indices remain unchanged during a call. Arbitrary global rotations are supported.

Define `U = max(1e-6*L, max_i(g0_i), max_(b,s,j)||boundary_displacements[b][s,j]||_2)`, `F = max(E_b)*L*U`, and `W = F*U`. For each scalar output component, the accepted error is `3e-7*scale + 3e-6*abs(expected)`, using scales `U`, `F`, and `W` for displacement, force, and energy/dissipation respectively. Output must be finite and have the specified shapes. Malformed array shapes, nonfinite numerical array data, invalid material parameters, nonunit normals, negative thresholds, nonpositive areas and invalid vertex indices raise `ValueError`.

"""Incremental frictional contact between two tetrahedral elastic solids."""


def solve_history(upper_mesh, lower_mesh, young_moduli, poisson_ratios,
                  fixed_vertices, boundary_displacements, contact_pairs,
                  normal, contact_areas, tresca_thresholds):
    """Return nodal displacements, upper contact forces, dissipation and energy."""
    raise NotImplementedError("Implement the incremental contact history")

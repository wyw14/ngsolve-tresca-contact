"""Construct a straight tetrahedral native mesh from vertex-indexed arrays."""
from collections import Counter
import numpy as np
import ngsolve as ng
from netgen.meshing import Mesh, MeshPoint, Pnt, Element3D, Element2D, FaceDescriptor


def make_mesh(points, tets):
    native = Mesh(dim=3)
    for point in points:
        native.Add(MeshPoint(Pnt(*map(float, point))))
    native.SetMaterial(1, 'solid')
    native.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    native.SetBCName(0, 'outer')
    for element in tets:
        tet = np.array(element, dtype=int)
        if np.linalg.det((points[tet[1:]] - points[tet[0]]).T) < 0:
            tet[1], tet[2] = tet[2], tet[1]
        native.Add(Element3D(1, [int(i) + 1 for i in tet]))
    faces = Counter(tuple(sorted(np.delete(tet, i))) for tet in tets for i in range(4))
    for face, count in faces.items():
        if count == 1:
            native.Add(Element2D(1, [int(i) + 1 for i in face]))
    return ng.Mesh(native)


def arguments(problem):
    return (make_mesh(problem['upper_points'], problem['upper_tets']),
            make_mesh(problem['lower_points'], problem['lower_tets']),
            problem['young_moduli'], problem['poisson_ratios'],
            (problem['upper_fixed'], problem['lower_fixed']),
            (problem['upper_boundary'], problem['lower_boundary']),
            problem['contact_pairs'], problem['normal'], problem['contact_areas'], problem['tresca_thresholds'])

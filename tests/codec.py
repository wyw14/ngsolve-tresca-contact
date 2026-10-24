"""Data-only numeric wire encoding; array shape survives empty leading axes."""
import math
import numpy as np
import ngsolve as ng


def encode(value):
    if isinstance(value, ng.Mesh):
        points = np.empty((value.nv, 3), dtype=float)
        for vertex in value.vertices:
            points[vertex.nr] = tuple(vertex.point)
        tets = np.array([[vertex.nr for vertex in element.vertices]
                         for element in value.Elements(ng.VOL)], dtype=int)
        return {'__wire_ngsolve_mesh__': encode({'points': points, 'tets': tets})}
    if isinstance(value,float) and not math.isfinite(value):
        return {'__wire_float__': 'nan' if math.isnan(value) else '+inf' if value>0 else '-inf'}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biufc':
            raise ValueError('Only numeric arrays are supported')
        return {'__wire_array__': {'shape': list(value.shape), 'data': encode(value.ravel().tolist())}}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, complex):
        return {'__wire_complex__': [value.real, value.imag]}
    if isinstance(value, tuple):
        return {'__wire_tuple__': [encode(x) for x in value]}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {key: encode(x) for key, x in value.items()}
    return value


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if isinstance(value, dict):
        if set(value) == {'__wire_ngsolve_mesh__'}:
            data = decode(value['__wire_ngsolve_mesh__'])
            return make_mesh(data['points'], data['tets'])
        if set(value)=={'__wire_float__'}:
            return {'nan':float('nan'),'+inf':float('inf'),'-inf':float('-inf')}[value['__wire_float__']]
        if set(value) == {'__wire_complex__'}:
            return complex(*value['__wire_complex__'])
        if set(value) == {'__wire_tuple__'}:
            return tuple(decode(x) for x in value['__wire_tuple__'])
        if set(value) == {'__wire_array__'}:
            data=value['__wire_array__']
            return np.asarray(decode(data['data'])).reshape(tuple(data['shape']))
        return {key: decode(x) for key,x in value.items()}
    return value


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


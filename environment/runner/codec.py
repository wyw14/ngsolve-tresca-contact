"""Data-only numeric wire encoding; preserve malformed returned fields as data."""
import math
import numpy as np
import ngsolve as ng
from collections import Counter
from netgen.meshing import Mesh, MeshPoint, Pnt, Element3D, Element2D, FaceDescriptor


def encode(value):
    if isinstance(value, ng.Mesh):
        points = np.empty((value.nv, 3), dtype=float)
        for vertex in value.vertices:
            points[vertex.nr] = tuple(vertex.point)
        tets = np.array([[vertex.nr for vertex in element.vertices]
                         for element in value.Elements(ng.VOL)], dtype=int)
        return {'__wire_ngsolve_mesh__': encode({'points': points, 'tets': tets})}
    if isinstance(value, float) and not math.isfinite(value):
        return {'__wire_float__': 'nan' if math.isnan(value) else '+inf' if value > 0 else '-inf'}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biufc':
            raise ValueError('Only numeric arrays are supported')
        return {'__wire_array__': {'shape': list(value.shape), 'data': encode(value.ravel().tolist())}}
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, complex):
        return {'__wire_complex__': [encode(value.real), encode(value.imag)]}
    if isinstance(value, tuple):
        return {'__wire_tuple__': [encode(x) for x in value]}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {key: encode(x) for key, x in value.items()}
    return value


def encode_output(value, _depth=0):
    """A bad returned field is an ordinary mismatch, not a worker crash."""
    if _depth > 24:
        return {'__invalid_output__': 'nested value exceeds numeric output structure'}
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if not isinstance(key, str):
                key = '__invalid_output_key_' + str(index)
                while key in value or key in result:
                    key += '_'
            try:
                result[key] = encode_output(item, _depth + 1)
            except Exception:
                result[key] = {'__invalid_output__': 'unencodable field'}
        return result
    if isinstance(value, (list, tuple)):
        items = [encode_output(item, _depth + 1) for item in value]
        return {'__wire_tuple__': items} if isinstance(value, tuple) else items
    if isinstance(value, np.ndarray) and value.dtype.kind not in 'biufc':
        return {'__invalid_output__': 'nonnumeric array'}
    if isinstance(value, np.generic):
        return encode_output(value.item(), _depth + 1)
    if value is None or isinstance(value, (str, bool, int, float, complex, np.ndarray)):
        try:
            return encode(value)
        except Exception:
            return {'__invalid_output__': 'unencodable numeric value'}
    return {'__invalid_output__': 'unsupported value type'}


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if isinstance(value, dict):
        if set(value) == {'__wire_ngsolve_mesh__'}:
            data = decode(value['__wire_ngsolve_mesh__'])
            return make_mesh(data['points'], data['tets'])
        if set(value) == {'__wire_float__'}:
            return {'nan': float('nan'), '+inf': float('inf'), '-inf': float('-inf')}[value['__wire_float__']]
        if set(value) == {'__wire_complex__'}:
            return complex(*(decode(x) for x in value['__wire_complex__']))
        if set(value) == {'__wire_tuple__'}:
            return tuple(decode(x) for x in value['__wire_tuple__'])
        if set(value) == {'__wire_array__'}:
            data = value['__wire_array__']
            return np.asarray(decode(data['data'])).reshape(tuple(data['shape']))
        return {key: decode(x) for key, x in value.items()}
    return value


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


def decode_output(value, _depth=0):
    """Decode numeric returned data without reconstructing native objects."""
    if _depth > 24:
        return {'__invalid_output__': 'nested value exceeds numeric output structure'}
    if isinstance(value, list):
        return [decode_output(item, _depth+1) for item in value]
    if isinstance(value, dict):
        if set(value) == {'__wire_ngsolve_mesh__'}:
            return {'__invalid_output__': 'native mesh is not a numeric output',
                    'raw_native_wire': value['__wire_ngsolve_mesh__']}
        try:
            if set(value) == {'__wire_float__'}:
                return {'nan':float('nan'), '+inf':float('inf'), '-inf':float('-inf')}[value['__wire_float__']]
            if set(value) == {'__wire_complex__'}:
                return complex(*(decode_output(item, _depth+1) for item in value['__wire_complex__']))
            if set(value) == {'__wire_tuple__'}:
                return tuple(decode_output(item, _depth+1) for item in value['__wire_tuple__'])
            if set(value) == {'__wire_array__'}:
                item = value['__wire_array__']
                return np.asarray(decode_output(item['data'], _depth+1)).reshape(tuple(item['shape']))
        except Exception:
            return {'__invalid_output__': 'malformed numeric wire value', 'raw':value}
        return {key:decode_output(item, _depth+1) for key,item in value.items()}
    return value

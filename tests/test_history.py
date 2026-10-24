"""Independent discrete elasticity and primal friction-cone fixture checks."""
from pathlib import Path
import numpy as np
import pytest
from mesh_io import arguments
from candidate import call

def solve_history(*args):
    return call("solve_history", *args)

FIXTURES = Path(__file__).parent / 'fixtures'
CASES = ['cycle', 'frictionless', 'mixed-zero-gap', 'zero-pressure-shear', 'local-compliance', 'rotated', 'scaled']
CASES += ['micro-gap-rotation-'+str(i) for i in range(4)]
CASES += ['zero-pressure-rotated-stick']
KEYS = ['upper_displacements', 'lower_displacements', 'contact_forces', 'cumulative_dissipation', 'elastic_energies']


def load(name):
    with np.load(FIXTURES/(name+'-input.npz')) as source:
        data = {k: source[k].copy() for k in source.files}
    with np.load(FIXTURES/(name+'-expected.npz')) as source:
        expected = {k: source[k].copy() for k in source.files}
    return data, expected


def scales(data):
    points = [data['upper_points'], data['lower_points']]
    length = max(np.linalg.norm(p[:, None]-p[None, :], axis=2).max() for p in points)
    gaps = (points[0][data['contact_pairs'][:, 0]]-points[1][data['contact_pairs'][:, 1]]) @ data['normal']
    U = max(1e-6*length, max(gaps), *(np.linalg.norm(data[b+'_boundary'], axis=2).max() for b in ['upper', 'lower']))
    F = max(data['young_moduli'])*length*U
    return {key: U if 'displacements' in key else F if key=='contact_forces' else F*U for key in KEYS}


@pytest.fixture(scope='module', params=CASES)
def computed(request):
    data, expected = load(request.param)
    before = {key: value.copy() for key, value in data.items()}
    args = arguments(data)
    mesh_vertices = [np.array([tuple(v.point) for v in mesh.vertices]) for mesh in args[:2]]
    result = solve_history(*args)
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])
    for mesh, vertices in zip(args[:2], mesh_vertices):
        np.testing.assert_array_equal(np.array([tuple(v.point) for v in mesh.vertices]), vertices)
    assert isinstance(result, dict) and set(result) == set(KEYS)
    return request.param, data, expected, result


@pytest.mark.parametrize('field', KEYS)
def test_independent_physical_outputs(computed, field):
    name, data, expected, result = computed
    actual = np.asarray(result[field])
    assert actual.shape == expected[field].shape, (name, field, actual.shape)
    assert actual.dtype.kind == 'f' and np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected[field], rtol=3e-6, atol=3e-7*scales(data)[field])


def test_normal_and_friction_laws(computed):
    _, data, expected, result = computed
    up, low = data['contact_pairs'].T
    n = data['normal']; area = data['contact_areas']; limits = area*data['tresca_thresholds']
    jump = result['upper_displacements'][:, up]-result['lower_displacements'][:, low]
    initial = (data['upper_points'][up]-data['lower_points'][low]) @ n
    gap = initial+jump @ n
    force = result['contact_forces']; pressure = force @ n
    tangent = force-pressure[:, :, None]*n
    norms = np.linalg.norm(tangent, axis=2)
    U,F=scales(data)['upper_displacements'],scales(data)['contact_forces']
    du=3e-7*U+3e-6*max(np.max(abs(expected[k])) for k in ['upper_displacements','lower_displacements'])
    df=3e-7*F+3e-6*np.max(abs(expected['contact_forces']))
    tol_u,tol_f=2*np.sqrt(3)*du,np.sqrt(3)*df
    assert gap.min() >= -tol_u
    assert pressure.min() >= -tol_f
    assert np.all(norms <= limits[None, :]+tol_f)
    expected_jump=expected['upper_displacements'][:,up]-expected['lower_displacements'][:,low]
    expected_gap=initial+expected_jump@n
    expected_pressure=expected['contact_forces']@n
    complementarity_budget=(tol_u*(abs(expected_pressure)+tol_f)+tol_f*abs(expected_gap)
                             +abs(expected_gap*expected_pressure))
    assert np.all(abs(gap*pressure) <= complementarity_budget+np.finfo(float).eps*U*F)
    # Positive geometric opening carries no interface force.
    opened = gap > 10*tol_u
    assert np.all(np.linalg.norm(force[opened], axis=1) <= 10*tol_f)


def test_dissipation_is_accumulated(computed):
    _, data, expected, result = computed
    tol = 3e-7*scales(data)['cumulative_dissipation']+3e-6*np.max(abs(expected['cumulative_dissipation']))
    D = result['cumulative_dissipation']
    assert np.min(D) >= -tol and np.min(np.diff(D), initial=0.) >= -2*tol
    pairs, n = data['contact_pairs'], data['normal']
    jump = result['upper_displacements'][:, pairs[:, 0]]-result['lower_displacements'][:, pairs[:, 1]]
    change = np.diff(np.concatenate([np.zeros_like(jump[:1]), jump]), axis=0)
    slip = change-(change @ n)[:, :, None]*n
    force = result['contact_forces']
    tangent = force-(force @ n)[:, :, None]*n
    actual_work = -np.sum(tangent*slip, axis=(1,2))
    expected_jump=expected['upper_displacements'][:,pairs[:,0]]-expected['lower_displacements'][:,pairs[:,1]]
    expected_change=np.diff(np.concatenate([np.zeros_like(expected_jump[:1]),expected_jump]),axis=0)
    expected_slip=expected_change-(expected_change@n)[:,:,None]*n
    expected_force=expected['contact_forces']
    expected_tangent=expected_force-(expected_force@n)[:,:,None]*n
    du=3e-7*scales(data)['upper_displacements']+3e-6*max(np.max(abs(expected[k])) for k in ['upper_displacements','lower_displacements'])
    df=3e-7*scales(data)['contact_forces']+3e-6*np.max(abs(expected_force))
    slip_error,force_error=4*np.sqrt(3)*du,np.sqrt(3)*df
    work_error=np.sum(force_error*np.linalg.norm(expected_slip,axis=2)+(np.linalg.norm(expected_tangent,axis=2)+force_error)*slip_error,axis=1)
    baseline_error=abs(np.diff(np.r_[0.,expected['cumulative_dissipation']])+np.sum(expected_tangent*expected_slip,axis=(1,2)))
    assert np.all(abs(np.diff(np.r_[0.,D])-actual_work) <= 2*tol+work_error+baseline_error)


@pytest.mark.parametrize('index', [0, 1])
def test_single_increment_and_repeat_call(index):
    data, expected = load('cycle')
    for side in ['upper', 'lower']:
        data[side+'_boundary'] = data[side+'_boundary'][index:index+1].copy()
    args = arguments(data)
    first = solve_history(*args)
    second = solve_history(*args)
    for field in KEYS:
        scale = scales(data)[field]
        np.testing.assert_allclose(first[field], expected[field][index:index+1], rtol=3e-6, atol=3e-7*scale)
        np.testing.assert_allclose(second[field], first[field], rtol=3e-6, atol=3e-7*scale)


@pytest.mark.parametrize('kind', ['negative-threshold', 'zero-area', 'nonunit-normal', 'nonfinite-boundary', 'bad-vertex'])
def test_invalid_arrays_raise_value_error(kind):
    data, _ = load('cycle')
    if kind == 'negative-threshold': data['tresca_thresholds'][1] = -1
    elif kind == 'zero-area': data['contact_areas'][0] = 0
    elif kind == 'nonunit-normal': data['normal'] *= 2
    elif kind == 'nonfinite-boundary': data['upper_boundary'][0,0,0] = np.nan
    elif kind == 'bad-vertex': data['contact_pairs'][0,0] = 1000
    with pytest.raises(ValueError):
        solve_history(*arguments(data))

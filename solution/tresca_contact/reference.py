"""Small-sliding nodal Tresca contact using native P1 elasticity matrices."""
import itertools
import numpy as np
import ngsolve as ng
from scipy.linalg import block_diag
from scipy.sparse import coo_matrix
from .high_precision import PrecisionFallback,solve_precise_history


def stiffness(mesh, young, poisson):
    space = ng.VectorH1(mesh, order=1)
    u, v = space.TnT()
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    form = ng.BilinearForm(space, symmetric=True)
    form += (2 * mu * ng.InnerProduct(ng.Sym(ng.Grad(u)), ng.Sym(ng.Grad(v)))
             + lam * ng.div(u) * ng.div(v)) * ng.dx
    form.Assemble()
    order = np.array([list(space.GetDofNrs(ng.NodeId(ng.VERTEX, i)))
                      for i in range(mesh.nv)], dtype=int).ravel()
    rows, cols, data = form.mat.COO()
    matrix = coo_matrix((data, (rows, cols)), shape=(space.ndof, space.ndof)).toarray()
    return matrix[np.ix_(order, order)]


def tangent_frame(normal):
    axis = np.eye(3)[np.argmin(abs(normal))]
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    return np.array([first, np.cross(normal, first)])


def project_disks(value, limits):
    value = np.asarray(value).reshape(-1, 2)
    norms = np.linalg.norm(value, axis=1)
    factor = np.minimum(1., np.divide(limits, norms, out=np.ones_like(norms), where=norms > 0))
    return (value * factor[:, None]).ravel()


def active_solution(inverse, free, normal, tangent, gap, previous, limits, active, with_error=False):
    count = len(gap)
    pressure, shear = np.zeros(count), np.zeros(2 * count)
    displacement_error, pressure_error = np.zeros(len(free)), np.zeros(count)
    if not len(active):
        result = (free, pressure, shear)
        return (*result, displacement_error, pressure_error) if with_error else result
    rows = (2 * active[:, None] + np.arange(2)).ravel()
    N, T = normal[active], tangent[rows]
    W = N @ inverse @ N.T
    rhs = -gap[active] - N @ free
    fixed_normal = free + inverse @ N.T @ np.linalg.solve(W, rhs)
    projected = inverse - inverse @ N.T @ np.linalg.solve(W, N @ inverse)
    displacement_transfer = projected @ T.T
    pressure_transfer = np.linalg.solve(W, N @ inverse @ T.T)
    H = T @ projected @ T.T
    H = (H + H.T) / 2
    c = T @ fixed_normal - previous[rows]
    eig = np.linalg.eigvalsh(H)
    if eig[0] <= 0:
        raise ValueError("Independent contact constraints are required")
    L = eig[-1]
    beta = (np.sqrt(L) - np.sqrt(eig[0])) / (np.sqrt(L) + np.sqrt(eig[0]))
    old = np.zeros(len(rows))
    y = old.copy()
    # The projected residual also controls displacement error through compliance.
    for _ in range(200000):
        new = project_disks(y - (H @ y + c) / L, limits[active])
        gradient_step = (H @ new + c) / L
        projected_residual = new - project_disks(new - gradient_step, limits[active])
        # Strong convexity bounds the dual error from the projected gradient
        # residual. Propagate that error through the actual local compliance.
        dual_error = L / eig[0] * np.linalg.norm(projected_residual)
        displacement_error = np.linalg.norm(displacement_transfer, axis=1) * dual_error
        active_pressure_error = np.linalg.norm(pressure_transfer, axis=1) * dual_error
        estimate_u = fixed_normal + displacement_transfer @ new
        estimate_p = np.linalg.solve(W, rhs - N @ inverse @ T.T @ new)
        force_roundoff = 256 * np.finfo(float).eps * max(np.max(abs(new)), np.max(abs(gradient_step)))
        if (np.max(abs(projected_residual)) <= force_roundoff
                and np.max(displacement_error) <= 1e-9 * (1 + np.max(abs(estimate_u)))
                and np.max(active_pressure_error) <= 1e-9 * (1 + np.max(abs(estimate_p)))
                and dual_error <= 1e-9 * (1 + np.max(abs(new)))):
            break
        y = new + beta * (new - old)
        old = new
    else:
        raise RuntimeError("Friction solve did not reach its numerical residual")
    shear[rows] = new
    pressure[active] = np.linalg.solve(W, rhs - N @ inverse @ T.T @ new)
    displacement = free + inverse @ (normal.T @ pressure + tangent.T @ shear)
    pressure_error[active] = active_pressure_error
    result = (displacement, pressure, shear)
    return (*result, displacement_error, pressure_error) if with_error else result


def local_error_bounds(K, inverse, force, normal, tangent, gap, u, p, t, active):
    """Componentwise KKT backward error, without a global force/gap floor."""
    N = normal[active]
    q = len(active)
    n = len(u)
    if q:
        W = N @ inverse @ N.T
        W_inverse = np.linalg.inv(W)
        coupling = inverse @ N.T @ W_inverse
        A_inverse = np.block([[inverse-coupling@N@inverse, coupling],
                              [-W_inverse@N@inverse, W_inverse]])
        A = np.block([[K, -N.T], [N, np.zeros((q,q))]])
        b = np.r_[force+tangent.T@t, -gap[active]]
        x = np.r_[u, p[active]]
    else:
        A, A_inverse, b, x = K, inverse, force+tangent.T@t, u
    gamma = 64 * (n + q + 8) * np.finfo(float).eps
    residual = abs(b-A@x)
    rounding = gamma * (abs(b)+abs(A)@abs(x))
    error = abs(A_inverse) @ (residual+rounding)
    pressure_error = np.zeros(len(gap))
    if q:
        pressure_error[active] = error[n:]
    return error[:n], pressure_error, gamma


def contact_step(K, force, normal, tangent, gap, previous, limits):
    inverse = np.linalg.inv(K)
    free = inverse @ force
    accepted = []
    for bits in itertools.product([False, True], repeat=len(gap)):
        active = np.flatnonzero(bits)
        u, p, t, friction_u_error, friction_p_error = active_solution(
            inverse, free, normal, tangent, gap, previous, limits, active, with_error=True)
        opening = gap + normal @ u
        u_error, p_error, gamma = local_error_bounds(K, inverse, force, normal, tangent, gap, u, p, t, active)
        u_error += friction_u_error
        p_error += friction_p_error
        gap_error = abs(normal)@u_error + gamma*(abs(gap)+abs(normal)@abs(u))
        if np.any(opening < -gap_error) or np.any(p < -p_error):
            continue
        if np.any(abs(opening[active]) > gap_error[active]):
            continue
        inactive = np.flatnonzero(~np.asarray(bits))
        if any(abs(p[i]) <= p_error[i] and (p[i] != 0 or p_error[i] != 0) for i in active):
            raise PrecisionFallback("Normal pressure sign is unresolved in native arithmetic")
        if any(abs(opening[i]) <= gap_error[i] and (opening[i] != 0 or gap_error[i] != 0) for i in inactive):
            raise PrecisionFallback("Open-gap sign is unresolved in native arithmetic")
        slip = (tangent @ u - previous).reshape(-1, 2)
        slip_error = (abs(tangent)@u_error + gamma*(abs(tangent)@abs(u)+abs(previous))).reshape(-1,2)
        # At a geometrically closed zero-pressure point, the tangential law
        # still applies; a zero-force inactive classification is permissible
        # only when it satisfies that law.
        if any(opening[i] <= gap_error[i] and limits[i] > 0
               and np.linalg.norm(slip[i]) > np.linalg.norm(slip_error[i]) for i in inactive):
            continue
        accepted.append((u, p, t, active))
    if not accepted:
        raise RuntimeError("No admissible incremental contact solution")
    for other in accepted[1:]:
        if max(np.max(abs(a - b)) for a, b in zip(accepted[0][:3], other[:3])) > 2e-7:
            raise PrecisionFallback("Native contact branches disagree")
    return accepted[0]


def _solve_native(upper_mesh, lower_mesh, young_moduli, poisson_ratios,
                  fixed_vertices, boundary_displacements, contact_pairs,
                  normal, contact_areas, tresca_thresholds):
    young, poisson = np.asarray(young_moduli, float), np.asarray(poisson_ratios, float)
    n = np.asarray(normal, float)
    areas, thresholds = np.asarray(contact_areas, float), np.asarray(tresca_thresholds, float)
    pairs = np.asarray(contact_pairs)
    meshes = [upper_mesh, lower_mesh]
    vertices = [np.array([tuple(v.point) for v in mesh.vertices]) for mesh in meshes]
    sizes = [len(v) for v in vertices]
    if len(fixed_vertices) != 2 or len(boundary_displacements) != 2:
        raise ValueError("Two bodies' boundary arrays are required")
    fixed_nodes = [np.asarray(v) for v in fixed_vertices]
    boundary = [np.asarray(v, float) for v in boundary_displacements]
    if (young.shape != (2,) or poisson.shape != (2,) or not np.all(np.isfinite(young))
            or not np.all(np.isfinite(poisson)) or np.any(young <= 0) or np.any(poisson <= -1)
            or np.any(poisson >= .5)):
        raise ValueError("Invalid isotropic material parameters")
    if n.shape != (3,) or not np.all(np.isfinite(n)) or abs(np.linalg.norm(n) - 1) > 1e-10:
        raise ValueError("normal must be a unit three-vector")
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not 1 <= len(pairs) <= 6 or pairs.dtype.kind not in 'iu':
        raise ValueError("contact_pairs must be an integer (Nc,2) array")
    m = len(pairs)
    if areas.shape != (m,) or thresholds.shape != (m,) or not np.all(np.isfinite(areas)) or not np.all(np.isfinite(thresholds)) or np.any(areas <= 0) or np.any(thresholds < 0):
        raise ValueError("Invalid contact areas or Tresca thresholds")
    if len(boundary) != 2 or boundary[0].ndim != 3:
        raise ValueError("Two boundary histories are required")
    steps = len(boundary[0])
    if not 1 <= steps <= 24:
        raise ValueError("History length outside supported domain")
    offsets = [0, 3 * sizes[0]]
    fixed = []
    for side in range(2):
        if meshes[side].dim != 3 or sizes[side] > 64:
            raise ValueError("Expected a supported three-dimensional mesh")
        ids = fixed_nodes[side]
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids) or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= sizes[side]):
            raise ValueError("Invalid prescribed vertex indices")
        if boundary[side].shape != (steps, len(ids), 3) or not np.all(np.isfinite(boundary[side])):
            raise ValueError("Invalid prescribed displacement history")
        if len(np.unique(pairs[:, side])) != m or np.any(pairs[:, side] < 0) or np.any(pairs[:, side] >= sizes[side]) or np.any(np.isin(pairs[:, side], ids)):
            raise ValueError("Contact vertices must be distinct free vertices")
        fixed.extend((offsets[side] + 3 * ids[:, None] + np.arange(3)).ravel())
    fixed = np.asarray(fixed, int)
    length = max(np.linalg.norm(p[:, None] - p[None, :], axis=2).max() for p in vertices)
    separation = vertices[0][pairs[:, 0]] - vertices[1][pairs[:, 1]]
    gaps = separation @ n
    if np.any(gaps < -1e-12 * length) or np.max(np.linalg.norm(separation - gaps[:, None] * n, axis=1)) > 1e-10 * length:
        raise ValueError("Contact pairs must match in the fixed tangent plane")
    gaps = np.maximum(gaps, 0.)
    scale_u = max(1e-6 * length, np.max(gaps), *(np.linalg.norm(v, axis=2).max() for v in boundary))
    scale_f = max(young) * length * scale_u
    frame = tangent_frame(n)
    total = 3 * sum(sizes)
    Bn, Bt = np.zeros((m, total)), np.zeros((2 * m, total))
    for i, (up, lo) in enumerate(pairs):
        Bn[i, 3 * up:3 * up + 3] = n
        Bn[i, offsets[1] + 3 * lo:offsets[1] + 3 * lo + 3] = -n
        Bt[2 * i:2 * i + 2, 3 * up:3 * up + 3] = frame
        Bt[2 * i:2 * i + 2, offsets[1] + 3 * lo:offsets[1] + 3 * lo + 3] = -frame
    K = block_diag(*(stiffness(mesh, E, nu) for mesh, E, nu in zip(meshes, young, poisson)))
    free = np.setdiff1d(np.arange(total), fixed)
    reduced = K[np.ix_(free, free)] * scale_u / scale_f
    if np.linalg.eigvalsh(reduced)[0] <= 0:
        raise ValueError("Prescribed vertices must eliminate rigid-body motions")
    previous = np.zeros(total)
    cumulative = 0.
    output = {key: [] for key in ['upper_displacements', 'lower_displacements', 'contact_forces', 'cumulative_dissipation', 'elastic_energies']}
    for index in range(steps):
        full = np.zeros(total)
        full[fixed] = np.concatenate([v[index].ravel() for v in boundary]) / scale_u
        force = -K[np.ix_(free, fixed)] @ full[fixed] * scale_u / scale_f
        try:
            u, p, t, active = contact_step(reduced, force, Bn[:, free], Bt[:, free],
                                         gaps / scale_u, Bt @ previous, areas * thresholds / scale_f)
        except RuntimeError as error:
            if isinstance(error,PrecisionFallback):
                raise
            raise RuntimeError(f"Increment {index}: {error}") from error
        full[free] = u
        slip = (Bt @ (full - previous)).reshape(m, 2)
        cumulative += sum(areas[i] * thresholds[i] * np.linalg.norm(slip[i]) * scale_u for i in active)
        physical = full * scale_u
        output['upper_displacements'].append(physical[:offsets[1]].reshape(sizes[0], 3))
        output['lower_displacements'].append(physical[offsets[1]:].reshape(sizes[1], 3))
        output['contact_forces'].append(scale_f * (p[:, None] * n + t.reshape(m, 2) @ frame))
        output['cumulative_dissipation'].append(cumulative)
        output['elastic_energies'].append(.5 * physical @ K @ physical)
        previous = full
    return {key: np.asarray(value, dtype=float) for key, value in output.items()}


def solve_history(upper_mesh, lower_mesh, young_moduli, poisson_ratios,
                  fixed_vertices, boundary_displacements, contact_pairs,
                  normal, contact_areas, tresca_thresholds):
    args=(upper_mesh,lower_mesh,young_moduli,poisson_ratios,fixed_vertices,
          boundary_displacements,contact_pairs,normal,contact_areas,tresca_thresholds)
    try:
        return _solve_native(*args)
    except PrecisionFallback:
        return solve_precise_history(*args,assemble=stiffness)

"""Exact rational certificates for fixed-normal, linear sticking subproblems.

The raw normal need not have unit length.  ``pressure_raw_q[i]`` is the
multiplier of that raw normal in the force on the upper body; its sign is
the physical pressure sign.  All returned Cartesian quantities use SI units.
An over-limit sticking trial is inconclusive about the sliding solution.
"""
import time

from gmpy2 import mpq


def _q(value):
    if isinstance(value, mpq):
        return value
    if hasattr(value, "as_integer_ratio"):
        numerator, denominator = value.as_integer_ratio()
        return mpq(int(numerator), int(denominator))
    return mpq(value)


def _dot(left, right):
    return sum((a*b for a, b in zip(left, right)), mpq(0))


def _cross(left, right):
    return [left[1]*right[2]-left[2]*right[1],
            left[2]*right[0]-left[0]*right[2],
            left[0]*right[1]-left[1]*right[0]]


class _SparseLDL:
    """Unpivoted exact LDL^T for a positive-definite rational matrix."""

    def __init__(self, lower, check):
        self.check = check
        self.n = len(lower)
        work = [{int(j): _q(a) for j, a in row.items() if a}
                for row in lower]
        self.rows = [dict() for _ in work]
        self.diagonal = []
        for k in range(self.n):
            check()
            diagonal = work[k].get(k, mpq(0))
            if diagonal <= 0:
                raise ValueError("Exact certificate requires a positive-definite matrix")
            self.diagonal.append(diagonal)
            neighbors = [(i, work[i][k]) for i in range(k+1, self.n)
                         if work[i].get(k, 0)]
            for i, value in neighbors:
                self.rows[i][k] = value/diagonal
            updates = 0
            for position, (i, value_i) in enumerate(neighbors):
                multiplier = self.rows[i][k]
                row = work[i]
                for j, value_j in neighbors[:position+1]:
                    updated = row.get(j, mpq(0))-multiplier*value_j
                    if updated:
                        row[j] = updated
                    else:
                        row.pop(j, None)
                    updates += 1
                    if updates % 32 == 0:
                        check()
                row.pop(k, None)
            work[k].clear()
        self.columns = [[] for _ in work]
        for i, row in enumerate(self.rows):
            for j, value in row.items():
                self.columns[j].append((i, value))

    def solve(self, rhs):
        if len(rhs) != self.n:
            raise ValueError("Exact linear right-hand side has the wrong length")
        result = [_q(x) for x in rhs]
        for i, row in enumerate(self.rows):
            self.check()
            result[i] -= sum((a*result[j] for j, a in row.items()), mpq(0))
        for i, diagonal in enumerate(self.diagonal):
            result[i] /= diagonal
        for i in range(self.n-1, -1, -1):
            self.check()
            result[i] -= sum((a*result[j] for j, a in self.columns[i]), mpq(0))
        return result


class ExactLinear:
    """Lazy exact elasticity and sticking certificates for one input history.

    ``rows_q`` is the full Cartesian stiffness as sparse dictionaries.  ``free``
    and ``fixed`` are global Cartesian DOF numbers, and ``offset`` is the lower
    body's global DOF offset.  ``raw_gaps_q[i]`` is the initial separation
    dotted with ``normal_q``.  The absolute monotonic ``deadline`` is shared
    with the caller; exhaustion raises ``TimeoutError`` rather than returning
    a certificate.  This class has no dependency on the precision module.
    """

    def __init__(self, rows_q, free, fixed, pairs, offset, normal_q,
                 raw_gaps_q, deadline):
        self.deadline = deadline
        self.rows_q = rows_q
        self.free = [int(x) for x in free]
        self.fixed = [int(x) for x in fixed]
        self.fixed_set = set(self.fixed)
        self.offset = int(offset)
        self.pairs = [(int(up), int(lo)) for up, lo in pairs]
        self.normal = [_q(x) for x in normal_q]
        self.raw_gaps = [_q(x) for x in raw_gaps_q]
        self.total = len(rows_q)
        self.index = {value: i for i, value in enumerate(self.free)}
        if (len(self.normal) != 3 or not any(self.normal)
                or len(self.raw_gaps) != len(self.pairs)
                or len(self.index) != len(self.free)
                or len(self.fixed_set) != len(self.fixed)
                or self.fixed_set.intersection(self.index)
                or set(self.index).union(self.fixed_set) != set(range(self.total))):
            raise ValueError("Invalid exact certificate geometry or DOF partition")
        axis = [mpq(0), mpq(0), mpq(0)]
        axis[min(range(3), key=lambda k: abs(self.normal[k]))] = mpq(1)
        first = _cross(self.normal, axis)
        self.directions = [self.normal, first, _cross(self.normal, first)]
        self._elastic = None
        self._contact_rows = {}
        self._contact_columns = {}
        self._gram_factors = {}
        self._check()

    def _check(self):
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise TimeoutError("Exact sticking certificate budget exhausted")

    def _elastic_factor(self):
        if self._elastic is None:
            lower = []
            for i, global_i in enumerate(self.free):
                self._check()
                row = {}
                for global_j, value in self.rows_q[global_i].items():
                    j = self.index.get(int(global_j))
                    if j is not None and j <= i and value:
                        row[j] = _q(value)
                lower.append(row)
            self._elastic = _SparseLDL(lower, self._check)
        return self._elastic

    def _constraint(self, key):
        if key not in self._contact_rows:
            pair, component = key
            up, lo = self.pairs[pair]
            row = {}
            for k, value in enumerate(self.directions[component]):
                if value:
                    try:
                        row[self.index[3*up+k]] = value
                        row[self.index[self.offset+3*lo+k]] = -value
                    except KeyError as error:
                        raise ValueError("Contact vertices must be free") from error
            self._contact_rows[key] = row
        return self._contact_rows[key]

    def _column(self, key):
        if key not in self._contact_columns:
            rhs = [mpq(0) for _ in self.free]
            for i, value in self._constraint(key).items():
                rhs[i] = value
            self._contact_columns[key] = self._elastic_factor().solve(rhs)
        return self._contact_columns[key]

    def _gram_factor(self, keys):
        if keys not in self._gram_factors:
            columns = [self._column(key) for key in keys]
            lower = []
            for i, key in enumerate(keys):
                self._check()
                row = self._constraint(key)
                lower.append({j: sum((a*columns[j][k] for k, a in row.items()), mpq(0))
                              for j in range(i+1)})
            self._gram_factors[keys] = _SparseLDL(lower, self._check)
        return self._gram_factors[keys]

    def solve(self, boundary_q, previous_relative_q, active, areas_q,
              thresholds_q):
        """Return an exact valid/invalid/not_stick branch certificate.

        ``boundary_q`` contains full global physical displacements (free
        entries are ignored).  ``previous_relative_q`` has one Cartesian
        physical relative-displacement vector per pair.  ``not_stick`` means
        at least one sticking force exceeds its Tresca disk: that normal
        branch must still be checked by a sliding solver.
        """
        self._check()
        boundary = [_q(value) for value in boundary_q]
        if len(boundary) != self.total:
            raise ValueError("Full exact boundary vector has the wrong length")
        if hasattr(previous_relative_q, "rows") and hasattr(previous_relative_q, "cols"):
            previous = [[_q(previous_relative_q[i, k]) for k in range(3)]
                        for i in range(len(self.pairs))]
        else:
            previous = [[_q(x) for x in row] for row in previous_relative_q]
        areas = [_q(x) for x in areas_q]
        thresholds = [_q(x) for x in thresholds_q]
        count = len(self.pairs)
        if (len(previous) != count or any(len(row) != 3 for row in previous)
                or len(areas) != count or len(thresholds) != count
                or any(a <= 0 for a in areas) or any(t < 0 for t in thresholds)):
            raise ValueError("Invalid exact contact arrays")
        requested = [int(i) for i in active]
        if len(set(requested)) != len(requested) or any(i < 0 or i >= count for i in requested):
            raise ValueError("Invalid exact normal active set")
        active = sorted(requested)
        active_set = set(active)
        limits = [a*t for a, t in zip(areas, thresholds)]
        keys = tuple((i, component) for i in active
                     for component in ([0, 1, 2] if limits[i] else [0]))
        forcing = []
        for i in self.free:
            self._check()
            forcing.append(-sum((_q(a)*boundary[int(j)]
                                 for j, a in self.rows_q[i].items()
                                 if int(j) in self.fixed_set), mpq(0)))
        u_free = self._elastic_factor().solve(forcing)
        multiplier = []
        if keys:
            rhs = []
            for i, component in keys:
                target = (-self.raw_gaps[i] if component == 0
                          else _dot(self.directions[component], previous[i]))
                row = self._constraint((i, component))
                rhs.append(target-sum((a*u_free[j] for j, a in row.items()), mpq(0)))
            multiplier = self._gram_factor(keys).solve(rhs)
            for key, value in zip(keys, multiplier):
                self._check()
                if value:
                    column = self._column(key)
                    for j in range(len(u_free)):
                        u_free[j] += column[j]*value
        u = boundary[:]
        for local, global_i in enumerate(self.free):
            u[global_i] = u_free[local]
        forces = [[mpq(0) for _ in range(3)] for _ in self.pairs]
        tangent = [[mpq(0) for _ in range(3)] for _ in self.pairs]
        pressure = [mpq(0) for _ in self.pairs]
        for (i, component), value in zip(keys, multiplier):
            if component == 0:
                pressure[i] = value
            for k, direction in enumerate(self.directions[component]):
                forces[i][k] += value*direction
                if component:
                    tangent[i][k] += value*direction
        relative = [[u[3*up+k]-u[self.offset+3*lo+k] for k in range(3)]
                    for up, lo in self.pairs]
        gaps = [self.raw_gaps[i]+_dot(self.normal, relative[i])
                for i in range(count)]
        # If this linear trial is outside a friction disk, a different sliding
        # minimizer may change every normal multiplier; do not reject the set.
        if any(_dot(tangent[i], tangent[i]) > limits[i]**2 for i in active):
            status = "not_stick"
        else:
            if any(gaps[i] != 0 for i in active):
                raise ArithmeticError("Exact normal equality was not satisfied")
            invalid = any(gap < 0 for gap in gaps) or any(pressure[i] < 0 for i in active)
            for i in range(count):
                if i in active_set or gaps[i] != 0 or not limits[i]:
                    continue
                increment = [relative[i][k]-previous[i][k] for k in range(3)]
                # A zero tangential force is admissible at a closed point with
                # positive threshold precisely when its tangential increment
                # vanishes.  An unnormalised rational normal suffices here.
                if any(_cross(self.normal, increment)):
                    invalid = True
            status = "invalid" if invalid else "valid"
        self._check()
        return {"status": status, "u_q": u, "force_cart_q": forces,
                "relative_cart_q": relative, "gaps_raw_q": gaps,
                "pressure_raw_q": pressure}

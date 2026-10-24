"""Precision escalation for contact decisions unresolved by native arithmetic."""
import itertools
import time
import numpy as np
import mpmath as mp
from gmpy2 import mpq


class PrecisionFallback(RuntimeError):
    pass


class UnresolvedSign(RuntimeError):
    pass


def rational(value):
    if isinstance(value, (int, np.integer)):
        return mpq(int(value))
    return mpq(*float(value).as_integer_ratio())


def number(value):
    if isinstance(value, mpq):
        return mp.mpf(int(value.numerator)) / int(value.denominator)
    return mp.mpf(float(value))


def qsolve(A, b):
    n = len(b)
    rows = [list(row) + [rhs] for row, rhs in zip(A, b)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        diagonal = rows[j][j]
        for k in range(j, n + 1):
            rows[j][k] /= diagonal
        for i in range(n):
            if i == j or not rows[i][j]:
                continue
            factor = rows[i][j]
            for k in range(j, n + 1):
                rows[i][k] -= factor * rows[j][k]
    return [row[-1] for row in rows]


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), 0)


def absm(value):
    return value.apply(abs)


def exact_stiffness(meshes, young, poisson):
    sizes = [mesh.nv for mesh in meshes]
    rows = [dict() for _ in range(3*sum(sizes))]
    points = []
    for side, mesh in enumerate(meshes):
        xyz = [[rational(x) for x in vertex.point] for vertex in mesh.vertices]
        points.append(xyz)
        E, nu = rational(young[side]), rational(poisson[side])
        mu, lam = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
        offset = 0 if side == 0 else 3*sizes[0]
        import ngsolve as ng
        for element in mesh.Elements(ng.VOL):
            ids = [v.nr for v in element.vertices]
            coordinates = [xyz[i] for i in ids]
            edges = [[coordinates[j][k]-coordinates[0][k] for k in range(3)] for j in [1,2,3]]
            determinant = (edges[0][0]*(edges[1][1]*edges[2][2]-edges[1][2]*edges[2][1])
                           -edges[0][1]*(edges[1][0]*edges[2][2]-edges[1][2]*edges[2][0])
                           +edges[0][2]*(edges[1][0]*edges[2][1]-edges[1][1]*edges[2][0]))
            if determinant == 0:
                raise ValueError("Degenerate tetrahedron")
            gradients = [qsolve(edges, [mpq(int(i==j)) for i in range(3)]) for j in range(3)]
            gradients.insert(0, [-sum(g[k] for g in gradients) for k in range(3)])
            volume = abs(determinant)/6
            for a, ga in enumerate(gradients):
                for b, gb in enumerate(gradients):
                    scalar = mu*dot(ga,gb)
                    for i in range(3):
                        row = rows[offset+3*ids[a]+i]
                        for j in range(3):
                            value = volume*(scalar*(i==j)+mu*ga[j]*gb[i]+lam*ga[i]*gb[j])
                            column = offset+3*ids[b]+j
                            row[column] = row.get(column,mpq(0))+value
    return rows, points


class Elasticity:
    """Sparse high-precision residuals with a verified double preconditioner."""
    def __init__(self, rows_q, free, fixed, native, U, F, deadline):
        self.rows_q, self.free, self.fixed = rows_q, list(free), list(fixed)
        self.U, self.F, self.deadline = U,F,deadline
        index = {j:i for i,j in enumerate(free)}
        factor = U/F
        self.rows = [{index[j]:number(value)*factor for j,value in rows_q[k].items() if j in index and value}
                     for k in free]
        n = len(free)
        self.diagonal = [mp.sqrt(self.rows[i][i]) for i in range(n)]
        exact_float = np.zeros((n,n))
        for i,row in enumerate(self.rows):
            for j,value in row.items(): exact_float[i,j]=float(value/(self.diagonal[i]*self.diagonal[j]))
        d = np.array([float(x) for x in self.diagonal])
        native_scaled = (native/d[:,None])/d[None,:]
        self.R = np.linalg.inv(native_scaled)
        gamma = 8*n*np.finfo(float).eps
        difference = abs(exact_float-native_scaled)
        self.M = abs(np.eye(n)-self.R@native_scaled)+gamma*(abs(self.R)@abs(native_scaled))+abs(self.R)@difference
        self.q = float(np.max(np.sum(self.M,axis=1)))
        if not np.isfinite(self.q) or self.q >= .9:
            self.R = np.linalg.inv(exact_float)
            self.M = abs(np.eye(n)-self.R@exact_float)+gamma*(abs(self.R)@abs(exact_float))
            self.q = float(np.max(np.sum(self.M,axis=1)))
        if self.q >= 1:
            raise UnresolvedSign("Double preconditioner has no contraction bound")
        self.error_inverse = np.linalg.inv(np.eye(n)-self.M)

    def solve(self, rhs):
        rhs = mp.matrix(rhs)
        n, columns = rhs.rows,rhs.cols
        X = mp.zeros(n,columns)
        errors = mp.zeros(n,columns)
        iteration=0
        while True:
            iteration+=1
            if time.monotonic()>self.deadline: raise UnresolvedSign("Precision computation budget exhausted")
            residual = mp.zeros(n,columns)
            rounding = mp.zeros(n,columns)
            for i,row in enumerate(self.rows):
                for k in range(columns):
                    value = sum((a*X[j,k] for j,a in row.items()),mp.mpf(0))
                    residual[i,k] = rhs[i,k]-value
                    rounding[i,k] = 256*(n+8)*mp.eps*(abs(rhs[i,k])+sum((abs(a*X[j,k]) for j,a in row.items()),mp.mpf(0)))
            maximum = max((abs(x) for x in residual),default=mp.mpf(0))
            magnitude = max((abs(x) for x in rhs),default=mp.mpf(0))
            if maximum == 0 or maximum <= mp.power(10,-mp.mp.dps+15)*magnitude:
                for k in range(columns):
                    scaled=[(abs(residual[i,k])+rounding[i,k])/self.diagonal[i] for i in range(n)]
                    scale=max(scaled,default=mp.mpf(0))
                    if not scale: continue
                    bound=self.error_inverse@(abs(self.R)@np.array([float(v/scale) for v in scaled]))
                    for i in range(n):errors[i,k]=scale*mp.mpf(float(max(0.,bound[i])*1.000001))/self.diagonal[i]
                return X,errors
            for k in range(columns):
                scaled=[residual[i,k]/self.diagonal[i] for i in range(n)]
                scale=max((abs(x) for x in scaled),default=mp.mpf(0))
                if not scale:continue
                correction=self.R@np.array([float(x/scale) for x in scaled])
                for i in range(n):X[i,k]+=scale*mp.mpf(float(correction[i]))/self.diagonal[i]

    def forcing(self,boundary):
        fixed = {int(i):boundary[int(i)] for i in self.fixed}
        return mp.matrix([-sum((number(a)*fixed[j] for j,a in self.rows_q[i].items() if j in fixed),mp.mpf(0))/self.F for i in self.free])


def disks(value,limits):
    result=value.copy()
    for i,bound in enumerate(limits):
        norm=mp.sqrt(value[2*i]**2+value[2*i+1]**2)
        if norm>bound:
            for j in range(2):result[2*i+j]=value[2*i+j]*bound/norm
    return result


def friction(H,c,limits,deadline):
    dimension=len(c)
    if not dimension:return mp.matrix([]),mp.mpf(0),mp.mpf(1)
    eig=mp.eigsy(H,eigvals_only=True)
    mu,L=eig[0]/2,2*eig[dimension-1]
    if mu<=0:raise UnresolvedSign("Contact compliance positivity unresolved")
    t=mp.zeros(dimension,1)
    target=mp.power(10,-mp.mp.dps+20)
    for iteration in range(160):
        if time.monotonic()>deadline:raise UnresolvedSign("Friction precision budget exhausted")
        y=t-(H*t+c)/L;projected=disks(y,limits);residual=t-projected
        error=L/mu*mp.norm(residual)
        magnitude=max([abs(x) for x in t]+[abs(x) for x in c]+[abs(x) for x in limits],default=mp.mpf(0))
        if not error or error<=target*magnitude:return t,error,mu
        P=mp.zeros(dimension,dimension)
        for i,bound in enumerate(limits):
            norm=mp.sqrt(y[2*i]**2+y[2*i+1]**2)
            if not bound:continue
            if norm<bound:
                P[2*i,2*i]=P[2*i+1,2*i+1]=1
            else:
                for a in range(2):
                    for b in range(2):P[2*i+a,2*i+b]=bound/norm*((a==b)-y[2*i+a]*y[2*i+b]/norm**2)
        J=mp.eye(dimension)-P*(mp.eye(dimension)-H/L)
        try:direction=mp.lu_solve(J,-residual)
        except (ZeroDivisionError,ValueError):direction=projected-t
        old=mp.norm(residual);alpha=mp.mpf(1)
        for backtrack in range(30):
            trial=t+alpha*direction
            trial_residual=trial-disks(trial-(H*trial+c)/L,limits)
            if mp.norm(trial_residual)<old:
                t=trial;break
            alpha/=2
        else:t=projected
    raise UnresolvedSign("High-precision friction stationarity unresolved")


def submatrix(matrix,rows,columns):
    return mp.matrix([[matrix[i,j] for j in columns] for i in rows])


def replay(data,rows_q,points_q,native,precision,deadline):
    meshes,young,poisson,fixed_nodes,histories,pairs,normal,areas,thresholds=data
    sizes=[mesh.nv for mesh in meshes];offset=3*sizes[0];total=3*sum(sizes);m=len(pairs)
    fixed=np.r_[(3*np.asarray(fixed_nodes[0])[:,None]+np.arange(3)).ravel(),offset+(3*np.asarray(fixed_nodes[1])[:,None]+np.arange(3)).ravel()]
    free=np.setdiff1d(np.arange(total),fixed);free_index={int(j):i for i,j in enumerate(free)}
    n=mp.matrix([number(rational(x)) for x in normal]);n/=mp.norm(n)
    axis=mp.zeros(3,1);axis[min(range(3),key=lambda i:abs(n[i]))]=1
    first=mp.matrix([n[1]*axis[2]-n[2]*axis[1],n[2]*axis[0]-n[0]*axis[2],n[0]*axis[1]-n[1]*axis[0]])
    first/=mp.norm(first)
    second=mp.matrix([n[1]*first[2]-n[2]*first[1],n[2]*first[0]-n[0]*first[2],n[0]*first[1]-n[1]*first[0]])
    frame=[n,first,second]
    nq=[rational(x) for x in normal];norm_q=dot(nq,nq)
    raw_gaps=[dot(nq,[points_q[0][int(up)][k]-points_q[1][int(lo)][k] for k in range(3)]) for up,lo in pairs]
    gaps=mp.matrix([number(g)/mp.sqrt(number(norm_q)) for g in raw_gaps])
    pfloat=[np.array([[float(x) for x in row] for row in xyz]) for xyz in points_q]
    length=max(np.linalg.norm(p[:,None]-p[None,:],axis=2).max() for p in pfloat)
    U=number(rational(max(1e-6*length,float(max(gaps)),*(np.linalg.norm(v,axis=2).max() for v in histories))))
    F=number(rational(max(young)))*number(rational(length))*U
    E=Elasticity(rows_q,free,fixed,native[np.ix_(free,free)]*float(U/F),U,F,deadline)
    B=mp.zeros(3*m,len(free))
    for i,(up,lo) in enumerate(pairs):
        for component,direction in enumerate(frame):
            for k in range(3):
                B[3*i+component,free_index[3*int(up)+k]]=direction[k]
                B[3*i+component,free_index[offset+3*int(lo)+k]]=-direction[k]
    Q,Qerror=E.solve(B.T);C=B*Q
    gamma=mp.mpf(1024)*(len(free)+3*m+8)*mp.eps
    Cerror=absm(B)*Qerror+gamma*(absm(B)*absm(Q))
    limits=[number(rational(a))*number(rational(t))/F for a,t in zip(areas,thresholds)]
    previous=mp.zeros(3*m,1);previous_error=mp.zeros(3*m,1)
    cumulative=mp.mpf(0);output=[];certificates=[];cache={}
    from .exact_zero import ExactLinear
    exact=ExactLinear(rows_q,free,fixed,pairs,offset,nq,raw_gaps,deadline)
    areas_q=[rational(x) for x in areas];thresholds_q=[rational(x) for x in thresholds]
    rational_anchor=[[mpq(0)]*3 for _ in range(m)];pending_linear=[];lazy_open_boundary=None

    def rational_previous():
        nonlocal rational_anchor,pending_linear,lazy_open_boundary
        if lazy_open_boundary is not None:
            saved=exact.solve(lazy_open_boundary,[[mpq(0)]*3 for _ in range(m)],[],areas_q,thresholds_q)
            if any(g<0 for g in saved['gaps_raw_q']):raise UnresolvedSign("Exact open anchor disagrees with the precision result")
            rational_anchor=saved['relative_cart_q'];lazy_open_boundary=None
        if rational_anchor is None:return None
        state=rational_anchor
        for saved_boundary,saved_active in pending_linear:
            saved=exact.solve(saved_boundary,state,saved_active,areas_q,thresholds_q)
            if saved['status']!='valid':
                rational_anchor=None;pending_linear=[]
                return None
            state=saved['relative_cart_q']
        rational_anchor=state
        pending_linear=[]
        return rational_anchor

    for step in range(len(histories[0])):
        boundary_q=[mpq(0)]*total
        for side in range(2):
            for local,vertex in enumerate(fixed_nodes[side]):
                for k in range(3):boundary_q[(0 if side==0 else offset)+3*int(vertex)+k]=rational(histories[side][step,local,k])
        boundary=mp.matrix([number(x) for x in boundary_q])
        uf,uferror=E.solve(E.forcing(boundary));b=B*uf;berror=absm(B)*uferror+gamma*(absm(B)*absm(uf))
        accepted=[];uncertain=False;exact_selected=None
        for bits in itertools.product([False,True],repeat=m):
            active=[i for i,flag in enumerate(bits) if flag]
            nr=[3*i for i in active];tr=[3*i+j for i in active for j in [1,2]]
            v=mp.zeros(3*m,1);verror=mp.zeros(3*m,1)
            if active:
                key=tuple(active)
                if key not in cache:
                    W=submatrix(C,nr,nr);Winv=W**-1
                    H=submatrix(C,tr,tr)-submatrix(C,tr,nr)*Winv*submatrix(C,nr,tr)
                    Ew=submatrix(Cerror,nr,nr)
                    inverse_residual=absm(mp.eye(len(nr))-W*Winv)
                    inverse_majorant=absm(Winv)*Ew
                    ratio=max(sum(inverse_majorant[i,j] for j in range(len(nr))) for i in range(len(nr)))
                    if ratio>=mp.mpf('.5'):raise UnresolvedSign("Normal compliance inverse error unresolved")
                    Einv=(absm(Winv)*(inverse_residual+Ew*absm(Winv)+gamma*(absm(W)*absm(Winv))))/(1-ratio)
                    Ctn,Cnt=submatrix(C,tr,nr),submatrix(C,nr,tr)
                    Etn,Ent=submatrix(Cerror,tr,nr),submatrix(Cerror,nr,tr)
                    Herror=(submatrix(Cerror,tr,tr)+Etn*(absm(Winv)+Einv)*(absm(Cnt)+Ent)
                            +absm(Ctn)*Einv*(absm(Cnt)+Ent)+absm(Ctn)*absm(Winv)*Ent
                            +gamma*(submatrix(absm(C),tr,tr)+absm(Ctn)*absm(Winv)*absm(Cnt)))
                    cache[key]=(Winv,H,Einv,Herror)
                Winv,H,Einv,Herror=cache[key]
                rhs=mp.matrix([-gaps[i]/U-b[3*i] for i in active])
                rhs_error=mp.matrix([berror[3*i]+gamma*abs(gaps[i]/U) for i in active])
                c=mp.matrix([b[i]-previous[i] for i in tr])+submatrix(C,tr,nr)*Winv*rhs
                cerror=(mp.matrix([berror[i]+previous_error[i] for i in tr])
                        +submatrix(Cerror,tr,nr)*(absm(Winv)+Einv)*(absm(rhs)+rhs_error)
                        +submatrix(absm(C),tr,nr)*Einv*(absm(rhs)+rhs_error)
                        +submatrix(absm(C),tr,nr)*absm(Winv)*rhs_error
                        +gamma*(mp.matrix([abs(b[i])+abs(previous[i]) for i in tr])+submatrix(absm(C),tr,nr)*absm(Winv)*absm(rhs)))
                t,terror,mu=friction(H,c,[limits[i] for i in active],deadline)
                if mp.norm(Herror)>=mu/2:raise UnresolvedSign("Tangential compliance perturbation unresolved")
                terror+=mp.norm(Herror*absm(t)+cerror)/(mu-mp.norm(Herror))
                p=Winv*(rhs-submatrix(C,nr,tr)*t)
                rhs_error+=submatrix(Cerror,nr,nr)*absm(p)+submatrix(Cerror,nr,tr)*absm(t)
                rhs_error+=submatrix(absm(C),nr,tr)*mp.matrix([terror]*len(tr))
                rhs_error+=gamma*(absm(rhs)+submatrix(absm(C),nr,nr)*absm(p)+submatrix(absm(C),nr,tr)*absm(t))
                perror=absm(Winv)*rhs_error
                for local,i in enumerate(nr):v[i]=p[local];verror[i]=perror[local]
                for local,i in enumerate(tr):v[i]=t[local];verror[i]=terror
            relative=b+C*v
            relative_error=berror+Cerror*absm(v)+absm(C)*verror+gamma*(absm(b)+absm(C)*absm(v))
            opening=mp.matrix([gaps[i]/U+relative[3*i] for i in range(m)])
            gerror=mp.matrix([relative_error[3*i]+gamma*abs(gaps[i]/U) for i in range(m)])
            if any(opening[i]<-gerror[i] for i in range(m)) or any(v[3*i]<-verror[3*i] for i in active):continue
            candidate_uncertain=False;bad=False
            for i in active:
                if abs(v[3*i])<=verror[3*i] and (v[3*i]!=0 or verror[3*i]!=0):candidate_uncertain=True
            for i in range(m):
                if bits[i]:continue
                if opening[i]>gerror[i]:continue
                if opening[i]==0 and gerror[i]==0:
                    slip=mp.sqrt(sum((relative[3*i+j]-previous[3*i+j])**2 for j in [1,2]))
                    if limits[i] and slip:bad=True;break
                else:candidate_uncertain=True
            if bad:continue
            if candidate_uncertain:
                prior_q=rational_previous()
                if prior_q is not None:
                    certificate=exact.solve(boundary_q,prior_q,active,areas_q,thresholds_q)
                    if certificate['status']=='valid':
                        exact_selected=certificate;break
                    if certificate['status']=='invalid':continue
                uncertain=True;continue
            accepted.append((uf+Q*v,v,relative,active,relative_error,verror))
        if exact_selected is not None:
            full_q=exact_selected['u_q'];forces_q=exact_selected['force_cart_q'];relative_q=exact_selected['relative_cart_q']
            u=mp.matrix([number(full_q[int(i)])/U for i in free])
            v=mp.matrix([dot(direction,[number(x) for x in force])/F for force in forces_q for direction in frame])
            relative=mp.matrix([dot(direction,[number(x) for x in row])/U for row in relative_q for direction in frame])
            relative_error=gamma*mp.matrix([dot([abs(x) for x in direction],[abs(number(x)) for x in row])/U for row in relative_q for direction in frame])
            rational_anchor=relative_q;pending_linear=[];lazy_open_boundary=None
            accepted=[(u,v,relative,active,relative_error,mp.zeros(3*m,1))]
        if uncertain and not accepted:raise UnresolvedSign(f"Increment {step}: a normal branch sign still spans zero at {precision} digits")
        if not accepted:raise RuntimeError("No admissible high-precision incremental contact solution")
        u,v,relative,active,relative_error,verror=accepted[0]
        for other in accepted[1:]:
            if max([abs(a-b) for a,b in zip(u,other[0])]+[abs(a-b) for a,b in zip(v,other[1])],default=mp.mpf(0))>mp.power(10,-precision+20):
                raise ValueError("Distinct high-precision physical contact solutions")
        if exact_selected is None:
            for i in active:
                slip=mp.sqrt(sum((relative[3*i+j]-previous[3*i+j])**2 for j in [1,2]))
                cumulative+=limits[i]*slip*F*U
        full=boundary.copy()
        for local,index in enumerate(free):full[int(index)]=u[local]*U
        forces=[]
        for i in range(m):forces.append([F*sum(v[3*i+j]*frame[j][k] for j in range(3)) for k in range(3)])
        energy=sum((full[i]*sum((number(a)*full[j] for j,a in row.items()),mp.mpf(0)) for i,row in enumerate(rows_q)),mp.mpf(0))/2
        if exact_selected is not None:
            full=mp.matrix([number(x) for x in full_q]);forces=[[number(x) for x in row] for row in forces_q]
            energy=number(sum((full_q[i]*sum((a*full_q[j] for j,a in row.items()),mpq(0)) for i,row in enumerate(rows_q)),mpq(0))/2)
        elif not active:
            # Save a linear anchor without performing a large rational factor
            # unless a later exact-zero decision actually needs it.
            rational_anchor=None;pending_linear=[];lazy_open_boundary=boundary_q
        elif all(not limits[i] or mp.sqrt(v[3*i+1]**2+v[3*i+2]**2)<limits[i]-mp.sqrt(verror[3*i+1]**2+verror[3*i+2]**2) for i in active):
            if rational_anchor is not None or lazy_open_boundary is not None:pending_linear.append((boundary_q,active))
        else:rational_anchor=None;pending_linear=[];lazy_open_boundary=None
        output.append({'upper_displacements':np.array([float(x) for x in full[:offset]]).reshape(sizes[0],3),
                       'lower_displacements':np.array([float(x) for x in full[offset:]]).reshape(sizes[1],3),
                       'contact_forces':np.array([[float(x) for x in row] for row in forces]),
                       'cumulative_dissipation':float(cumulative),'elastic_energies':float(energy)})
        certificates.append({'step':step,'precision':precision,'active':active,'preconditioner_q':E.q,'exact_linear_certificate':exact_selected is not None})
        previous=relative;previous_error=relative_error
    return {key:np.array([row[key] for row in output]) for key in output[0]},certificates


def solve_precise_history(upper_mesh,lower_mesh,young,poisson,fixed,histories,pairs,normal,areas,thresholds,assemble):
    start=time.monotonic();deadline=start+150
    meshes=[upper_mesh,lower_mesh]
    rows,points=exact_stiffness(meshes,young,poisson)
    from scipy.linalg import block_diag
    native=block_diag(*(assemble(mesh,E,nu) for mesh,E,nu in zip(meshes,young,poisson)))
    data=(meshes,young,poisson,fixed,histories,np.asarray(pairs),normal,areas,thresholds)
    last=None
    for precision in [80,160,320,640]:
        if time.monotonic()>deadline:break
        with mp.workdps(precision):
            try:return replay(data,rows,points,native,precision,deadline)[0]
            except UnresolvedSign as error:last=error
    raise RuntimeError(f"High-precision contact decision remains unresolved: {last}")

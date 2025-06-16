import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular, lapack
from scipy.interpolate import RectBivariateSpline

def evalF(vx, vy, h, nx, ny):
    F = np.zeros((nx - 2, ny - 2))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            nonlinear1 = h / 2 * vx[i, j] * (vx[i + 1, j] - vx[i - 1, j])
            nonlinear2 = h / 2 * vy[i, j] * (vx[i, j + 1] - vx[i, j - 1])
            F[i - 1, j - 1] = vx[i, j] - 0.25 * (
                vx[i + 1, j]
                + vx[i - 1, j]
                + vx[i, j + 1]
                + vx[i, j - 1]
                - nonlinear1
                - nonlinear2
            )

    return F


def isDiagonallyDominant(J):
    n = J.shape[0]

    for i in range(n):
        sumaFila = np.sum(np.abs(J[i, :])) - np.abs(J[i, i])
        if np.abs(J[i, i]) < sumaFila:
            print(f"Suma de los valores de la fila: {sumaFila}")
            print(f"Valor comparado: {J[i, i]}")

    return True


def getDiagIndex(i, j, ny):
    return (i - 1) * (ny - 2) + (j - 1)


def evalJ(vx, vy, h, nx, ny):
    N = (nx - 2) * (ny - 2)
    J = np.zeros((N, N))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            index = getDiagIndex(i, j, ny)

            J[index, index] = 1 + (h / 8) * (vx[i + 1, j] - vx[i - 1, j])

            if i + 1 < nx - 1:
                J[index, getDiagIndex(i + 1, j, ny)] = - \
                    0.25 + (h / 8) * vx[i, j]

            if i - 1 >= 1:
                J[index, getDiagIndex(i - 1, j, ny)] = - \
                    0.25 - (h / 8) * vx[i, j]

            if j + 1 < ny - 1:
                J[index, getDiagIndex(i, j + 1, ny)] = - \
                    0.25 + (h / 8) * vy[i, j]

            if j - 1 >= 1:
                J[index, getDiagIndex(i, j - 1, ny)] = - \
                    0.25 - (h / 8) * vy[i, j]

    return J


def graphJacobian(J):
    plt.figure(figsize=(12, 7))

    plt.spy(J, markersize=1)
    plt.title("Estructura del Jacobiano")
    plt.xlabel("Columnas")
    plt.ylabel("Filas")

    plt.tight_layout()
    plt.show()


def solve(vx, vy, h, nx, ny, tol, method, maxIter=1000):
    for k in range(maxIter):
        F = evalF(vx, vy, h, nx, ny)

        normInfF = np.linalg.norm(F, ord=np.inf)
        
        if normInfF < tol:
            print(
                f"Convergencia alcanzada en {k+1} iteraciones con norma infinito = {normInfF}"
            )
            return k+1, vx

        J = evalJ(vx, vy, h, nx, ny)

        match method:
            case 1:
                H = gaussJordan(J, -F.flatten())

            case 2:
                H = richardson(J, -F.flatten())

            case 3:
                H = jacobi(J, -F.flatten())

            case 4:
                H = gaussSeidel(J, -F.flatten())

            case 5:
                H = descendantGradient(J, -F.flatten())

            case 6:
                H = conjugateGradient(J, -F.flatten())

        newVx = vx[1: nx - 1, 1: ny - 1].flatten() + H
        vx[1: nx - 1, 1: ny - 1] = newVx.reshape((nx - 2, ny - 2))

    print(
        f"El método no convergió en {maxIter} iteraciones, norma infinito = {normInfF}"
    )
    return maxIter, vx


def gaussJordan(A, b):
    _, _, x, _ = lapack.dgesv(A, b)

    return x


def richardson(A, b, x0=None, M=500, tol=1e-6):
    n = len(b)
    I = np.eye(n)

    convTerm = I - I @ A
    normInfConv = np.linalg.norm(convTerm, ord=np.inf)

    print(normInfConv)
    if normInfConv >= 1:
        print("Puede que Richardson no converga")

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    inverseQxbTerm = I @ b

    for k in range(M):
        xNew = convTerm @ x + inverseQxbTerm

        normInfX = np.linalg.norm(xNew - x)

        x = xNew.copy()

        if normInfX < tol:
            print(
                f"Richardson convergió en {k + 1} iteraciones con norma infinito = {normInfX}"
            )
            return x

    print(
        f"Richardson no convergió en {M} iteraciones, norma infinito = {normInfX}")
    return x


def jacobi(A, b, x0=None, M=500, tol=1e-6):
    n = len(b)
    I = np.eye(n)
    Q = np.diag(A)
    inverseQ = np.diag(1 / Q)

    convTerm = I - inverseQ @ A
    normInfConv = np.linalg.norm(convTerm, ord=np.inf)

    if normInfConv >= 1:
        print("Puede que Jacobi no converga")

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    inverseQxbTerm = inverseQ @ b

    for k in range(M):
        xNew = convTerm @ x + inverseQxbTerm

        normInfX = np.linalg.norm(xNew - x)

        x = xNew.copy()

        if normInfX < tol:
            print(
                f"Jacobi convergió en {k + 1} iteraciones con norma infinito = {normInfX}"
            )
            return x

    print(
        f"Jacobi no convergió en {M} iteraciones, norma infinito = {normInfX}")
    return x


def conjugateGradient(A, b, x0=None, M=500, tol=1e-6):
    n = len(b)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - A @ x
    p = r.copy()
    dotROld = np.dot(r, r)

    for k in range(M):
        Ap = A @ p
        a = np.dot(r, r) / np.dot(p, Ap)
        x += a * p
        r -= a * Ap
        dotRNew = np.dot(r, r)

        normInfR = np.linalg.norm(r, ord=np.inf)

        if normInfR < tol:
            print(
                f"Gradiente Conjugado convergió en {k + 1} iteraciones con norma infinito = {normInfR}"
            )
            return x

        b = dotRNew / dotROld
        p = r + b * p
        dotROld = dotRNew

    print(
        f"Gradiente Conjugado no convergió en {M} iteraciones, norma infinito = {normInfR}"
    )
    return x


def descendantGradient(A, b, x0=None, M=500, tol=1e-6):
    n = len(b)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    for k in range(M):
        r = b - A @ x

        normInfR = np.linalg.norm(r, ord=np.inf)

        a = np.dot(r, r) / np.dot(r, A @ r)

        x += a * r

        if normInfR < tol:
            print(
                f"Gradiente Descendiente convergió en {k + 1} iteraciones con norma infinito = {normInfR}"
            )
            return x

    print(
        f"Gradiente Descendiente no convergió en {M} iteraciones, norma infinito = {normInfR}"
    )
    return x


def gaussSeidel(A, b, x0=None, M=500, tol=1e-6):
    n = len(b)
    I = np.eye(n)
    Q = np.tril(A)
    inverseQ = solve_triangular(Q, I, lower=True)

    convTerm = I - inverseQ @ A
    normInfConv = np.linalg.norm(convTerm, ord=np.inf)

    if normInfConv >= 1:
        print("Puede que Gauss-Seidel no converga")

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    inverseQxbTerm = inverseQ @ b

    for k in range(M):
        xNew = convTerm @ x + inverseQxbTerm

        normInfX = np.linalg.norm(xNew - x)

        x = xNew.copy()

        if normInfX < tol:
            print(
                f"Gauss-Seidel convergió en {k + 1} iteraciones con norma infinito = {normInfX}"
            )
            return x

    print(
        f"Gauss-Seidel no convergió en {M} iteraciones, norma infinito = {normInfX}")
    return x


import matplotlib.pyplot as plt

def graphSolution(sol, nx, ny, method, numIter):
    methods = {
        1: "Gauss-Jordan (LU)",
        2: "Richardson",
        3: "Jacobi",
        4: "Gauss-Seidel",
        5: "Gradiente Descendiente",
        6: "Gradiente Conjugado"
    }

    method_name = methods.get(method, "Método Desconocido")
    
    plt.figure(figsize=(12, 5))
    plt.imshow(sol.T, aspect="auto", cmap="jet", extent=[0, nx - 1, 0, ny - 1])
    plt.colorbar(label="$v_x$")
    plt.title(f"Distribución de $v_x$ — Método: {method_name} — Iteraciones: {numIter}")
    plt.xlabel("Índice j")
    plt.ylabel("Índice i")
    plt.show()




def initVxVy(numPartitions, scalarValue, initialVxValue, initialVyValue):
    assert numPartitions <= (nx - 2), "Too many partitions for too little rows"

    vx = np.zeros((nx, ny))
    vy = np.zeros((nx, ny))

    vx[0, :] = initialVxValue
    vx[:, 0] = 0
    vx[:, -1] = 0

    vy[0, :] = initialVyValue
    vy[:, 0] = 0
    vy[:, -1] = 0

    regionLength = (nx - 2) // numPartitions

    for p in range(numPartitions):
        start = 1 + p * regionLength
        end = 1 + (p + 1) * regionLength if p < numPartitions - 1 else nx - 1
        valueVx = initialVxValue * (scalarValue ** (p + 1))
        valueVy = initialVyValue * (scalarValue ** (p + 1))

        vx[start:end, 1: ny - 1] = valueVx
        vy[start:end, 1: ny - 1] = valueVy

    return vx, vy

def isSymmetric(A):
    return np.allclose(A, A.T)

def applySpline(factor, nx, ny, sol, type):
    valsVx = np.linspace(0, 1, nx)
    valsVy = np.linspace(0, 1, ny)

    match type:
        case 1: 
            spline = RectBivariateSpline(valsVx, valsVy, sol, kx = 1, ky = 1)
            
        case 2:
            spline = RectBivariateSpline(valsVx, valsVy, sol, kx = 2, ky = 2)
            
        case 3:
            spline = RectBivariateSpline(valsVx, valsVy, sol, kx = 3, ky = 3)
            
    newNx, newNy = nx*factor, ny*factor
    newVx = np.linspace(0, 1, newNx)
    newVy = np.linspace(0, 1, newNy)

    return newNx, newNy, spline(newVx, newVy)

# Parámetros
tol = 1e-6
nx, ny = 60, 20
h = 0.001
method = 1
numPartitions, scalarValue = 6, 1 / 8
initialVxValue, initialVyValue = 1, 0.001

vx, vy = initVxVy(numPartitions, scalarValue, initialVxValue, initialVyValue)
numIter, sol = solve(vx, vy, h, nx, ny, tol, method)
newNx, newNy, interpolated_sol = applySpline(100, nx, ny, sol, 3)

graphSolution(interpolated_sol, newNx, newNy, method, numIter)
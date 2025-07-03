import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular, lapack
from scipy.interpolate import CubicSpline

# Función que inicializa las matrices vx y vy de acuerdo a unos valores iniciales,
# un número de particiones y un valor escalar.
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

# Función que crea una matriz F que modelará el sistema y la evalua de acuerdo a parámetros 
# iniciales (vx, vy, h) siguiendo la ecuación característica del problema.
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
    
# Función que crea una matriz J correspondiente al Jacobiano y la evalua de acuerdo
# a los valores de las matrices vx, vy y el valor h siguiendo los diferentes casos
# para la derivada definidos.
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

# Función que aplica Newton-Raphson. Toma argumentos como las matrices
# de velocidad, vx y vy, el step, h, las dimensiones de la matriz, nx y ny,
# la tolerancia para el criterio de parada, tol, el método a utilizar, method, y 
# el máximo número de iteraciones, maxIter.
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

# Función que hace uso de lapack.dgesv para encontrar el vector de
# corrección.
def gaussJordan(A, b):
    _, _, x, _ = lapack.dgesv(A, b)

    return x

# Función que aplica el método iterativo Richardson para el cálculo
# del vector de corrección
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

# Función que aplica el método iterativo Jacobi para el cálculo
# del vector de corrección
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

# Función que aplica el método de Krylov Grad. Conjugado para el cálculo
# del vector de corrección
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

# Función que aplica el método de Krylov Grad. Descendiente para el cálculo
# del vector de corrección
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

# Función que aplica el método iterativo Gauss-Seidel para el cálculo
# del vector de corrección
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

# Función que aplica un spline cúbico lineal a una matriz haciendo
# uso de un factor de reescalamiento dado.
def applySpline(sol, factor):
    nx, ny = sol.shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)   
    newNx = nx * factor
    newNy = ny * factor
    x_new = np.linspace(0, 1, newNx)
    y_new = np.linspace(0, 1, newNy)

    # Interpolación por filas
    interp_rows = np.zeros((nx * factor, ny))
    for j in range(ny):
        cs = CubicSpline(x, sol[:, j], bc_type='natural')
        interp_rows[:, j] = cs(x_new)

    # Interpolación por columnas
    interp_final = np.zeros((nx * factor, ny * factor))
    for i in range(nx * factor):
        cs = CubicSpline(y, interp_rows[i, :], bc_type='natural')
        interp_final[i, :] = cs(y_new)

    return newNx, newNy, interp_final

# Función que grafica el Jacobiano.
def graphJacobian(J):
    plt.figure(figsize=(12, 7))

    plt.spy(J, markersize=1)
    plt.title("Estructura del Jacobiano")
    plt.xlabel("Columnas")
    plt.ylabel("Filas")

    plt.tight_layout()
    plt.show()

# Función que verifica si una matriz es diagonalmente dominante.
def isDiagonallyDominant(J):
    n = J.shape[0]

    for i in range(n):
        sumaFila = np.sum(np.abs(J[i, :])) - np.abs(J[i, i])
        if np.abs(J[i, i]) < sumaFila:
            print(f"Suma de los valores de la fila: {sumaFila}")
            print(f"Valor comparado: {J[i, i]}")

    return True

# Función que grafica la solución final del problema.
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

# Función que convierte índices 2D a 1D.
def getDiagIndex(i, j, ny):
    return (i - 1) * (ny - 2) + (j - 1)

# Función que verifica si una matriz es simétrica.
def isSymmetric(A):
    return np.allclose(A, A.T)

# Parámetros
tol = 1e-6
nx, ny = 60, 20
h = 0.001
method = 4
numPartitions, scalarValue = 6, 1 / 8
initialVxValue, initialVyValue = 1, 0.001

vx, vy = initVxVy(numPartitions, scalarValue, initialVxValue, initialVyValue)
numIter, sol = solve(vx, vy, h, nx, ny, tol, method)
newNx, newNy, interpolatedSolution = applySpline(sol, 100)

graphSolution(interpolatedSolution, newNx, newNy, 4, numIter)

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

np.set_printoptions(precision=5, suppress=True)


class Function:
    """Data treatment class. it allows to treat data as a real function from R1
    in R1. This class was inspired by rocketpy.Function."""

    def __init__(
        self,
        X,
        Y=None,
        xlabel="Abscissas",
        ylabel="Ordenadas",
        name=None,
        intervalo=[0, 10],
        method="linear",
        extrapolation=None,
    ):
        if type(Y) == list or type(Y) == np.ndarray:
            self.__X_source__ = np.array(X)
            self.__Y_source__ = np.array(Y)
            self.method = method
            self.extrapolation = extrapolation
            self.I = [X[0], X[-1]]
            if method == "linear":
                self.f = self.splineLinear  # A spline Linear não precisa de processamento
            elif method == "MMQ":
                self.MMQ()
            elif method == "spline":
                self.splineSimples()
            elif method == "cubicSpline":
                self.cubicSpline()
            elif method == "interPol":
                self.interPolinomial()
            elif method == "Newton":
                self.polNewton()
            else:
                print("Method not found. Linear interpolation was assumed.")
                self.f = self.splineLinear
        else:
            self.f = X
            self.I = intervalo
            self.method = "userFunc"

        self.__X_source_label__ = xlabel
        self.__Y_source_label__ = ylabel
        self.name = name

    def __call__(self, *args):
        if len(args) == 0:
            return self.plot2D()
        else:
            return self.getValue(*args)

    def getValue(self, x):
        if self.method == "linear":
            if type(x) == float or type(x) == int or type(x) == np.float64:
                if x < self.__X_source__[0]:
                    return self.__Y_source__[0]
                elif x > self.__X_source__[-1]:
                    return self.__Y_source__[-1]
                else:
                    y = self.splineLinear(x)

            # treat for array
            else:
                y = np.zeros(len(x))
                for i in range(len(x)):
                    # extrapolate if necessary
                    if x[i] < self.__X_source__[0]:
                        y[i] = self.__Y_source__[0]
                    elif x[i] > self.__X_source__[-1]:
                        y[i] = self.__Y_source__[-1]
                    # no extrapolation needed
                    else:
                        y[i] = self.splineLinear(x[i])

        elif self.method == "cubicSpline":
            if type(x) == float or type(x) == int or type(x) == np.float64:
                if x < self.__X_source__[0]:
                    y = self.__Y_source__[0]
                elif x > self.__X_source__[-1]:
                    y = self.__Y_source__[-1]
                else:
                    y = self.f(x)

            # treat for array
            else:
                y = np.zeros(len(x))
                for i in range(len(x)):
                    # extrapolate if necessary
                    if x[i] < self.__X_source__[0]:
                        y[i] = self.__Y_source__[0]

                    elif x[i] > self.__X_source__[-1]:
                        y[i] = self.__Y_source__[-1]

                    # no extrapolation needed
                    else:
                        y[i] = self.f(x[i])

        elif (
            self.method == "userFunc"
            or self.method == "MMQ"
            or self.method == "spline"
            or self.method == "interPol"
            or self.method == "Newton"
        ):
            y = self.f(x)
        return y

    def plot2D(
        self,
        title="",
        lower=None,
        upper=None,
        export=False,
        xscale="linear",
        yscale="linear",
        display=True,
        style=None,
    ):
        # Função para gerar os plots que serão utilizados no relatório.
        if title == "":
            title = self.__X_source_label__ + " x " + self.__Y_source_label__
        if style == "matplotlib" or style == "science":
            if style == "science":
                plt.style.use("science")
            lower = self.I[0] if lower is None else lower
            upper = self.I[1] if upper is None else upper
            X = np.linspace(lower, upper, 2001)
            Y = self.getValue(X)

            plt.figure(figsize=(8.09016994375, 5))
            plt.plot(X, Y)
            plt.xlabel(self.__X_source_label__, fontsize=14)
            plt.ylabel(self.__Y_source_label__, fontsize=14)
            plt.title(title, fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.grid()

            if self.name:
                plt.legend([self.name])

            if type(export) == bool:
                if export:
                    plt.savefig(title + ".pdf")
            elif type(export) == str:
                plt.savefig(export + ".pdf")

            if display:
                plt.show()

        else:  # Uses plotly
            lower = self.I[0] if lower is None else lower
            upper = self.I[1] if upper is None else upper
            X = np.linspace(lower, upper, 10001)
            Y = self.getValue(X)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X, y=Y, name=self.name))
            fig.update_layout(
                title=title,
                xaxis_title=self.__X_source_label__,
                yaxis_title=self.__Y_source_label__,
            )
            if type(export) == bool:
                if export:
                    fig.write_image(title + ".svg")
            elif type(export) == str:
                fig.write_image(export + ".pdf")

            if display:
                fig.show()

    def plotparametric(self, title="", export=False, display=True, style="plotly"):
        # Função para gerar os plots que serão utilizados no relatório.
        X = self.__X_source__
        Y = self.__Y_source__
        if title == "":
            title = self.__X_source_label__ + " x " + self.__Y_source_label__

        if style == "plotly":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X, y=Y, name=self.name))
            fig.update_layout(
                title=title,
                xaxis_title=self.__X_source_label__,
                yaxis_title=self.__Y_source_label__,
            )
            if type(export) == bool:
                if export:
                    fig.write_image(title + ".png", scale=2, width=700, height=700)
            elif type(export) == str:
                fig.write_image(export, scale=2, width=700, height=700)
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            if display:
                fig.show()
        else:
            plt.figure()
            plt.plot(X, Y)
            plt.xlabel(self.__X_source_label__, fontsize=14)
            plt.ylabel(self.__Y_source_label__, fontsize=14)
            plt.title(title, fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.axis("equal")
            plt.grid()
            if display:
                plt.show()

    def compara2Plots(
        self,
        dataB,
        title="",
        lower=None,
        upper=None,
        export=False,
        xscale="linear",
        yscale="linear",
        display=True,
        style=None,
    ):
        # Função para gerar os plots que serão utilizados no relatório.
        if style == "matplotlib" or style == "science":
            if style == "science":
                plt.style.use("science")
            lower = self.I[0] if lower is None else lower
            upper = self.I[1] if upper is None else upper
            X = np.linspace(lower, upper, 2001)
            Y = self.getValue(X)

            plt.figure(figsize=(8.09016994375, 5))
            plt.plot(X, Y, dataB.X, dataB.Y)
            plt.xlabel(self.__X_source_label__, fontsize=14)
            plt.ylabel(self.__Y_source_label__, fontsize=14)
            plt.title(title, fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.grid()
            if self.name:
                plt.legend([self.name, dataB.name])

            if type(export) == bool:
                if export:
                    plt.savefig(title + ".pdf")
            elif type(export) == str:
                plt.savefig(export + ".pdf")

            plt.show()

        else:
            lower = self.I[0] if lower is None else lower
            upper = self.I[1] if upper is None else upper
            X = np.linspace(lower, upper, 10001)
            Y = self.getValue(X)
            Yb = dataB.getValue(X)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X, y=Y, name=self.name))
            fig.add_trace(go.Scatter(x=X, y=Yb, name=dataB.name))
            fig.update_layout(
                title=title,
                xaxis_title=self.__X_source_label__,
                yaxis_title=self.__Y_source_label__,
            )
            if type(export) == bool:
                if export:
                    fig.write_image(title + ".svg")
            elif type(export) == str:
                fig.write_image(export + ".pdf")

            if display:
                fig.show()
            # return fig

    def comparaNPlots(
        self,
        data,
        title="",
        lower=None,
        upper=None,
        export=False,
        xscale="linear",
        yscale="linear",
        display=True,
        style=None,
    ):
        # Function use to compare plots. Insert a lists of plots that should be compared with the main plot (function used to call the compare plot)
        if style == "matplotlib" or style == "science":
            if style == "science":
                plt.style.use("science")
            lower = self.I[0] if lower is None else lower
            upper = self.I[1] if upper is None else upper
            X = np.linspace(lower, upper, 2001)
            Y = self.getValue(X)

            plt.figure(figsize=(8.09016994375, 5))
            plt.plot(X, Y)
            for i in range(len(data)):
                plt.plot(X, data[i].getValue(X))
            plt.xlabel(self.__X_source_label__, fontsize=14)
            plt.ylabel(self.__Y_source_label__, fontsize=14)
            plt.title(title, fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.grid()
            if self.name:
                legenda = [self.name]
                for i in range(len(data)):
                    legenda.append(data[i].name)
                plt.legend(legenda)

            if type(export) == bool:
                if export:
                    plt.savefig(title + ".pdf")
            elif type(export) == str:
                plt.savefig(export + ".pdf")

            plt.show()

        else:
            lower = self.I[0] if lower is None else lower
            upper = self.I[1] if upper is None else upper
            X = np.linspace(lower, upper, 20001)
            Y = self.getValue(X)
            if title == "":
                title = self.__X_source_label__ + " x " + self.__Y_source_label__ + " Comparison "

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X, y=Y, name=self.name))

            for i in range(len(data)):
                fig.add_trace(go.Scatter(x=X, y=data[i].getValue(X), name=data[i].name))

            fig.update_layout(
                title=title,
                xaxis_title=self.__X_source_label__,
                yaxis_title=self.__Y_source_label__,
            )
            if type(export) == bool:
                if export:
                    fig.write_image(title + ".svg")
            elif type(export) == str:
                fig.write_image(export + ".pdf")

            if display:
                fig.show()

    def splineLinear(self, x):
        # Nome do método: "linear"
        if self.I[0] <= x <= self.I[-1]:
            position = np.searchsorted(self.__X_source__, x)
        elif x > self.I[-1]:
            position = len(self.__X_source__) - 1
        else:
            position = 1
        dx = float(self.__X_source__[position] - self.__X_source__[position - 1])
        dy = float(self.__Y_source__[position] - self.__Y_source__[position - 1])
        return self.__Y_source__[position - 1] + (dy / dx) * (x - self.__X_source__[position - 1])

    def MMQ(self, ordem=5):
        """
        Faz a interpolação dos dados pelo método do mínimos quadrados.
        Input: ordem - ordem da interpolação;
        Output: f - função interpolada.
        Nome do método: "MMQ"
        """

        # Função auxiliar para gerar os polinômios interpoladores.
        def polinomios(exp):
            return lambda x: x**exp

        # Vetor com os polinômios interpoladores
        G = [polinomios(i) for i in range(ordem + 1)]
        # Cálculo dos coeficientes para as matrizes do método dos mínimos quadrados
        A = np.zeros((len(G), len(G)))
        B = np.zeros((len(G)))
        for i in range(len(G)):
            for j in range(len(G)):
                for k in range(len(self.__X_source__)):
                    A[i][j] += G[i](self.__X_source__[k]) * G[j](self.__X_source__[k])
                    if j == 0:
                        B[i] += self.__Y_source__[k] * G[i](self.__X_source__[k])

        root = np.linalg.solve(A, B)  # Resolução do sistema linear para o cálculo dos coeficientes.

        # Criação da função interpoladora com os coeficientes calculados.
        def f(x):
            fx = 0  # Inicialização do fx.
            # Cálculo do valor da função a partir de seus coeficientes.
            for i in range(len(root)):
                fx += root[i] * x**i
            return fx

        self.f = f
        # self.__Y_source__ = self.f(self.__X_source__)
        return f

    def interPolinomial(self):
        # Retorna a função MMQ em que a ordem é igual ao número de pontos.
        # Nome do método: "interPol"
        return self.MMQ(ordem=len(self.__X_source__))

    def splineSimples(self):
        self.method = "spline"
        # Gera a spline simples dos dados de entrada.
        # Nome do método: "spline"
        n = self.__X_source__

        def P(x):
            # Calcula o polinômio interpolador na forma de Lagrange.
            p = 0
            for k in range(len(n)):

                def l(x):
                    lx = 1
                    for j in range(0, k):
                        lx *= (x - self.__X_source__[j]) / (self.__X_source__[k] - self.__X_source__[j])
                    for j in range(k + 1, len(n)):
                        lx *= (x - self.__X_source__[j]) / (self.__X_source__[k] - self.__X_source__[j])
                    return self.__Y_source__[k] * lx

                p += l(x)
            return p

        self.f = P
        self.__Y_source__ = self.f(self.__X_source__)
        return P

    def cubicSpline(self, di=None, df=None):
        self.method = "cubicSpline"
        n = len(self.__X_source__)
        M = np.zeros((n, n))
        b = np.zeros(n)

        for i in range(1, n - 1):
            hi = self.__X_source__[i] - self.__X_source__[i - 1]
            hi_1 = self.__X_source__[i + 1] - self.__X_source__[i]
            b[i] = (
                6
                / (hi + hi_1)
                * (
                    (self.__Y_source__[i + 1] - self.__Y_source__[i]) / hi_1
                    - (self.__Y_source__[i] - self.__Y_source__[i - 1]) / hi
                )
            )
            M[i][i - 1] = hi / (hi + hi_1)
            M[i][i] = 2
            M[i][i + 1] = hi_1 / (hi + hi_1)

        if di is not None and df is not None:
            h1 = self.__X_source__[1] - self.__X_source__[0]
            hn = self.__X_source__[n - 1] - self.__X_source__[n - 2]
            M[0][0] = 2
            M[0][1] = 1
            b[0] = 6 / h1 * ((self.__Y_source__[1] - self.__Y_source__[0]) / h1 - di)
            M[n - 1][n - 2] = 1
            M[n - 1][n - 1] = 2
            b[n - 1] = 6 / hn * (df - (self.__Y_source__[n - 1] - self.__Y_source__[n - 2]) / hn)
        else:
            M[0][0] = 1
            b[0] = 0
            M[n - 1][n - 1] = 1
            b[n - 1] = 0

        self.cubicSMi = np.linalg.solve(M, b)

        def S(x):
            if self.I[0] < x <= self.I[-1]:
                position = np.searchsorted(self.__X_source__, x)
            elif x > self.I[-1]:
                position = len(self.__X_source__) - 1
            else:
                position = 1
            hi = self.__X_source__[position] - self.__X_source__[position - 1]
            xi = self.__X_source__[position]
            xi1 = self.__X_source__[position - 1]
            y = (
                (xi - x) / hi * self.__Y_source__[position - 1]
                + (x - xi1) / hi * self.__Y_source__[position]
                + hi**2 / 6 * (((xi - x) / hi) ** 3 - (xi - x) / hi) * self.cubicSMi[position - 1]
                + hi**2 / 6 * (((x - xi1) / hi) ** 3 - (x - xi1) / hi) * self.cubicSMi[position]
            )
            return y

        self.f = S
        return S

    def trapezios(self, n):
        a = self.I[0]
        b = self.I[1]
        h = (b - a) / n
        integral = self.getValue(a) + self.getValue(b)
        for i in range(1, n):
            integral += 2 * self.getValue(a + i * h)
        integral *= h / 2
        return integral

    def romberg(self, n):
        T = np.zeros((n, n))
        for i in range(n):
            T[i][0] = self.trapezios(2**i)
            for j in range(1, i + 1):
                T[i][j] = T[i][j - 1] + (T[i][j - 1] - T[i - 1][j - 1]) / (4**j - 1)
        return T

    def derivative(self, x=0, order=2):
        if order == 1:
            # TODO: add the method to execute the derivative
            # return (self.getValue(x + dx) - self.getValue(x)) / dx
            return 0
        else:
            return self.derivative_second_order(x)

    def derivative_second_order(self, x=0):
        """Derivate the function at point x using partial differences with error
        of 2nd order."""

        x_index = np.searchsorted(self.__X_source__, x)

        # Treat left border
        if x_index == 0:
            h = self.__X_source__[x_index + 1] - self.__X_source__[x_index]
            df = (
                -self.__Y_source__[x_index + 2] + 4 * self.__Y_source__[x_index + 1] - 3 * self.__Y_source__[x_index]
            ) / (2 * h)
            return df

        # Treat right border
        elif x_index == len(self.__X_source__) - 1:
            h = self.__X_source__[x_index] - self.__X_source__[x_index - 1]
            df = (
                3 * self.__Y_source__[x_index] - 4 * self.__Y_source__[x_index - 1] + self.__Y_source__[x_index - 2]
            ) / (2 * h)
            return df

        # center of the array
        else:
            df = (self.__Y_source__[x_index + 1] - self.__Y_source__[x_index - 1]) / (
                self.__X_source__[x_index + 1] - self.__X_source__[x_index - 1]
            )
            return df

    def derivative_function(self, order=2, method="linear"):
        d_F = np.zeros(len(self.__X_source__))
        for i in range(len(self.__X_source__)):
            d_F[i] = self.derivative(self.__X_source__[i], order=order)
        self.derivative = Function(
            self.__X_source__,
            d_F,
            self.__X_source_label__,
            "Derivative of " + self.__Y_source_label__,
            self.name,
            method=method,
        )
        return self.derivative

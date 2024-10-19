import sympy as sp
import numpy as np
from typing import Callable, Sequence


def square_fit(
    x: np.ndarray, y: np.ndarray, func: Callable, init_parms: np.ndarray = None
) -> np.ndarray:
    # TODO 数值计算任意函数的最小二乘法拟合
    # TODO 下面的那个用的sympy，sympy实际上并不好用
    """
    最小二乘法拟合
    parms:
        x ,y :一维数组，表示数据点
        func :带参数的拟合函数
        func格式: func(x: np.ndarray, parameters: np.ndarray), 其中parameters是要拟合的参数
        init_parms: 可以指定一组初始的参数，默认都是1
    return:
        一个表示拟合参数的数组
    """
    return None


def Square_Fitting(
    In_x: np.ndarray, In_y: np.ndarray, Base: Sequence[Callable], weight: Sequence = []
) -> np.ndarray:
    if not weight:  # 权函数默认为1
        weight = [1 for i in range(min(In_x.shape[0], In_y.shape[0]))]

    symbols_set = set()
    for element in Base:
        expr = sp.sympify(str(element))
        symbols_set |= expr.free_symbols
    symbols_list = list(symbols_set)  # 找到所有的符号变量
    if len(symbols_list) != 1:  # 仅支持一个变量的拟合
        raise ValueError("Best_Square_Square: Invalid Input")
    else:
        z = symbols_list[0]
        length = min(In_x.shape[0], In_y.shape[0])
        if In_x.shape[0] != In_y.shape[0]:  # 检查x,y列表长度是否相同
            raise ValueError(
                "Best_Square_Fitting Waring: The length of the x list is not equal to the length of the y list"
            )
            In_x = In_x[:length]
            In_y = In_y[:length]
        else:
            M_Gram = np.array(
                [  # 创建Gram矩阵
                    [
                        sum(
                            weight[i]
                            * Base[m].subs(z, In_x[i]).evalf()
                            * Base[n].subs(z, In_x[i]).evalf()
                            for i in range(length)
                        )
                        for m in range(len(Base))
                    ]
                    for n in range(len(Base))
                ]
            )
            if np.linalg.det(M_Gram) == 0:  # 检查Gram矩阵是否奇异
                raise ValueError("Best_Square_Fitting: Gram matrix is singular")
            else:
                F_Vec = np.array(
                    [  # 法方程等号右边的向量
                        sum(
                            In_y[i] * Base[m].subs(z, In_x[i]).evalf() * weight[i]
                            for i in range(length)
                        )
                        for m in range(len(Base))
                    ]
                )

                Parameter_Vector = np.linalg.solve(M_Gram, F_Vec)
                # Fitting_Function = sum(
                #     Parameter_Vector[i] * Base[i] for i in range(len(Base))
                # )
                return Parameter_Vector


# TODO 下面的插值都需要优化
def products(List, level=-1):
    # TODO 用到这个函数的地方都用数组优化掉
    if not List:
        return 0
    else:
        if level < -1:
            return 0
        elif level == 0:
            return 1
        elif level == -1:
            result = 1
            for i in range(len(List)):
                result *= List[i]
            return result
        else:
            result = 1
            for i in range(level):
                result *= List[i]
            return result


# 拉格朗日差值
def Lagrange_Interval(x, y, Ex_x=[]):
    if not x or not y:
        print("Warning(Lagrange_Interval): x or y is empty")
        return 0
    else:
        length = min(len(x), len(y))

        if not len(x) == len(y):
            print(
                "Warning(Lagrange_Interval): the length of the list of x coordinates is not the same as the length of the list of y coordinates."
            )
            print(f"The program has kept the first {length} coordinates")
            x = x[:length]
            y = y[:length]
        else:
            pass

        z = sp.symbols("z")

        Lg_Ele = [
            [
                sp.Piecewise(((z - x[n]) / (x[m] - x[n]), m != n), (1, m == n))
                for n in range(length)
            ]
            for m in range(length)
        ]
        Lg_Base = [sp.simplify(sp.Mul(*Lg_Ele[m])) for m in range(length)]
        Ori_Lg_Interpolation_Pol = sum(y[i] * Lg_Base[i] for i in range(length))
        Lg_Interpolation_Pol = sp.simplify(Ori_Lg_Interpolation_Pol)

        if not Ex_x:
            Ex_y = []
        else:
            Ex_y = [
                Lg_Interpolation_Pol.subs(z, Ex_x[m]).evalf() for m in range(len(Ex_x))
            ]

        if not Ex_y:
            return Lg_Interpolation_Pol
        else:
            return Lg_Interpolation_Pol, Ex_y


# 厄米插值
def Hermite_Interval(x, y, yprime, Ex_x=[]):
    if not x or not y or not yprime:
        print("Warning(Hermite_Interval): x, y or yprime is empty")
        return 0
    else:
        length = min(len(x), len(y), len(yprime))

        if not len(x) == len(y) == len(yprime):
            print(
                "Warning(Hermite_Interval): the length of the lists of x, y, yprime are not equal."
            )
            print(f"The program has kept the first {length} coordinates")
            x = x[:length]
            y = y[:length]
            yprime = yprime[:length]
        else:
            pass

        z = sp.symbols("z")

        Lg_Ele = [
            [
                sp.Piecewise(((z - x[n]) / (x[m] - x[n]), m != n), (1, m == n))
                for n in range(length)
            ]
            for m in range(length)
        ]
        Lg_Base = [sp.simplify(sp.Mul(*Lg_Ele[m])) for m in range(length)]
        DLg_Base = [sp.diff(Lg_Base[m], z) for m in range(length)]
        Hmb_List = [-2 * DLg_Base[m].subs(z, x[m]).evalf() for m in range(length)]
        H = [
            (1 + Hmb_List[m] * (z - x[m])) * (Lg_Base[m] ** 2).evalf()
            for m in range(length)
        ]
        HQ = [(z - x[m]) * (Lg_Base[m] ** 2) for m in range(length)]

        Ori_Hm_Interpolation_Pol = sum(
            y[m] * H[m] + yprime[m] * HQ[m] for m in range(length)
        )
        Hm_Interpolation_Pol = sp.simplify(Ori_Hm_Interpolation_Pol)

        if not Ex_x:
            Ex_y = []
        else:
            Ex_y = [
                Hm_Interpolation_Pol.subs(z, Ex_x[m]).evalf() for m in range(len(Ex_x))
            ]

        if not Ex_y:
            return Hm_Interpolation_Pol
        else:
            return Hm_Interpolation_Pol, Ex_y


# 输入：坐标(x,y)的列表，数组x = [x1, x2, ……], y = [y1, y2, ……]
# 输出：差分表
def Diff(x: np.ndarray, y: np.ndarray):
    length = min(x.shape[0], y.shape[0])
    if not x.shape[0] == y.shape[0]:
        print(
            "Warning(Diff): the length of the list of x coordinates is not the same as the length of the list of y coordinates."
        )
        print(f"The program has kept the first {length} coordinates")
        x = x[:length]
        y = y[:length]
    else:
        pass

    Diff_Table = np.zeros((length, length - 1))
    Diff_Table[0] = np.array([y if (i == 0) else 0 for i in range(length - 1)])
    Diff_Table = np.array(
        [
            [
                (Diff_Table[m - 1][i + 1] - Diff_Table[m - 1][i]) / (x[i + m] - x[i])
                if (m + i < length)
                else 0
                for i in range(length - m)
            ]
            for m in range(1, length)
        ]
    )

    return Diff_Table


# 牛顿插值
def Newton_Interval(x: np.ndarray, y: np.ndarray, Ex_x=[]):
    Diff_Quot_Table = Diff(x, y)
    length = min(x.shape[0], y.shape[0])

    z = sp.symbols("z")

    Delta_X = z - x
    Poly_Terms = np.array(
        [Diff_Quot_Table[m][0] * sp.prod(Delta_X[:m]) for m in range(length)]
    )
    Ori_Nt_Interpolation_Pol = sum(Poly_Terms[m] for m in range(length))
    Nt_Interpolation_Pol = sp.simplify(Ori_Nt_Interpolation_Pol)

    if not Ex_x:
        Ex_y = []
    else:
        Ex_y = [Nt_Interpolation_Pol.subs(z, Ex_x[m]).evalf() for m in range(len(Ex_x))]

    if not Ex_y:
        return Nt_Interpolation_Pol
    else:
        return Nt_Interpolation_Pol, Ex_y


# 分段插值，每一段是三次的厄米插值
#!代码没写完，还要写可选不是三次厄米插值
def Piecewise_Her_Interpolation(x, y, yprime, Ex_x=[]):
    if not x or not y or not yprime:
        print("Warning(Hermite_Interval): x, y or yprime is empty")
        return 0
    else:
        length = min(len(x), len(y), len(yprime))

        if not len(x) == len(y) == len(yprime):
            print(
                "Warning(Hermite_Interval): the length of the lists of x, y, yprime are not equal."
            )
            print(f"The program has kept the first {length} coordinates")
            x = x[:length]
            y = y[:length]
            yprime = yprime[:length]
        else:
            pass

    z = sp.symbols("z")

    Piecewise_Functions = [
        Hermite_Interval([x[i], x[i + 1]], [y[i], y[i + 1]], [yprime[i], yprime[i + 1]])
        for i in range(length - 1)
    ]

    Interval_Polynomial = sp.Piecewise(
        *[
            (Piecewise_Functions[i], (z > x[i]) & (z <= x[i + 1]))
            for i in range(length - 1)
        ]
    )

    if not Ex_x:
        Ex_y = []
    else:
        Ex_y = [Interval_Polynomial.subs(z, Ex_x[m]).evalf() for m in range(len(Ex_x))]

    if not Ex_y:
        return Interval_Polynomial
    else:
        return Interval_Polynomial, Ex_y

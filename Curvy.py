from CurveEditor import CurveEditorWindow
import numpy as np
from Cholesky import solve


# Utils functions
def hermite_to_bezier_points(points, m):
    """Return the corresponding Bezier points from points and derivatives """
    spline_points = []
    for k in range(len(points) - 1):
        spline_points.extend([
            points[k], points[k] + (1 / 3) * m[k],
            points[k + 1] - (1 / 3) * m[k + 1]
        ])
    spline_points.append(points[-1])

    return spline_points


# Osculator circle
def conditional_swap(e1, e2, condition):
    """Return the tuple (e1, e2) swaped iff condition is True"""
    if condition:
        return e2, e1
    else:
        return e1, e2


def get_horizontal_intersections_angle(radius, y, orientation):
    """Return the angles of the two points on the horizontal line at y (with respect to the center) which intersect the circle if they exist"""
    if abs(y) < radius:
        a = np.arcsin(abs(y) / radius)
        if y >= 0:
            return conditional_swap(a, np.pi - a, orientation == "up")
        else:
            return conditional_swap(2 * np.pi - a, np.pi + a,
                                    orientation == "up")


def get_vertical_intersections_angle(radius, x, orientation):
    """Return the angles of the two points on the vertical line at x (with respect to the center) which intersect the circle if they exist"""
    if abs(x) < radius:
        a = np.arccos(abs(x) / radius)
        if x >= 0:
            return conditional_swap(-a, a, orientation == "left")
        else:
            return conditional_swap(np.pi + a, np.pi - a,
                                    orientation == "left")


def clipped_circle(bounding_rectangle, cx, cy, radius, resolution):
    """Return the circle curve clipped in bounding_rectengle which must have the form (x, y, width, height)"""
    x, y, width, height = tuple(bounding_rectangle)
    intervals = []

    th_angles = get_horizontal_intersections_angle(radius, y - cy, "up")
    rv_angles = get_vertical_intersections_angle(radius, x + width - cx,
                                                 "right")
    bh_angles = get_horizontal_intersections_angle(radius, y + height - cy,
                                                   "down")
    lv_angles = get_vertical_intersections_angle(radius, x - cx, "left")

    if (th_angles):
        if (x <= cx + radius * np.cos(th_angles[0]) <= x + width):
            intervals.append(th_angles[0])
        if (x <= cx + radius * np.cos(th_angles[1]) <= x + width):
            intervals.append(th_angles[1])

    if (rv_angles):
        if (y <= cy + radius * np.sin(rv_angles[0]) <= y + height):
            intervals.append(rv_angles[0])
        if (y <= cy + radius * np.sin(rv_angles[1]) <= y + height):
            intervals.append(rv_angles[1])

    if (bh_angles):
        if (x <= cx + radius * np.cos(bh_angles[0]) <= x + width):
            intervals.append(bh_angles[0])
        if (x <= cx + radius * np.cos(bh_angles[1]) <= x + width):
            intervals.append(bh_angles[1])

    if (lv_angles):
        if (y <= cy + radius * np.sin(lv_angles[0]) <= y + height):
            intervals.append(lv_angles[0])
        if (y <= cy + radius * np.sin(lv_angles[1]) <= y + height):
            intervals.append(lv_angles[1])

    # Circular permutation to have intervals to draw
    circle = []
    if (len(intervals) > 0):
        intervals = intervals[1:] + [intervals[0]]

        for i in range(0, len(intervals) - 1, 2):
            a = intervals[i]
            b = intervals[i + 1]
            circle.extend(
                arc(cx, cy, radius, a, b,
                    int(2 * resolution / len(intervals))))
    else:
        circle.extend(arc(cx, cy, radius, 0, 2 * np.pi, resolution))

    return circle


def osculator_circle(curve, first_derivatives, second_derivatives, circle_t,
                     resolution, canvas_size):

    # Compute osculator circle curve
    i = int(circle_t * (resolution - 3))
    c = curvature(first_derivatives[i], second_derivatives[i])
    normal = np.array([-first_derivatives[i][1], first_derivatives[i][0]
                       ]) / np.linalg.norm(first_derivatives[i])
    center = curve[i] + (1 / c) * normal

    return clipped_circle((0, 0, *canvas_size), center[0], center[1], 1 / c,
                          resolution)


# Curvature
def forward_difference(elements, i):
    return elements[i + 1] - elements[i]


def forward_differences(elements, i, r):
    if r == 1:
        return forward_difference(elements, i)

    return forward_differences(elements, i + 1, r - 1) - forward_differences(
        elements, i, r - 1)


def curvature(first_derivative, second_derivative):
    det = np.linalg.det(np.array([first_derivative, second_derivative]))
    norm = np.linalg.norm(first_derivative)**3
    return det / norm


def plot_curvature(first_derivatives, second_derivatives, plot, **kwargs):
    curvatures = np.array(
        list(
            map(lambda x: curvature(x[0], x[1]),
                zip(first_derivatives, second_derivatives))))
    plot.plot(curvatures, **kwargs)


def cubic_spline_derivatives(points, resolution):
    """Return evaluation of cubic spline according to resolution"""
    U = np.linspace(
        0, 1, resolution
    )[:-1]  # prevent double counting of right extreme point of each segment
    curve = []
    first_derivatives = []
    second_derivatives = []
    for k in range(0, len(points) - 3, 3):
        iterations = decasteljau_iterations(
            np.array([points[k], points[k + 1], points[k + 2], points[k + 3]]),
            U)
        for iteration in iterations:
            curve.append(iteration[-1][0])
            first_derivatives.append(3 * forward_difference(iteration[-2], 0))
            second_derivatives.append(6 *
                                      forward_differences(iteration[-3], 0, 2))

    return curve + [points[-1]], first_derivatives, second_derivatives


def decasteljau_iterations(points, T, canvas_size=None):
    """Return all the iterations of DeCasteljau algorithm over the evaluations of the curve"""
    n = points.shape[0] - 1
    result = []
    for t in T:
        r = [points.copy()]
        for k in range(0, n):
            r.append([])
            for i in range(0, n - k):
                r[k + 1].append(
                    np.array([(1 - t) * r[k][i][0] + t * r[k][i + 1][0],
                              (1 - t) * r[k][i][1] + t * r[k][i + 1][1]]))
        result.append(r)

    return result


# Algorithms
def DeCasteljau(points, T, canvas_size=None):
    n = points.shape[0] - 1
    result = []
    for t in T:
        r = points.copy()
        for k in range(0, n):
            for i in range(0, n - k):
                r[i, :] = (1 - t) * r[i, :] + t * r[i + 1, :]

        result.append(r[0, :])

    return (result, )


def SplineC0(points, circle_t, show_circle, plot, T, canvas_size):
    # Scale the derivatives according to the cubic degree
    for i in range(len(points)):
        if i % 3 == 1:
            points[i] = (points[i] + 2 * points[i - 1]) / 3
        elif i % 3 == 2:
            (points[i] + 2 * points[i + 1]) / 3

    # Compute curve
    curve, first_derivatives, second_derivatives = cubic_spline_derivatives(
        points, int(3 * len(T) / (len(points) - 1)))

    plot_curvature(first_derivatives, second_derivatives, plot, c='red')
    if show_circle:
        return (curve,
                osculator_circle(curve, first_derivatives, second_derivatives,
                                 circle_t, len(T), canvas_size))
    else:
        return (curve, )


def SplineC1(points, c, circle_t, show_circle, plot, T, canvas_size):
    # Compute the derivatives
    m = [points[1] - points[0]
         ] + [(1 - c) * (points[i + 1] - points[i - 1]) / 2
              for i in range(1, len(points) - 1)] \
        + [points[-1] - points[-2]]

    # Compute curve
    bezier_points = hermite_to_bezier_points(points, m)
    curve, first_derivatives, second_derivatives = cubic_spline_derivatives(
        bezier_points, int(3 * len(T) / (len(bezier_points) - 1)))

    plot_curvature(first_derivatives, second_derivatives, plot, c='black')
    if show_circle:
        return (curve,
                osculator_circle(curve, first_derivatives, second_derivatives,
                                 circle_t, len(T), canvas_size))
    else:
        return (curve, )


def SplineC2(points, circle_t, show_circle, plot, T, canvas_size):
    # Use Cholesky decomposition to solve the linear system and compute dervatives
    N = len(points) - 2
    linf = [1 for _ in range(N + 1)]
    ldiag = [2] + [4 for _ in range(N)] + [2]
    b = [3 * (points[1] - points[0])
         ] + [3 * (points[i + 1] - points[i - 1])
              for i in range(1, N + 1)] + [3 * (points[-1] - points[-2])]

    m = solve(linf, ldiag, b)

    # Compute curve
    bezier_points = hermite_to_bezier_points(points, m)
    curve, first_derivatives, second_derivatives = cubic_spline_derivatives(
        bezier_points, int(3 * len(T) / (len(bezier_points) - 1)))

    plot_curvature(first_derivatives, second_derivatives, plot, c='green')
    if show_circle:
        return (curve,
                osculator_circle(curve, first_derivatives, second_derivatives,
                                 circle_t, len(T), canvas_size))
    else:
        return (curve, )


def Lagrange(points, T, canvas_size):
    """Performs Aitken-Neville algorithm"""
    n = points.shape[0] - 1
    U = T * n

    result = []
    for u in U:
        r = points.copy()
        for k in range(1, n + 1):
            for i in range(0, n - k + 1):
                r[i, :] = (i + k - u) / k * r[i, :] + (u - i) / k * r[i + 1, :]

        result.append(r[0, :])

    return (result, )


# Bonus utils functions
def arc(cx, cy, radius, begin_angle, end_angle, resolution):
    """Return the evaluation of the circle arc """
    while begin_angle > end_angle:
        begin_angle -= 2 * np.pi
    result = []
    for t in np.linspace(begin_angle, end_angle, resolution):
        result.append(
            np.array([cx + radius * np.cos(t), cy + radius * np.sin(t)]))

    return result


def de_boor(points, knots, p, T, segmentated):
    assert len(knots) - len(points) - 1
    result = [[]]

    U = (1 - T) * knots[p] + T * knots[len(points)]  # B-Spline domain

    # knots interval index initialization
    k = p
    for u in U:
        # Update the index of knots interval
        while k < len(points) - 1 and u >= knots[k + 1]:
            if segmentated:
                result.append([])
            k += 1

        #DeBoor algorithm
        d = [points[j + k - p] for j in range(0, p + 1)]

        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                alpha = (u - knots[j + k - p]) / (knots[j + 1 + k - r] -
                                                  knots[j + k - p])
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

        result[-1].append(d[p])
    return [result[i] + [result[i + 1][0]]
            for i in range(len(result) - 1)] + [result[-1]]


def uniform_fill(editor_window, **kwargs):
    points = kwargs['points']
    p = kwargs['degree']
    closed = kwargs['closed']

    if closed:
        values = [k for k in range(0, len(points) - 3)]
    else:
        values = [0 for _ in range(p + 1)] + [
            k for k in range(1, editor_window.n_points - p)
        ] + [editor_window.n_points - p for _ in range(p + 1)]

    editor_window.entry_knots.set(values)


# Bonus algorithms
def RationalDeCasteljau(points, weights, T, canvas_size):
    homogeneous_points = points * weights[:, np.newaxis]
    homogeneous_points = np.concatenate(
        (homogeneous_points, weights.reshape((len(weights), 1))), axis=1)

    homogeneous_result = DeCasteljau(homogeneous_points, T, canvas_size)[0]

    result = [p[:-1] / p[-1] for p in homogeneous_result]

    return (result, )


def BSpline(points, knots, p, closed, show_segments, T, canvas_size):
    if not closed:
        assert len(knots) == len(points) + p + 1
        return (*de_boor(points, knots, p, T, show_segments), )

    assert len(knots) == len(points) - 3
    h = 1
    real_knots = [
        knots[0] - (p + 2) * h + i * h for i in range(p + 2)
    ] + list(knots) + [knots[-1] + i * h for i in range(1, p + 3)]
    real_points = list(points) + [points[i] for i in range(p)]

    return (*de_boor(real_points, real_knots, p, T, show_segments), )


def NURBS(points, weights, knots, p, closed, show_segments, T, canvas_size):
    homogeneous_points = points * weights[:, np.newaxis]
    homogeneous_points = np.concatenate(
        (homogeneous_points, weights.reshape((len(weights), 1))), axis=1)

    homogeneous_result = BSpline(homogeneous_points, knots, p, closed,
                                 show_segments, T, canvas_size)

    result = [[p[:-1] / p[-1] for p in segment]
              for segment in homogeneous_result]
    return (*result, )


if __name__ == "__main__":
    window = CurveEditorWindow([{
        "name": "Bezier",
        "algo": DeCasteljau,
        "pointed": True,
        "colors": ("green", )
    }, {
        "name": "R-Bezier",
        "algo": RationalDeCasteljau,
        "pointed": True,
        "weighted": True,
        "colors": ("lime", )
    }, {
        "name": "B-Spline",
        "algo": BSpline,
        "pointed": True,
        "knotted": True,
        "colors": ("red", "blue"),
        "parameters": {
            "closed": {
                "type": "check",
            },
            "show segments": {
                "type": "check",
            }
        },
        "buttons": {
            "uniform": {
                "type": "button",
                "command": uniform_fill,
            }
        }
    }, {
        "name": "NURBS",
        "algo": NURBS,
        "pointed": True,
        "weighted": True,
        "knotted": True,
        "colors": ("purple", "red"),
        "parameters": {
            "closed": {
                "type": "check",
            },
            "show segments": {
                "type": "check",
            },
        },
        "buttons": {
            "uniform": {
                "type": "button",
                "command": uniform_fill,
            }
        }
    }, {
        "name": "Lagrange",
        "algo": Lagrange,
        "pointed": True,
        "colors": ("green", )
    }, {
        "name": "Spline-C0",
        "algo": SplineC0,
        "pointed": True,
        "colors": ("red", "blue"),
        "parameters": {
            "t": {
                "type": "slider",
                "from": 0.0,
                "to": 1.0,
                "resolution": 0.01,
                "default": 0.0
            },
            "show osculator circle": {
                "type": "check",
            }
        },
        "plotted": True
    }, {
        "name": "Spline-C1",
        "algo": SplineC1,
        "pointed": True,
        "parameters": {
            "c": {
                "type": "slider",
                "from": 0.0,
                "to": 1.0,
                "resolution": 0.05,
                "default": 0.5
            },
            "t": {
                "type": "slider",
                "from": 0.0,
                "to": 1.0,
                "resolution": 0.01,
                "default": 0.0
            },
            "show osculator circle": {
                "type": "check",
            }
        },
        "colors": ("black", "blue"),
        "plotted": True
    }, {
        "name": "Spline-C2",
        "algo": SplineC2,
        "pointed": True,
        "parameters": {
            "t": {
                "type": "slider",
                "from": 0.0,
                "to": 1.0,
                "resolution": 0.01,
                "default": 0.0
            },
            "show osculator circle": {
                "type": "check",
            }
        },
        "colors": ("green", "blue"),
        "plotted": True
    }])
    window.mainloop()

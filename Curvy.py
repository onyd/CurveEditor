"""
author: Anthony Dard
description: this script is a tkinter GUI curve editor which allows user to:
-add/remove point on  the canvas
-drag the points on the canvas
-select and edit coordinates of a point on th canvas (validate with enter)
-choose the algorithm used to compute the curve
-change the resolution of final linear interpolation of the curve
-reset button clear all points

To add algorithm, see the __init__ doc of CurveEditorWindow
"""

import time
from tkinter import *
import numpy as np
from itertools import chain
from MultiIntEntry import MultIntiEntry
import matplotlib.pyplot as plt
from Cholesky import solve


class CurveEditorWindow(Tk):
    def __init__(self, compute_algorithms) -> None:
        """compute_algorithms is a list of dict {"name" : string, "algo" : function, "weighted" : bool, "knotted": bool) which 
        -string is the text in button selection of algorithm
        -function has the signature: def <name>(points, [weights], [knots], T, canvas_size) -> list of points
            where points is the list of points drawn on the canvas and T the array of t in [0; 1] with resolution given by the slider
        -weighted precise if weights are needeed and passed in arguments
        -knotted precise if knots are needeed and passed in arguments"""
        super().__init__()

        self.title("Curvy editor")
        self.geometry("720x520")

        self._selected_data = {"x": 0, "y": 0, 'item': None}
        self.radius = 5
        self.compute_algorithms = compute_algorithms
        self.n_points = 0

        self.rowconfigure([0, 1], weight=1, minsize=200)
        self.columnconfigure([0, 1], weight=1, minsize=200)

        self.columnconfigure(2, weight=0, minsize=200)

        self.setup_canvas()
        self.setup_panel()

    # Setup
    def setup_canvas(self):
        self.graph = Canvas(self, bd=2, cursor="plus", bg="#fbfbfb")
        self.graph.grid(column=0,
                        padx=2,
                        pady=2,
                        rowspan=2,
                        columnspan=2,
                        sticky="nsew")

        self.graph.bind('<Button-1>', self.handle_canvas_click)
        self.graph.tag_bind("control_points", "<ButtonRelease-1>",
                            self.handle_drag_stop)
        self.graph.bind("<B1-Motion>", self.handle_drag)

    def setup_panel(self):
        # Right panel for options
        self.frame_pannel = Frame(self, relief=RAISED, bg="#e1e1e1")
        self.frame_curve_type = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_edit_mode = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_edit_position = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_resolution_sliders = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_weights_sliders = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_knots_entry = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_parameter = Frame(self.frame_pannel, bg="#e1e1e1")

        # Selection of curve type
        self.label_curve_type = Label(self.frame_curve_type,
                                      text="Curve algorithms:")
        self.label_curve_type.grid(row=0, column=1)
        curve_types = [algo['name'] for algo in self.compute_algorithms]

        self.check_curve_buttons = [None] * len(self.compute_algorithms)
        self.curve_types = [
            IntVar() for _ in range(len(self.compute_algorithms))
        ]
        for i in range(len(self.compute_algorithms)):
            self.check_curve_buttons[i] = Checkbutton(
                self.frame_curve_type,
                text=curve_types[i],
                onvalue=1,
                offvalue=0,
                variable=self.curve_types[i],
                fg=self.compute_algorithms[i]['color'])
            self.check_curve_buttons[i].grid(row=i // 3 + 1, column=i % 3)
            self.check_curve_buttons[i].bind(
                "<ButtonRelease-1>", lambda event: self.graph.after(
                    100, lambda: self.handle_algo_selection()))

        # Selection of edit mode
        self.label_edit_mode = Label(self.frame_edit_mode, text="Edit mode")
        self.label_edit_mode.pack()
        edit_mode = ['Add', 'Remove', 'Drag', 'Select']
        edit_mode_val = ["add", "remove", "drag", "select"]
        self.edit_mode = StringVar()
        self.edit_mode.set(edit_mode_val[0])

        self.radio_edit_buttons = [None] * 4
        for i in range(4):
            self.radio_edit_buttons[i] = Radiobutton(self.frame_edit_mode,
                                                     variable=self.edit_mode,
                                                     text=edit_mode[i],
                                                     value=edit_mode_val[i],
                                                     bg="#e1e1e1")
            self.radio_edit_buttons[i].pack(side='left', expand=1)
            self.radio_edit_buttons[i].bind(
                "<ButtonRelease-1>", lambda event: self.reset_selection())

        # Edit position of selected point widget
        self.label_pos_x = Label(self.frame_edit_position, text='x: ')
        self.label_pos_y = Label(self.frame_edit_position, text='y: ')
        self.pos_x = StringVar()
        self.pos_y = StringVar()
        self.entry_position_x = Entry(self.frame_edit_position,
                                      textvariable=self.pos_x)
        self.entry_position_y = Entry(self.frame_edit_position,
                                      textvariable=self.pos_y)
        self.label_pos_x.pack(side=LEFT)
        self.entry_position_x.pack(side=LEFT)
        self.label_pos_y.pack(side=LEFT)
        self.entry_position_y.pack(side=LEFT)

        self.entry_position_x.bind("<FocusOut>", self.update_pos)
        self.entry_position_x.bind("<KeyPress-Return>", self.update_pos)
        self.entry_position_x.bind("<KeyPress-KP_Enter>", self.update_pos)

        self.entry_position_y.bind("<FocusOut>", self.update_pos)
        self.entry_position_y.bind("<KeyPress-Return>", self.update_pos)
        self.entry_position_x.bind("<KeyPress-KP_Enter>", self.update_pos)

        # Slider for parameter update
        self.label_resolution = Label(self.frame_resolution_sliders,
                                      text="Resolution: ")
        self.slider_resolution = Scale(self.frame_resolution_sliders,
                                       from_=5,
                                       to=200,
                                       orient=HORIZONTAL,
                                       bg="#e1e1e1")
        self.slider_resolution.set(50)
        self.label_resolution.pack(side=LEFT)
        self.slider_resolution.pack(fill="x")
        self.slider_resolution.bind("<ButtonRelease-1>",
                                    lambda event: self.draw_curve())

        # Weights
        self.label_weights = Label(self.frame_weights_sliders,
                                   text="Weights: ")
        self.slider_weights = Scale(self.frame_weights_sliders,
                                    from_=1.0,
                                    to=10.0,
                                    resolution=0.1,
                                    orient=HORIZONTAL,
                                    bg="#e1e1e1")
        self.slider_weights.set(1.0)
        self.label_weights.pack(side=LEFT)
        self.slider_weights.pack(fill="x")
        self.slider_weights.bind("<ButtonRelease-1>", self.update_weight)

        # Knots
        self.label_knots = Label(self.frame_knots_entry, text="knots: ")
        self.label_knots.grid(row=0, column=0)
        self.entry_knots = MultIntiEntry(self.frame_knots_entry, 1, [0],
                                         self.draw_curve)
        self.entry_knots.grid(row=0, column=1)

        self.label_degree = Label(self.frame_knots_entry, text="degree: ")
        self.label_degree.grid(row=1, column=0)
        self.entry_degree = Entry(self.frame_knots_entry)
        self.entry_degree.insert(0, "3")
        self.entry_degree.grid(row=1, column=1)
        self.entry_degree.bind("<KeyPress-Return>",
                               lambda event: self.draw_curve())
        self.entry_degree.bind("<KeyPress-KP_Enter>",
                               lambda event: self.draw_curve())

        self.label_knots_fill = Label(self.frame_knots_entry,
                                      text="knots fill: ")
        self.label_knots_fill.grid(row=2, column=0)
        self.button_uniform_fill = Button(self.frame_knots_entry,
                                          text="uniform")
        self.button_uniform_fill.grid(row=2, column=1)
        self.button_uniform_fill.bind("<ButtonRelease-1>", self.uniform_fill)

        # Parameter
        self.label_parameter = Label(self.frame_parameter, text="parameter: ")
        self.slider_parameter = Scale(self.frame_parameter,
                                      from_=0.0,
                                      to=1.0,
                                      resolution=0.05,
                                      orient=HORIZONTAL,
                                      bg="#e1e1e1")
        self.slider_parameter.set(0.5)
        self.label_parameter.pack(side=LEFT)
        self.slider_parameter.pack(fill="x")
        self.slider_parameter.bind("<ButtonRelease-1>",
                                   lambda event: self.draw_curve())

        # Frame pack
        self.frame_pannel.grid(row=0,
                               column=2,
                               padx=2,
                               pady=2,
                               rowspan=2,
                               sticky="nswe")
        self.frame_curve_type.pack(fill="x")
        self.frame_edit_mode.pack(fill="x")
        self.frame_edit_position.pack(fill="x")
        self.frame_resolution_sliders.pack(fill="x")

        self.button_reset = Button(self.frame_pannel, text="Reset")
        self.button_reset.pack(side=BOTTOM, fill="x")
        self.button_reset.bind("<ButtonRelease-1>",
                               lambda event: self.graph.delete("all"))

    # Drawing
    def get_points(self):
        points = []
        for item in self.graph.find_withtag("control_points"):
            coords = self.graph.coords(item)
            points.append([
                float(coords[0] + self.radius),
                float(coords[1] + self.radius)
            ])  # Ensure curve accuracy
        return points

    def get_weights(self):
        weights = []
        for item in self.graph.find_withtag("control_points"):
            weights.append(float(
                self.graph.gettags(item)[1][7:]))  # Get weight from tags
        return weights

    def create_point(self, x, y, color):
        """Create a token at the given coordinate in the given color"""
        item = self.graph.create_oval(x - self.radius,
                                      y - self.radius,
                                      x + self.radius,
                                      y + self.radius,
                                      outline=color,
                                      fill=color,
                                      tags=("control_points", f"weight_{1.0}"))
        self.n_points += 1
        return item

    def draw_polygon(self):
        self.graph.delete("control_polygon")
        points = self.get_points()
        for i in range(0, len(points) - 1):
            self.graph.create_line(points[i][0],
                                   points[i][1],
                                   points[i + 1][0],
                                   points[i + 1][1],
                                   fill="blue",
                                   tags="control_polygon")

    def animate(self, algo, steps, interval):
        curves = []
        for step in steps:
            self.graph.delete("animated")
            T = np.linspace(0, 1, self.slider_resolution.get())
            canvas_size = (self.graph.winfo_width(), self.graph.winfo_height())
            curves.append(
                np.array(algo['algo'](*self.get_args(
                    {
                        "points": self.get_points,
                        "weights": self.get_weights,
                        "knots": self.entry_knots.get,
                        "degree": self.entry_degree.get
                    },
                    algo=algo), T, canvas_size, step)))

            for i in range(0, curves[-1].shape[0] - 1):
                self.graph.create_line(curves[-1][i, 0],
                                       curves[-1][i, 1],
                                       curves[-1][i + 1, 0],
                                       curves[-1][i + 1, 1],
                                       fill=algo['color'],
                                       width=3,
                                       tags=("curve", "animated"))
                self.graph.update()

            time.sleep(interval / 1000)
        algo['animated']['callback'](curves, T, canvas_size, steps)

    def draw_curve(self):
        self.graph.delete("curve")

        # Select algorithms with right arguments
        animated_algorithms = []
        algorithms = []
        for i, curve_type in enumerate(self.curve_types):
            if curve_type.get() != 0:
                if "animated" in self.compute_algorithms[i]:
                    animated_algorithms.append(self.compute_algorithms[i])
                else:
                    algorithms.append(self.compute_algorithms[i])

        curves = []
        colors = []
        for algo in algorithms:
            T = np.linspace(0, 1, self.slider_resolution.get())
            canvas_size = (self.graph.winfo_width(), self.graph.winfo_height())
            curves.append(
                np.array(algo['algo'](*self.get_args(
                    {
                        "points": self.get_points,
                        "weights": self.get_weights,
                        "knots": self.entry_knots.get,
                        "degree": self.entry_degree.get,
                        "parameter": self.slider_parameter.get
                    },
                    algo=algo), T, canvas_size)))
            colors.append(algo['color'])

        for k, curve in enumerate(curves):
            for i in range(0, curve.shape[0] - 1):
                self.graph.create_line(curve[i, 0],
                                       curve[i, 1],
                                       curve[i + 1, 0],
                                       curve[i + 1, 1],
                                       fill=colors[k],
                                       width=3,
                                       tags="curve")

        # Animated algo treatment
        for animated_algo in animated_algorithms:
            self.animate(animated_algo, animated_algo['animated']['steps'],
                         animated_algo['animated']["interval"])

    # Event handling
    def find_closest_with_tag(self, x, y, radius, tag):
        distances = []
        for item in self.graph.find_withtag(tag):
            c = self.graph.coords(item)
            d = (x - c[0])**2 + (y - c[1])**2
            if d <= radius**2:
                distances.append((item, c, d))

        return min(distances,
                   default=(None, [0, 0], float("inf")),
                   key=lambda p: p[2])

    def reset_selection(self):
        if self._selected_data['item'] is not None:
            self.graph.itemconfig(self._selected_data['item'], fill='red')

        self._selected_data['item'] = None
        self._selected_data["x"] = 0
        self._selected_data["y"] = 0

    def handle_canvas_click(self, event):
        self.reset_selection()

        if self.edit_mode.get() == "add":
            item = self.create_point(event.x, event.y, "red")
            self.update_pos_entry(item)
            points = self.get_points()

            if len(points) > 1:
                self.graph.create_line(points[-2][0],
                                       points[-2][1],
                                       points[-1][0],
                                       points[-1][1],
                                       fill="blue",
                                       tag="control_polygon")

                self.draw_curve()

        elif self.edit_mode.get() == "remove":
            self._selected_data[
                'item'], coords, _ = self.find_closest_with_tag(
                    event.x, event.y, 3 * self.radius, "control_points")
            if self._selected_data['item'] is not None:
                self.graph.delete(self._selected_data['item'])
                self.n_points -= 1
                self.draw_polygon()
                self.draw_curve()

        elif self.edit_mode.get() == "drag":
            self._selected_data[
                'item'], coords, _ = self.find_closest_with_tag(
                    event.x, event.y, 3 * self.radius, "control_points")

            if self._selected_data['item'] is not None:
                self._selected_data["x"] = event.x
                self._selected_data["y"] = event.y
                self.graph.move(self._selected_data['item'],
                                event.x - coords[0] - self.radius,
                                event.y - coords[1] - self.radius)

        else:
            self._selected_data[
                'item'], coords, _ = self.find_closest_with_tag(
                    event.x, event.y, 3 * self.radius, "control_points")
            if self._selected_data['item'] is not None:
                self.graph.itemconfig(self._selected_data['item'],
                                      fill='orange')  # Mark as selected
                self.update_pos_entry(self._selected_data['item'])

                self.slider_weights.set(
                    self.graph.gettags(self._selected_data['item'])[1]
                    [7:])  # Move the slider to selected point weight

    def handle_drag_stop(self, event):
        """End drag of an object"""
        if self.edit_mode.get() != "drag":
            return
        self.reset_selection()

    def handle_drag(self, event):
        """Handle dragging of an object"""
        if self.edit_mode.get() != "drag" or self._selected_data[
                'item'] is None or "control_points" not in self.graph.gettags(
                    self._selected_data['item']):
            return

        # compute how much the mouse has moved
        delta_x = event.x - self._selected_data["x"]
        delta_y = event.y - self._selected_data["y"]
        # move the object the appropriate amount
        self.graph.move(self._selected_data['item'], delta_x, delta_y)
        # record the new position
        self._selected_data["x"] = event.x
        self._selected_data["y"] = event.y

        self.update_pos_entry(self._selected_data['item'])
        self.draw_polygon()
        self.draw_curve()

    def update_weight(self, event):
        if self.edit_mode.get(
        ) != "select" or self._selected_data['item'] is None:
            return

        self.graph.itemconfig(self._selected_data['item'],
                              tags=("control_points",
                                    f"weight_{self.slider_weights.get()}"))
        self.draw_curve()

    def update_pos_entry(self, item):
        coords = self.graph.coords(item)
        self.entry_position_x.delete(0, END)
        self.entry_position_x.insert(0, int(coords[0]))
        self.entry_position_y.delete(0, END)
        self.entry_position_y.insert(0, int(coords[1]))

    def update_pos(self, event):
        if self.edit_mode.get(
        ) != "select" or self._selected_data['item'] is None:
            return

        coords = self.graph.coords(self._selected_data['item'])
        self.graph.move(self._selected_data['item'],
                        int(self.pos_x.get()) - coords[0],
                        int(self.pos_y.get()) - coords[1])

        self.draw_polygon()
        self.draw_curve()

    def is_pointed(self, algo):
        return 'pointed' in algo and algo['pointed']

    def is_weighted(self, algo):
        return 'weighted' in algo and algo['weighted']

    def is_knotted(self, algo):
        return 'knotted' in algo and algo['knotted']

    def is_parametered(self, algo):
        return "parametered" in algo and algo['parametered']

    def get_args(self, kwargs, algo):
        args = []
        if self.is_pointed(algo):
            args.append(np.array(kwargs['points']()))
        if self.is_weighted(algo):
            args.append(np.array(kwargs['weights']()))
        if self.is_knotted(algo):
            args.append(np.array(kwargs['knots']()))
            args.append(int(kwargs['degree']()))
        if self.is_parametered(algo):
            args.append(float(kwargs['parameter']()))

        return args

    def handle_algo_selection(self):
        self.frame_weights_sliders.pack_forget()
        self.frame_knots_entry.pack_forget()
        self.frame_parameter.pack_forget()

        # Show only necessary edition tools
        weighted = False
        knotted = False
        parametered = None
        for i, curve_type in enumerate(self.curve_types):
            if curve_type.get() != 0:
                weighted = weighted or self.is_weighted(
                    self.compute_algorithms[i])
                knotted = knotted or self.is_knotted(
                    self.compute_algorithms[i])
                parametered = parametered or self.is_parametered(
                    self.compute_algorithms[i])

        if weighted:
            self.frame_weights_sliders.pack(fill="x")
        if knotted:
            self.frame_knots_entry.pack(fill="x")
        if parametered is not None:
            self.frame_parameter.pack(fill="x")

        self.draw_curve()

    def uniform_fill(self, event):
        p = int(self.entry_degree.get())
        values = [0.0 for _ in range(p + 1)]
        for k in range(1, self.n_points - p):
            values.append(k / (self.n_points - p))
        values.extend([1.0 for _ in range(p + 1)])

        self.entry_knots.set(values)


# ---------- Here add algorithms ----------


def DeCasteljau(points, T, canvas_size=None):
    n = points.shape[0] - 1
    result = []
    for t in T:
        r = points.copy()
        for k in range(0, n):
            for i in range(0, n - k):
                r[i, :] = (1 - t) * r[i, :] + t * r[i + 1, :]

        result.append(r[0, :])

    return result


def RationalDeCasteljau(points, weights, T, canvas_size):
    homogeneous_points = points * weights[:, np.newaxis]
    homogeneous_points = np.concatenate(
        (homogeneous_points, weights.reshape((len(weights), 1))), axis=1)

    homogeneous_result = DeCasteljau(homogeneous_points, T, canvas_size)

    result = [p[:-1] / p[-1] for p in homogeneous_result]

    return result


def circle(points, T, canvas_size):
    result = []
    r = min(*canvas_size) / 8
    for t in T:
        cx = canvas_size[0] / 2
        cy = canvas_size[1] / 2
        result.append(
            [cx + r * np.cos(2 * t * np.pi), cy + r * np.sin(2 * t * np.pi)])

    return result


def quarterCircle(T, canvas_size):
    result = []
    r = min(*canvas_size) / 3
    for t in T:
        cx = canvas_size[0] / 2
        cy = canvas_size[1] / 2
        result.append(
            [cx + r * np.cos(t * np.pi / 2), cy + r * np.sin(t * np.pi / 2)])

    return result


def CircleApprox4(T, canvas_size, step):
    alpha = step
    r = min(*canvas_size) / 3
    cx = canvas_size[0] / 2
    cy = canvas_size[1] / 2

    points = np.array([[1, 0], [1, alpha], [alpha, 1], [0, 1]]) * r + np.array(
        [cx, cy])
    result = DeCasteljau(points, T, canvas_size)

    return result


def DeBoor(points, knots, p, T, canvas_size):
    assert len(knots) - len(points) - 1
    result = []

    # knots interval index initialization
    k = p
    T_it = iter(T)
    if knots[k] != 0.0:
        last = next(T_it)
        while last < knots[k]:
            last = next(T_it)
        T_it = chain((last, ), T_it)

    for t in T_it:
        # Update the index of knots interval
        while k + 1 < len(knots) and t >= knots[k + 1] and t < 1.0:
            k += 1

        # Out of domain
        if k >= len(points):
            break

        #DeBoor algorithm
        d = [points[j + k - p] for j in range(0, p + 1)]

        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                alpha = (t - knots[j + k - p]) / (knots[j + 1 + k - r] -
                                                  knots[j + k - p])
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

        result.append(d[p])
    return result


def RationalDeBoor(points, weights, knots, p, T, canvas_size):
    homogeneous_points = points * weights[:, np.newaxis]
    homogeneous_points = np.concatenate(
        (homogeneous_points, weights.reshape((len(weights), 1))), axis=1)

    homogeneous_result = DeBoor(homogeneous_points, knots, p, T, canvas_size)

    result = [p[:-1] / p[-1] for p in homogeneous_result]
    return result


def CubicSpline(points, resolution):
    U = np.linspace(0, 1, resolution)
    result = []
    for k in range(0, len(points) - 3, 3):
        result.extend(
            DeCasteljau(
                np.array(
                    [points[k], points[k + 1], points[k + 2], points[k + 3]]),
                U))

    return result


def getHermiteSplinePoints(points, m):
    spline_points = []
    for k in range(len(points) - 1):
        spline_points.extend([
            points[k], points[k] + (1 / 3) * m[k],
            points[k + 1] - (1 / 3) * m[k + 1]
        ])
    spline_points.append(points[-1])

    return spline_points


def SplineHermiteAutomatic(points, c, T, canvas_size):
    m = [points[1] - points[0]
         ] + [(1 - c) * (points[i + 1] - points[i - 1]) / 2
              for i in range(1, len(points) - 1)] \
        + [points[-1] - points[-2]]

    return CubicSpline(getHermiteSplinePoints(points, m),
                       int(len(T) / len(points)))


def SplineHermiteC2(points, T, canvas_size):
    N = len(points) - 2
    linf = [1 for _ in range(N + 1)]
    ldiag = [2] + [4 for _ in range(N)] + [2]
    b = [3 * (points[1] - points[0])
         ] + [3 * (points[i + 1] - points[i - 1])
              for i in range(1, N + 1)] + [3 * (points[-1] - points[-2])]

    m = solve(linf, ldiag, b)

    return CubicSpline(getHermiteSplinePoints(points, m),
                       int(len(T) / len(points)))


def SplineHermite(points, T, canvas_size):
    U = T * (len(points) - 1)

    result = []
    k = 0
    t = []
    for u in U:
        while 3 * (k + 1) < len(points) and u > k + 1:
            result.extend(
                DeCasteljau(
                    np.array([
                        points[3 * k],
                        (points[3 * k + 1] + 2 * points[3 * k]) / 3,
                        (points[3 * k + 2] + 2 * points[3 * (k + 1)]) / 3,
                        points[3 * (k + 1)]
                    ]), np.array(t), canvas_size))
            k += 1
            t = []

        t.append(u - k)

    return result


def Lagrange(points, T, canvas_size):
    n = points.shape[0] - 1
    U = T * n

    result = []
    for u in U:
        r = points.copy()
        for k in range(1, n + 1):
            for i in range(0, n - k + 1):
                r[i, :] = (i + k - u) / k * r[i, :] + (u - i) / k * r[i + 1, :]

        result.append(r[0, :])

    return result


def plotErrorCircle(curves, T, canvas_size, steps):
    circle = []
    r = min(*canvas_size) / 3
    for t in T:
        cx = canvas_size[0] / 2
        cy = canvas_size[1] / 2
        circle.append(
            [cx + r * np.cos(t * np.pi / 2), cy + r * np.sin(t * np.pi / 2)])

    E = []
    for curve in curves:
        d = map(lambda x: np.linalg.norm(x[0] - x[1]), zip(curve, circle))
        E.append(max(d))

    print("alpha_min = ", steps[np.argmin(E)])
    plt.plot(steps, E)
    plt.xlabel("alpha")
    plt.ylabel("E(alpha)")
    plt.show()


if __name__ == "__main__":
    window = CurveEditorWindow([{
        "name": "Bezier",
        "algo": DeCasteljau,
        "pointed": True,
        "color": "green"
    }, {
        "name": "R-Bezier",
        "algo": RationalDeCasteljau,
        "pointed": True,
        "weighted": True,
        "color": "lime"
    }, {
        "name": "Circle",
        "algo": quarterCircle,
        "color": "magenta"
    }, {
        "name": "ApproxCircle",
        "algo": CircleApprox4,
        "color": "gray",
        "animated": {
            "steps": np.linspace(0, 1, 50),
            "interval": 100,
            "callback": plotErrorCircle
        }
    }, {
        "name": "B-Spline",
        "algo": DeBoor,
        "pointed": True,
        "knotted": True,
        "color": "red"
    }, {
        "name": "NURBS",
        "algo": RationalDeBoor,
        "pointed": True,
        "weighted": True,
        "knotted": True,
        "color": "purple"
    }, {
        "name": "Auto H-Spline",
        "algo": SplineHermiteAutomatic,
        "pointed": True,
        "parametered": True,
        "color": "brown"
    }, {
        "name": "H-Spline",
        "algo": SplineHermite,
        "pointed": True,
        "color": "black"
    }, {
        "name": "H-Spline-C2",
        "algo": SplineHermiteC2,
        "pointed": True,
        "color": "black"
    }, {
        "name": "Lagrange",
        "algo": Lagrange,
        "pointed": True,
        "color": "black"
    }])
    window.mainloop()

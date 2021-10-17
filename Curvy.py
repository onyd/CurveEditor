from tkinter import *
import numpy as np
from itertools import chain


class CurveEditorWindow(Tk):
    def __init__(self, compute_algorithms) -> None:
        """compute_algorithms is a list of tuple {"name" : string, "algo" : function, "weighted" : bool, "knotted": bool) which 
        -string is the text in button selection of algorithm
        -function has the signature: def <name>(points, T) 
            where points is the list of points drawn on the canvas and T the array of t in [0; 1] with resolution given by the slider
        -weighted precise if weights are needeed and passed in arguments
        -knotted precise if knots are needeed and passed in arguments"""
        super().__init__()

        self.title("Curvy editor")
        self.geometry("720x520")

        self._selected_data = {"x": 0, "y": 0, 'item': None}
        self.radius = 5
        self.curve = None
        self.compute_algorithms = compute_algorithms

        self.rowconfigure([0, 1], weight=1, minsize=200)
        self.columnconfigure([0, 1], weight=1, minsize=200)

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
        self.frame_curve_type = Frame(self.frame_pannel)
        self.frame_edit_type = Frame(self.frame_pannel)
        self.frame_edit_position = Frame(self.frame_pannel)
        self.frame_resolution_sliders = Frame(self.frame_pannel)
        self.frame_weights_sliders = Frame(self.frame_pannel)
        self.frame_knots_entry = Frame(self.frame_pannel)

        # Selection of curve type
        curve_types = [algo['name'] for algo in self.compute_algorithms]
        curve_types_val = list(range(len(self.compute_algorithms)))
        self.curve_type = IntVar()
        self.curve_type.set(curve_types_val[0])

        self.radio_curve_buttons = [None] * len(self.compute_algorithms)
        for i in range(len(self.compute_algorithms)):
            self.radio_curve_buttons[i] = Radiobutton(self.frame_curve_type,
                                                      variable=self.curve_type,
                                                      text=curve_types[i],
                                                      value=curve_types_val[i],
                                                      bg="#e1e1e1")
            self.radio_curve_buttons[i].grid(row=i // 4, column=i % 4)
            self.radio_curve_buttons[i].bind(
                "<ButtonRelease-1>",
                lambda event: self.graph.after(100, lambda: self.draw_curve()))

        # Selection of edit mode
        edit_types = ['Add', 'Remove', 'Drag', 'Select']
        edit_types_val = ["add", "remove", "drag", "select"]
        self.edit_types = StringVar()
        self.edit_types.set(edit_types_val[0])

        self.radio_edit_buttons = [None] * 4
        for i in range(4):
            self.radio_edit_buttons[i] = Radiobutton(self.frame_edit_type,
                                                     variable=self.edit_types,
                                                     text=edit_types[i],
                                                     value=edit_types_val[i],
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

        self.label_knots = Label(self.frame_knots_entry, text="knots: ")
        self.knots = StringVar()
        self.entry_knots = Entry(self.frame_knots_entry,
                                 textvariable=self.knots)
        self.label_knots.pack(side=LEFT)
        self.entry_knots.pack(fill="x")

        self.entry_knots.bind("<KeyPress-Return>",
                              lambda event: self.draw_curve())
        self.entry_knots.bind("<KeyPress-KP_Enter>",
                              lambda event: self.draw_curve())

        self.frame_pannel.grid(row=0,
                               column=2,
                               padx=2,
                               pady=2,
                               rowspan=2,
                               sticky="nswe")
        self.frame_curve_type.pack(fill="x")
        self.frame_edit_type.pack(fill="x")
        self.frame_edit_position.pack(fill="x")
        self.frame_resolution_sliders.pack(fill="x")
        self.frame_weights_sliders.pack(fill="x")
        self.frame_knots_entry.pack(fill="x")

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

    def get_knots(self):
        knots = [""]
        for c in self.knots.get():
            if c != " ":
                knots[-1] += c
            else:
                knots.append("")
        return [float(knot) for knot in knots]

    def create_point(self, x, y, color):
        """Create a token at the given coordinate in the given color"""
        item = self.graph.create_oval(x - self.radius,
                                      y - self.radius,
                                      x + self.radius,
                                      y + self.radius,
                                      outline=color,
                                      fill=color,
                                      tags=("control_points", f"weight_{1.0}"))
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

    def draw_curve(self):
        self.graph.delete("curve")
        points = self.get_points()
        if len(points) <= 1:
            return

        # Select algorithm with right arguments
        k = self.curve_type.get()
        self.curve = np.array(
            self.compute_algorithms[k]['algo'](*self.get_args(
                {
                    "points":
                    self.get_points,
                    "weights":
                    self.get_weights,
                    "knots":
                    self.get_knots,
                    "T":
                    np.linspace(0, 1, self.slider_resolution.get()),
                    "canvas_size": (self.graph.winfo_width(),
                                    self.graph.winfo_height())
                },
                weighted=self.compute_algorithms[k]['weighted'],
                knotted=self.compute_algorithms[k]['knotted'])))

        colors = ['green', 'black', 'purple', 'yellow']
        ic = 0
        for i in range(0, self.curve.shape[0] - 1):
            if np.isnan(np.sum(self.curve[i, :])):
                ic = (ic + 1) % len(colors)
                continue
            if np.isnan(np.sum(self.curve[i + 1, :])):
                continue

            self.graph.create_line(self.curve[i, 0],
                                   self.curve[i, 1],
                                   self.curve[i + 1, 0],
                                   self.curve[i + 1, 1],
                                   fill=colors[ic],
                                   width=3,
                                   tags="curve")

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

        if self.edit_types.get() == "add":
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

        elif self.edit_types.get() == "remove":
            self._selected_data[
                'item'], coords, _ = self.find_closest_with_tag(
                    event.x, event.y, 3 * self.radius, "control_points")
            if self._selected_data['item'] is not None:
                self.graph.delete(self._selected_data['item'])

                self.draw_polygon()
                self.draw_curve()

        elif self.edit_types.get() == "drag":
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
        if self.edit_types.get() != "drag":
            return
        self.reset_selection()

    def handle_drag(self, event):
        """Handle dragging of an object"""
        if self.edit_types.get() != "drag" or self._selected_data[
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
        if self.edit_types.get(
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
        if self.edit_types.get(
        ) != "select" or self._selected_data['item'] is None:
            return

        coords = self.graph.coords(self._selected_data['item'])
        self.graph.move(self._selected_data['item'],
                        int(self.pos_x.get()) - coords[0],
                        int(self.pos_y.get()) - coords[1])

        self.draw_polygon()
        self.draw_curve()

    def get_args(self, kwargs, weighted, knotted):
        if weighted and knotted:
            return (np.array(kwargs['points']()),
                    np.array(kwargs['weights']()), np.array(kwargs['knots']()),
                    kwargs['T'], kwargs['canvas_size'])
        elif weighted:
            return (np.array(kwargs['points']()),
                    np.array(kwargs['weights']()), kwargs['T'],
                    kwargs['canvas_size'])
        elif knotted:
            return (np.array(kwargs['points']()), np.array(kwargs['knots']()),
                    kwargs['T'], kwargs['canvas_size'])
        else:
            return (np.array(kwargs['points']()), kwargs['T'],
                    kwargs['canvas_size'])


# ---------- Here add algorithms ----------


def DeCasteljau(points, T, canvas_size):
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
        (homogeneous_points, weights.reshape((5, 1))), axis=1)

    homogeneous_result = DeCasteljau(homogeneous_points, T, canvas_size)

    result = [p[:-1] / p[-1] for p in homogeneous_result]

    return result


def DeCasteljauWithCircle(points, T, canvas_size):
    result = DeCasteljau(points, T, canvas_size)

    # Separation of curves
    result.append(np.array([np.nan, np.nan]))
    r = min(*canvas_size) / 2 - 40
    for t in T:
        cx = canvas_size[0] / 2
        cy = canvas_size[1] / 2
        result.append(
            [cx + r * np.cos(2 * t * np.pi), cy + r * np.sin(2 * t * np.pi)])

    return result


def BothDeCastelJau(points, weights, T, canvas_size):
    result = DeCasteljau(points, T, canvas_size)
    result.append([np.nan, np.nan])
    result.extend(RationalDeCasteljau(points, weights, T, canvas_size))

    return result


def DeBoor(points, knots, T, canvas_size):
    p = len(knots) - len(points) - 1  # degree
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


def RatinalDeBoor(points, weights, knots, T, canvas_size):
    homogeneous_points = points * weights[:, np.newaxis]
    homogeneous_points = np.concatenate(
        (homogeneous_points, weights.reshape((5, 1))), axis=1)

    homogeneous_result = DeBoor(homogeneous_points, knots, T, canvas_size)

    result = [p[:-1] / p[-1] for p in homogeneous_result]
    return result


if __name__ == "__main__":
    window = CurveEditorWindow([{
        "name": "Bezier",
        "algo": DeCasteljau,
        "weighted": False,
        "knotted": False
    }, {
        "name": "R-Bezier",
        "algo": RationalDeCasteljau,
        "weighted": True,
        "knotted": False
    }, {
        "name": "Both Bezier",
        "algo": BothDeCastelJau,
        "weighted": True,
        "knotted": False
    }, {
        "name": "Bezier+Circle",
        "algo": DeCasteljauWithCircle,
        "weighted": False,
        "knotted": False
    }, {
        "name": "B-Spline",
        "algo": DeBoor,
        "weighted": False,
        "knotted": True
    }, {
        "name": "NURBS",
        "algo": RatinalDeBoor,
        "weighted": True,
        "knotted": True
    }])
    window.mainloop()

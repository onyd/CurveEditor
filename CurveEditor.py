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
from tkinter import ttk
import numpy as np
from MultiEntry import MultiEntry
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)


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

        self._selected_data = {"x": 0, "y": 0, 'item': None}
        self.radius = 5
        self.compute_algorithms = compute_algorithms
        self.n_points = 0

        self.rowconfigure([0, 1], weight=1, minsize=200)
        self.columnconfigure([0, 1], weight=1, minsize=200)

        self.columnconfigure(2, weight=0, minsize=200)

        self.setup_canvas()
        self.setup_panel()
        self.setup_plot()

    # Setup
    def setup_canvas(self):
        self.graph = Canvas(self, bd=2, cursor="plus", bg="#fbfbfb")
        self.graph.grid(row=0,
                        column=0,
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
        self.frame_control_polygon = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_edit_position = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_resolution_sliders = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_weights_sliders = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_knots_entry = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_parameters = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_buttons = Frame(self.frame_pannel, bg="#e1e1e1")

        # Parameters
        self.label_parameters = []
        self.parameters = []

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
                fg=self.compute_algorithms[i]['colors'][0])
            self.check_curve_buttons[i].grid(row=i // 3 + 1, column=i % 3)
            self.check_curve_buttons[i].bind(
                "<ButtonRelease-1>", lambda event: self.graph.after(
                    100, lambda: self.handle_algo_selection()))

        # Selection of edit mode
        self.label_edit_mode = Label(self.frame_edit_mode, text="Edit mode:")
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

        # Edit control polygon type
        self.label_cp = Label(self.frame_control_polygon,
                              text="Control polygon mode: ")
        self.label_cp.pack()
        cp_mode = ['none', 'classic', 'closed']
        self.cp_mode = StringVar()
        self.cp_mode.set(cp_mode[0])

        self.radio_cp_buttons = [None] * 3
        for i in range(3):
            self.radio_cp_buttons[i] = Radiobutton(self.frame_control_polygon,
                                                   variable=self.cp_mode,
                                                   text=cp_mode[i],
                                                   value=cp_mode[i],
                                                   bg="#e1e1e1")
            self.radio_cp_buttons[i].pack(side='left', expand=1)
            self.radio_cp_buttons[i].bind(
                "<ButtonRelease-1>", lambda event: self.graph.after(
                    100, lambda: self.draw_polygon()))

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
                                       to=500,
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
                                    bg="#e1e1e1",
                                    command=self.update_weight)
        self.slider_weights.set(1.0)
        self.label_weights.pack(side=LEFT)
        self.slider_weights.pack(fill="x")

        # Knots
        self.label_knots = Label(self.frame_knots_entry, text="knots: ")
        self.label_knots.grid(row=0, column=0)
        self.entry_knots = MultiEntry(self.frame_knots_entry, 1, [0],
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

        # Frame pack
        self.frame_pannel.grid(row=0,
                               column=2,
                               padx=2,
                               pady=2,
                               rowspan=2,
                               sticky="nswe")
        self.frame_curve_type.pack(fill="x")
        self.frame_edit_mode.pack(fill="x")
        self.frame_control_polygon.pack(fill="x")
        self.frame_edit_position.pack(fill="x")
        self.frame_resolution_sliders.pack(fill="x")

        self.button_reset = Button(self.frame_pannel, text="Reset")
        self.button_reset.pack(side=BOTTOM, fill="x")
        self.button_reset.bind("<ButtonRelease-1>", lambda event: self.reset())

    def setup_plot(self):
        self.frame_plot = Frame(self.frame_pannel, bg="#e1e1e1")

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)

        self.plot_canvas.get_tk_widget().pack()

        toolbar = NavigationToolbar2Tk(self.plot_canvas, self.frame_plot)
        toolbar.update()

        self.plot_canvas.get_tk_widget().pack()

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
        return self.entry_knots.get()

    def get_degree(self):
        return int(self.entry_degree.get())

    def get_parameters(self):
        result = {}
        for parameter, parameter_options, parameter_widget in self.parameters:
            if parameter_options['type'] == "slider":
                result[parameter] = parameter_widget.get()
            elif parameter_options['type'] == "check":
                result[parameter] = parameter_widget.instate(['selected'])
        return result

    def get_args(self, algo, parameters):
        args = {}
        if algo is not None:
            if self.is_pointed(algo):
                args['points'] = np.array(self.get_points())
            if self.is_weighted(algo):
                args['weights'] = np.array(self.get_weights())
            if self.is_knotted(algo):
                args['knots'] = np.array(self.get_knots())
                args['degree'] = self.get_degree()
            if self.is_parametered(algo):
                for parameter in algo['parameters']:
                    args[parameter] = parameters[parameter]
            if self.is_plotted(algo):
                args['plot'] = self.plot
        else:
            args['points'] = np.array(self.get_points())
            args['weights'] = np.array(self.get_weights())
            args['knots'] = np.array(self.get_knots())
            args['degree'] = self.get_degree()
            args.update(parameters)
            args['plot'] = self.plot

        return args

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

        if self.cp_mode.get() in ["classic", "closed"]:
            points = self.get_points()
            for i in range(0, len(points) - 1):
                self.graph.create_line(points[i][0],
                                       points[i][1],
                                       points[i + 1][0],
                                       points[i + 1][1],
                                       fill="blue",
                                       tags="control_polygon")
        if self.cp_mode.get() == "closed":
            self.graph.create_line(points[-1][0],
                                   points[-1][1],
                                   points[0][0],
                                   points[0][1],
                                   fill="blue",
                                   tags="control_polygon")

    def execute_algo(self, algo, parameters, *args):
        return algo['algo'](*tuple(self.get_args(algo, parameters).values()),
                            *args)

    def draw_breakline(self, points, color, tags):
        for i in range(0, points.shape[0] - 1):
            self.graph.create_line(points[i, 0],
                                   points[i, 1],
                                   points[i + 1, 0],
                                   points[i + 1, 1],
                                   fill=color,
                                   width=3,
                                   tags=tags)

    def animate(self, algo, parameters, steps, interval):
        curves = []
        for step in steps:
            self.graph.delete("animated")
            T = np.linspace(0, 1, self.slider_resolution.get())
            canvas_size = (self.graph.winfo_width(), self.graph.winfo_height())
            curves.append(
                self.execute_algo(algo, parameters, T, canvas_size, step))

            for i, curve in enumerate(curves[-1]):
                self.draw_breakline(np.array(curve),
                                    algo['colors'][i],
                                    tags=("curve", "animated"))
            self.graph.update()

            time.sleep(interval / 1000)
        algo['animated']['callback'](curves, T, canvas_size, steps)

    def draw_curve(self):
        self.graph.delete("curve")

        # Select algorithms with right arguments
        simple_algorithms = []
        animated_algorithms = []
        plotted_algorithms = []
        for i, curve_type in enumerate(self.curve_types):
            if curve_type.get() != 0:
                if "animated" in self.compute_algorithms[i]:
                    animated_algorithms.append(self.compute_algorithms[i])
                elif self.is_plotted(self.compute_algorithms[i]):
                    plotted_algorithms.append(self.compute_algorithms[i])
                else:
                    simple_algorithms.append(self.compute_algorithms[i])

        T = np.linspace(0, 1, self.slider_resolution.get())
        parameters = self.get_parameters()

        # Simple algo treatment
        for algo in simple_algorithms:
            canvas_size = (self.graph.winfo_width(), self.graph.winfo_height())
            curves = self.execute_algo(algo, parameters, T, canvas_size)
            for i, curve in enumerate(curves):
                self.draw_breakline(np.array(curve),
                                    algo['colors'][i % len(algo['colors'])],
                                    tags="curve")

        # Animated algo treatment
        for animated_algo in animated_algorithms:
            self.animate(animated_algo, parameters,
                         animated_algo['animated']['steps'],
                         animated_algo['animated']["interval"])

        # Plotted algo treatment
        self.plot.clear()
        for algo in plotted_algorithms:
            canvas_size = (self.graph.winfo_width(), self.graph.winfo_height())
            curves = self.execute_algo(algo, parameters, T, canvas_size)
            for i, curve in enumerate(curves):
                self.draw_breakline(np.array(curve),
                                    algo['colors'][i],
                                    tags="curve")

            self.plot_canvas.draw()

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
                self.draw_polygon()
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

    def reset(self):
        self.graph.delete("all")
        self.n_points = 0

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

    def is_plotted(self, algo):
        return "plotted" in algo and algo['plotted']

    def is_parametered(self, algo):
        return 'parameters' in algo

    def has_buttons(self, algo):
        return 'buttons' in algo

    def add_parameters(self, parameters):
        for parameter, parameter_options in parameters.items():
            if parameter_options['type'] == "slider":
                label_parameter = Label(self.frame_parameters,
                                        text=f"{parameter}: ")
                parameter_widget = Scale(
                    self.frame_parameters,
                    from_=parameter_options['from'],
                    to=parameter_options['to'],
                    resolution=parameter_options['resolution'],
                    orient=HORIZONTAL,
                    bg="#e1e1e1",
                    command=lambda x: self.draw_curve())
                parameter_widget.set(parameter_options['default'])
                label_parameter.pack(side=LEFT)
                parameter_widget.pack(fill="x")
                self.label_parameters.append(label_parameter)

            elif parameter_options['type'] == "check":
                parameter_widget = ttk.Checkbutton(self.frame_parameters,
                                                   text=parameter,
                                                   onvalue=True,
                                                   offvalue=False,
                                                   command=self.draw_curve)
                parameter_widget.pack(fill="x")

            self.parameters.append(
                (parameter, parameter_options, parameter_widget))

    def clear_parameters(self):
        for _, _, parameter in self.parameters:
            parameter.pack_forget()
        for label in self.label_parameters:
            label.pack_forget()

        self.label_parameters = []
        self.parameters = []

    def reduce_parameters(self, *parameters):
        """Merge idendical parameters"""
        result = {}
        for parameter in parameters:
            result.update(parameter)

        return result

    def add_buttons(self, buttons):
        for name, button_options in buttons.items():
            button = Button(
                self.frame_buttons,
                text=name,
                command=lambda: button_options['command']
                (self, **self.get_args(None, self.get_parameters())))
            button.pack(fill="x")

    def clear_buttons(self):
        for buttons in self.frame_buttons.winfo_children():
            buttons.destroy()

    def reduce_buttons(self, *buttons):
        """Merge idendical buttons"""
        result = {}
        for button in buttons:
            result.update(button)

        return result

    def handle_algo_selection(self):
        self.frame_weights_sliders.pack_forget()
        self.frame_knots_entry.pack_forget()
        self.clear_parameters()
        self.frame_parameters.pack_forget()
        self.frame_buttons.pack_forget()
        self.clear_buttons()
        self.frame_plot.pack_forget()

        # Show only necessary edition tools
        weighted = False
        knotted = False
        is_plotted = False

        all_parameters = []
        all_buttons = []
        for i, curve_type in enumerate(self.curve_types):
            if curve_type.get() != 0:
                if self.is_parametered(self.compute_algorithms[i]):
                    all_parameters.append(
                        self.compute_algorithms[i]['parameters'])
                if self.has_buttons(self.compute_algorithms[i]):
                    all_buttons.append(self.compute_algorithms[i]['buttons'])
                weighted = weighted | self.is_weighted(
                    self.compute_algorithms[i])
                knotted = knotted | self.is_knotted(self.compute_algorithms[i])
                is_plotted = is_plotted | self.is_plotted(
                    self.compute_algorithms[i])

        parameters = self.reduce_parameters(*all_parameters)
        self.add_parameters(parameters)
        buttons = self.reduce_buttons(*all_buttons)
        self.add_buttons(buttons)

        if weighted:
            self.frame_weights_sliders.pack(fill="x")
        if knotted:
            self.frame_knots_entry.pack(fill="x")
        if len(parameters) > 0:
            self.frame_parameters.pack(fill="x")
        if len(buttons) > 0:
            self.frame_buttons.pack(fill="x")
        if is_plotted:
            self.frame_plot.pack(fill="x")

        self.draw_curve()

from tkinter import Entry

from tkinter import *


class MultiEntry:
    def __init__(self, master, n, values, callback) -> None:
        assert len(values) == n

        self.frame = Frame(master)
        self.frame_left = Frame(self.frame)
        self.frame_entries = Frame(self.frame)
        self.frame_right = Frame(self.frame)

        self.callback = callback

        self.vcmd = (master.register(self.validate), '%d', '%i', '%P', '%s',
                     '%S', '%v', '%V', '%W')
        self.entries = []
        self.build_entries(n)

        self.button_add_left = Button(self.frame_left, text="+", width=2)
        self.button_rem_left = Button(self.frame_left, text="-", width=2)
        self.button_add_left.pack()
        self.button_rem_left.pack()
        self.button_add_left.bind("<ButtonPress-1>", self.add_left)
        self.button_rem_left.bind("<ButtonPress-1>", self.remove_left)

        self.place_entries()
        self.set(values, False)

        self.button_add_right = Button(self.frame_right, text="+", width=2)
        self.button_rem_right = Button(self.frame_right, text="-", width=2)
        self.button_add_right.pack()
        self.button_rem_right.pack()
        self.button_add_right.bind("<ButtonPress-1>", self.add_right)
        self.button_rem_right.bind("<ButtonPress-1>", self.remove_right)

        self.frame_left.pack(side="left")
        self.frame_entries.pack(side="left")
        self.frame_right.pack(side="left")

    def build_entries(self, n):
        for entry in self.entries:
            entry.grid_forget()

        self.entries = [
            Entry(self.frame_entries,
                  width=6,
                  validate='key',
                  validatecommand=self.vcmd) for _ in range(n)
        ]
        for entry in self.entries:
            entry.bind("<KeyPress-Return>", lambda event: self.callback())
            entry.bind("<KeyPress-KP_Enter>", lambda event: self.callback())

    def pack(self, side):
        self.frame.pack(side=side)

    def grid(self, row, column):
        self.frame.grid(row=row, column=column)

    def get(self):
        return [float(entry.get()) for entry in self.entries]

    def set(self, values, execute_callback=True):
        self.build_entries(len(values))
        self.place_entries()

        for i in range(len(self.entries)):
            self.entries[i].insert(0, f"{values[i]}")
        if execute_callback:
            self.callback()

    def place_entries(self):
        for i, entry in enumerate(self.entries):
            entry.grid_forget()
            entry.grid(row=i // 5, column=i % 5, padx=2, sticky="nsew")

    def add_left(self, event):
        self.entries.insert(
            0,
            Entry(self.frame_entries,
                  width=6,
                  validate='key',
                  validatecommand=self.vcmd))
        self.entries[0].insert(0, self.entries[1].get())

        self.place_entries()
        self.callback()

    def add_right(self, event):
        self.entries.append(
            Entry(self.frame_entries,
                  width=6,
                  validate='key',
                  validatecommand=self.vcmd))
        self.entries[-1].insert(0, self.entries[-2].get())

        self.place_entries()
        self.callback()

    def remove_left(self, event):
        self.entries[0].grid_forget()
        del self.entries[0]

        self.place_entries()
        self.callback()

    def remove_right(self, event):
        self.entries[-1].grid_forget()
        del self.entries[-1]

        self.place_entries()
        self.callback()

    def validate(self, action, index, value_if_allowed, prior_value, text,
                 validation_type, trigger_type, widget_name):
        if value_if_allowed:
            try:
                float(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False

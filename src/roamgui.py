#!/usr/bin/env python3
# Roam - Copyright 2018 David J. C. Beach; see LICENSE in project root

import math
import tkinter as tk

import numpy as np
import requests

SERVER = "http://localhost:8123"
URL_VARS = SERVER + "/vars"
URL_WEIGHTS = SERVER + "/weights"
URL_YPOS = SERVER + "/ypos"
URL_FILTER = SERVER + "/filter"
URL_DATA = SERVER + "/data"
URL_UPDATE = SERVER + "/update"
URL_TYPE = SERVER + "/type"


def get_vars():
    return requests.get(URL_VARS).json()


def get_weights():
    return requests.get(URL_WEIGHTS).json()


def get_ypos():
    return requests.get(URL_YPOS).json()


def get_filter():
    return requests.get(URL_FILTER).json()


def get_data():
    return requests.get(URL_DATA).json()


def do_update(new_weights):
    return requests.post(URL_UPDATE, json=new_weights).json()


def do_filter(indices):
    return requests.post(URL_FILTER, json=list(indices)).json()


def do_type_swap(next_type):
    return requests.post(URL_TYPE, json=next_type).json()


class Model:

    def __init__(self):
        self._vars = []
        self._data = []
        self._ypos = []
        self._weights = {}
        self._selection = set()
        self._listeners = []
        self._highlight = None
        self._filter = set()
        self._colors = []
        self._type = 1

    @property
    def vars(self):
        return self._vars

    @vars.setter
    def vars(self, vars):
        self._vars = vars
        self._notify("vars", vars)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        N = len(data)
        self._filter = set(range(N))
        self._colors = [DEFAULT_COLOR] * N
        self._notify("data", data)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
        self._notify("weights", weights)

    @property
    def ypos(self):
        return self._ypos

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        prior = self.type
        self._type = type
        self._notify("type", prior=prior, current=type)

    @ypos.setter
    def ypos(self, ypos):
        prior = self._ypos
        self._ypos = ypos
        self._notify("ypos", prior=prior, current=ypos)

    @property
    def highlight(self):
        return self._highlight

    @highlight.setter
    def highlight(self, index):
        prior = self._highlight
        if prior != index:
            self._highlight = index
            self._notify("highlight", prior=prior, current=index)

    def add_selection(self, indices):
        if not indices: return
        prior = self._selection.copy()
        self._selection.update(indices)
        self._notify("selection", prior=prior, current=self._selection)

    def remove_selection(self, indices):
        if not indices: return
        prior = self._selection.copy()
        self._selection.difference_update(indices)
        self._notify("selection", prior=prior, current=self._selection)

    def clear_selection(self):
        prior = self._selection
        if not prior: return
        self._selection = set()
        self._notify("selection", prior=prior, current=self._selection)

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, selection):
        prior = self._selection
        self._selection = set(selection)
        self._notify("selection", prior=self._selection, current=self._selection)

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, indices):
        prior = self._filter
        self._filter = set(indices)
        self._notify("filter", prior=prior, current=self._filter)

    def clear_filter(self):
        self.filter = set(range(len(self.data)))

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, colors):
        if colors is None:
            colors = [DEFAULT_COLOR] * len(self._data)
        self._colors = colors
        self._notify("colors", colors)

    def reset_colors(self):
        self.colors = None

    def _notify(self, kind, *args, **kw):
        methname = "_update_" + kind
        for listener in self._listeners:
            method = getattr(listener, methname, None)
            if method:
                method(*args, **kw)

    def add_listener(self, listener):
        self._listeners.append(listener)

    def remove_listener(self, listener):
        self._listeners.remove(listener)


DEFAULT_COLOR = "#808080"
HISTOGRAM_COLOR = "#C0C0C0"
HIGHLIGHT_COLOR = "#C00000"
SELECTION_COLOR = "#3070E0"

LABEL_FONT = "Helvetica 14"
HIGHLIGHT_FONT = "Helvetica 12 bold"

# http://colorbrewer2.org/#type=qualitative&scheme=Set2&n=8
QUALITATIVE_PALETTE = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
]

# http://colorbrewer2.org/#type=diverging&scheme=PRGn&n=8
DIVERGING_PALETTE = [
    # "#762a83",
    # "#9970ab",
    # "#c2a5cf",
    # "#e7d4e8",
    # "#d9f0d3",
    # "#a6dba0",
    # "#5aae61",
    # "#1b7837",

    # hand edited colors to be darker (greyer) in middle
    # not ideal, but hard to find diverging palettes that
    # use darker colors
    "#762a83",
    "#79508b",
    "#72456f",
    "#776478",
    "#799073",
    "#76ab70",
    "#5a7e31",
    "#1b7837",
]

def get_color_scale(var, val, palette=DIVERGING_PALETTE):
    if val is None:
        return DEFAULT_COLOR
    z = z_score(var, val)
    idx = int(math.floor(z)) + 4
    idx = min(max(idx, 0), 7)
    return palette[idx]


def get_color_cat(var, val, palette=QUALITATIVE_PALETTE):
    assert var['kind'] == 'cat', "only for categorical variables"
    cats = var['cats']
    if val is None:
        return DEFAULT_COLOR
    else:
        return palette[cats.index(val)]


class MapView:

    def __init__(self, parent, model, size=800, pointsize=8):
        self._model = model
        self._size = size
        self._canvas = tk.Canvas(parent, width=size, height=size)
        self._canvas.pack()
        self._markers = []
        self._pointsize = pointsize

        c = self._canvas
        c.bind()
        c.bind("<Button-1>", self._startselect)
        c.bind("<B1-Motion>", self._moveselect)
        c.bind("<ButtonRelease-1>", self._endselect)
        c.bind("<Button-2>", self._clear_selection)
        c.bind("<Motion>", self._movehighlight)

        self._model.add_listener(self)

    def _set_pos(self, pos):
        self._pos = pos
        self._screenpos = self._pos2screen(pos)

        c = self._canvas
        model = self._model
        ps = self._pointsize / 2

        if len(self._markers) == len(pos):
            for i in model.filter:
                sx, sy = self._screenpos[i]
                marker = self._markers[i]
                c.coords(marker, sx-ps, sy-ps, sx+ps, sy+ps)
        else:
            c.delete(tk.ALL)
            markers = []
            for i, (sx, sy) in enumerate(self._screenpos):
                marker = c.create_oval(sx-ps, sy-ps, sx+ps, sy+ps,
                    fill=DEFAULT_COLOR,
                    outline=DEFAULT_COLOR,
                    tag="datum",
                    state=tk.NORMAL if i in model.filter else tk.HIDDEN)
                markers.append(marker)
            marker2index = {m:i for i, m in enumerate(markers)}

            self._markers = markers
            self._marker2index = marker2index

    def _update_filter(self, prior, current):
        c = self._canvas
        markers = self._markers

        to_hide = prior - current
        for i in to_hide:
            c.itemconfig(markers[i], state=tk.HIDDEN)

        to_show = current - prior
        for i in to_show:
            c.itemconfig(markers[i], state=tk.NORMAL)


    def _update_ypos(self, prior, current):
        if not self._markers:
            self._set_pos(current)
            return
        oldpos = np.array(prior)
        newpos = np.array(current)
        c = self._canvas
        for i, alpha in enumerate(np.linspace(0.0, 1.0, 15)):
            mixpos = oldpos * (1 - alpha) + newpos * alpha
            c.after(i * 30, self._set_pos, mixpos.tolist())

    def _pos2screen(self, pos):
        filtpos = [pos[i] for i in self._model.filter]
        if not filtpos: filtpos = pos
        minx = min(x for x, y in filtpos)
        miny = min(y for x, y in filtpos)
        maxx = max(x for x, y in filtpos)
        maxy = max(y for x, y in filtpos)

        cx = minx + (maxx - minx) / 2
        cy = miny + (maxy - miny) / 2
        maxspan = max(maxx-minx, maxy-miny)
        halfspan = maxspan / 2

        halfwidth = self._size / 2
        scale = (halfwidth / halfspan) * 0.95
        return [(halfwidth + (px-cx) * scale, halfwidth + (py-cy) * scale) for px, py in pos]

    def _startselect(self, event):
        c = self._canvas
        cx = c.canvasx(event.x)
        cy = c.canvasy(event.y)
        self._selanchor = (cx, cy)
        c.create_rectangle(cx, cy, cx, cy, fill="#E0E0E0", tag="selrect")
        c.tag_lower("selrect")

    def _moveselect(self, event):
        c = self._canvas
        cx = c.canvasx(event.x)
        cy = c.canvasy(event.y)
        ax, ay = self._selanchor
        c.coords("selrect", ax, ay, cx, cy)

    def _endselect(self, event):
        c = self._canvas
        c.delete("selrect")
        ax, ay = self._selanchor
        cx = c.canvasx(event.x)
        cy = c.canvasy(event.y)
        items = c.find_overlapping(ax, ay, cx, cy)
        indices = [self._marker2index[m] for m in items]
        self._model.add_selection(indices)

    def _clear_selection(self, event):
        self._model.clear_selection()

    def _movehighlight(self, event):
        r = 20 # max highlight radius
        c = self._canvas
        cx = c.canvasx(event.x)
        cy = c.canvasy(event.y)
        marks = c.find_overlapping(cx-r, cy-r, cx+r, cy+r)
        if not marks:
            index = None
        else:
            indices = [self._marker2index[m] for m in marks]
            def dist(i):
                mx, my = self._screenpos[i]
                return math.sqrt((mx-cx)**2+(my-cy)**2)
            index = min(indices, key=dist)
            if dist(index) > r:
                index = None
        self._model.highlight = index

    def _update_highlight(self, prior, current):
        markers = self._markers
        marks = []
        if prior is not None:
            marks.append(markers[prior])
        if current is not None:
            marks.append(markers[current])
        self._update_markers(marks)

    def _update_selection(self, prior, current):
        idxs = prior.copy()
        idxs.update(current)
        markers = self._markers
        marks = [markers[i] for i in idxs]
        self._update_markers(marks)

    def _update_colors(self, colors):
        # update all markers as all colors have likely changed
        self._update_markers(self._markers)

    def _update_markers(self, markers):
        c = self._canvas
        model = self._model
        sel = model.selection
        highlight = model.highlight
        colors = model.colors
        marker2index = self._marker2index
        for marker in markers:
            i = marker2index[marker]
            if i == highlight:
                color = HIGHLIGHT_COLOR
            elif i in sel:
                color = SELECTION_COLOR
            else:
                color = colors[i]
            c.itemconfig(marker, fill=color, outline=color)


def z_score(var, value):
    assert var['kind'] == "scale", "z-scoring only works for scalar variables"
    if value is None:
        return None
    return (float(value) - var['mean']) / var['sd']


class HistogramBuilder:

    def z_to_bucket_scale(self, z):
        return max(-8, min(8, z*2)) + 8

    def z_to_bucket(self, z):
        return min(int(self.z_to_bucket_scale(z)), 15)

    def compute_hdata(self, vars, data):
        hdata = []
        for row in data:
            buckrow = {}
            for var in vars:
                name = var['name']
                kind = var['kind']
                val = row[name]
                if val is None:
                    bucket = None
                elif kind == "scale":
                    z = z_score(var, val)
                    bucket = self.z_to_bucket(z)
                elif kind == "cat":
                    bucket = var['cats'].index(val)
                else:
                    bucket = None
                buckrow[name] = bucket

            hdata.append(buckrow)

        return hdata

    def make_buckets(self, var):
        kind = var['kind']
        if kind == "scale":
            return [0.0] * 16
        elif kind == "cat":
            return [0.0] * len(var["cats"])
        else:
            return []

    def sum_hdata(self, vars, hdata):
        varhists = {v['name']: self.make_buckets(v) for v in vars}
        for row in hdata:
            for var in vars:
                name = var['name']
                bucket = row[name]
                if bucket is None: continue
                varhists[name][bucket] += 1

        return varhists


class VarTable:

    ROW_HEIGHT = 30
    NAME_WIDTH = 130
    HIST_WIDTH = 160
    SLIDE_WIDTH = 120
    BTN_WIDTH = 40
    PAD_WIDTH = 10

    def __init__(self, parent, model):
        self._model = model
        self._canvas = tk.Canvas(parent, width=480, height=800)
        self._canvas.pack()
        self._data = []
        self._vars = []
        self._histbuilder = HistogramBuilder()
        self._colorvar = None

        self._model.add_listener(self)
        self._canvas.bind("<Leave>", lambda e: self._hide_hover_label())

    def _update_data(self, data):
        self._data = data
        self._update_hist()

    def _row_y(self, i):
        return (i + 1) * self.ROW_HEIGHT

    def _row_yc(self, i):
        return (i + 0.5) * self.ROW_HEIGHT

    def _update_vars(self, vars):
        self._vars = vars

        c = self._canvas
        c.delete(tk.ALL)

        scales = {}
        for i, var in enumerate(vars):
            name = var['name']
            kind = var['kind']
            yc = self._row_yc(i)
            c.create_text(self.NAME_WIDTH, yc,
                text=name,
                font=LABEL_FONT,
                anchor=tk.E,
                tag="label")
            if kind != "id":
                scale = tk.Scale(c, length=self.SLIDE_WIDTH,
                    from_=0,
                    to=100,
                    showvalue=False,
                    orient=tk.HORIZONTAL)
                c.create_window(self.NAME_WIDTH+self.HIST_WIDTH+self.PAD_WIDTH*2, yc+6,
                    width=self.SLIDE_WIDTH,
                    height=self.ROW_HEIGHT,
                    anchor=tk.W,
                    window=scale,
                    tag="scale")
                scales[name] = scale

                do_colorize = lambda v: lambda: self._colorize(v)

                colorbtn = tk.Button(c, text="c", command=do_colorize(var))
                c.create_window(self.NAME_WIDTH+self.HIST_WIDTH+self.SLIDE_WIDTH+self.PAD_WIDTH*3, yc,
                    width=self.BTN_WIDTH,
                    height=self.ROW_HEIGHT,
                    anchor=tk.W,
                    window=colorbtn,
                    tag="colorbtn")

        self._scales = scales

        self._update_hist()

    def _update_weights(self, weights):

        def w2s(w):
            if w < 0.01: return 0.0
            s = (math.log10(w) + 2) * 50
            s = max(s, 0)
            return s

        for name, scale in self._scales.items():
            weight = weights.get(name, 0.0)
            scale.set(w2s(weight))

    def get_user_weights(self):

        def s2w(s):
            if s == 0: return 0.0
            w = math.pow(10, (s / 50.) - 2.)
            return w

        weights = {}
        for name, scale in self._scales.items():
            weights[name] = s2w(scale.get())

        return weights

    def _update_hist(self):
        if not (self._vars and self._data):
            return
        hdata = self._histbuilder.compute_hdata(self._vars, self._data)
        varhists = self._histbuilder.sum_hdata(self._vars, hdata)
        self._hdata = hdata
        self._varhists = varhists
        self._maxhbucket = {name: max(hist, default=None) for name, hist in varhists.items()}
        self._draw_hist(varhists, tag="hist")

    def _update_selection(self, prior, current):
        selhdata = [self._hdata[i] for i in current]
        selhists = self._histbuilder.sum_hdata(self._vars, selhdata)
        self._draw_hist(selhists, tag="histsel", color=SELECTION_COLOR)

    def _draw_hist(self, hists, tag, color=None):
        self._canvas.delete(tag)
        for i, var in enumerate(self._vars):
            name = var['name']
            kind = var['kind']
            hist = hists[name]
            if kind == "scale":
                self._draw_scale_hist(i, var, hist, tag, color=color)
            elif kind == "cat":
                self._draw_cat_hist(i, var, hist, tag, color=color)

    def _draw_scale_hist(self, i, var, hist, tag, color=None):
        c = self._canvas
        yc = self._row_yc(i)
        maxb = self._maxhbucket[var['name']]
        yscale = (self.ROW_HEIGHT - 5) / (maxb*2)
        for j, bval in enumerate(hist):
            if bval == 0: continue
            left = self.NAME_WIDTH + self.PAD_WIDTH + j*10
            barcolor = color
            if barcolor is None:
                if var is self._colorvar:
                    barcolor = DIVERGING_PALETTE[j//2]
                else:
                    barcolor = HISTOGRAM_COLOR
            c.create_rectangle(left, yc-bval*yscale, left+10, yc+bval*yscale, 
                fill=barcolor,
                outline=barcolor,
                tag=tag)

    def _draw_cat_hist(self, i, var, hist, tag, color=None):
        c = self._canvas
        yc = self._row_yc(i)
        hwidth = self.HIST_WIDTH / float(len(var['cats']))
        maxb = self._maxhbucket[var['name']]
        yscale = (self.ROW_HEIGHT - 5) / (maxb*2)
        for j, bval in enumerate(hist):
            if bval == 0: continue
            left = self.NAME_WIDTH + self.PAD_WIDTH + j*hwidth
            barcolor = color
            if barcolor is None:
                if var is self._colorvar:
                    barcolor = QUALITATIVE_PALETTE[j]
                else:
                    barcolor = HISTOGRAM_COLOR
            bar = c.create_rectangle(left+3, yc-bval*yscale, left+hwidth-3, yc+bval*yscale, 
                fill=barcolor,
                outline=barcolor,
                tag=tag)

            def make_bar_enter():
                label = var['cats'][j]
                xc = left + hwidth/2
                return lambda e: self._show_hover_label(label, xc, yc)
            def make_bar_exit():
                return lambda e: self._hide_hover_label()
            c.tag_bind(bar, "<Enter>", make_bar_enter())

    def _update_highlight(self, prior, current):
        self._canvas.delete("highlight")
        if current is None: return

        row = self._model.data[current]

        for i, var in enumerate(self._vars):
            name = var['name']
            kind = var['kind']
            val = row[name]
            if val is None:
                continue

            if kind == "scale":
                self._update_highlight_scale(i, var, val)
            elif kind == "cat":
                self._update_highlight_cat(i, var, val)
            elif kind == "id":
                self._update_highlight_id(i, var, val)

    def _update_highlight_scale(self, i, var, val):
        y = self._row_y(i)
        yc = self._row_yc(i)
        left = self.NAME_WIDTH + self.PAD_WIDTH
        z = z_score(var, val)
        bpos = self._histbuilder.z_to_bucket_scale(z)
        xpos = left + bpos * 10
        self._canvas.create_line(xpos, y+1, xpos, y+1-self.ROW_HEIGHT,
            width=2.0,
            fill=HIGHLIGHT_COLOR,
            tag="highlight")
        if bpos < 10:
            textx = xpos + 5
            textanchor = tk.W
        else:
            textx = xpos - 5
            textanchor = tk.E
        self._canvas.create_text(textx, yc,
            text=val,
            font=HIGHLIGHT_FONT,
            fill=HIGHLIGHT_COLOR,
            anchor=textanchor,
            tag="highlight")

    def _update_highlight_cat(self, i, var, val):
        yc = self._row_yc(i)
        left = self.NAME_WIDTH + self.PAD_WIDTH
        cats = var['cats']
        bucket = cats.index(val)
        nbuckets = len(cats)
        bpos = bucket + 0.5
        xpos = left + bpos * (self.HIST_WIDTH / nbuckets)
        self._canvas.create_text(xpos, yc,
            text=cats[bucket],
            font=HIGHLIGHT_FONT,
            anchor=tk.CENTER,
            fill=HIGHLIGHT_COLOR,
            tag="highlight")

    def _update_highlight_id(self, i, var, val):
        left = self.NAME_WIDTH + self.PAD_WIDTH
        yc = self._row_yc(i)        
        self._canvas.create_text(left + self.HIST_WIDTH/2, yc,
            text=val,
            font=HIGHLIGHT_FONT,
            anchor=tk.CENTER,
            fill=HIGHLIGHT_COLOR,
            tag="highlight")

    def _colorize(self, var):
        name = var['name']
        kind = var['kind']
        data = self._model.data
        if kind == "scale":
            colors = [get_color_scale(var, row[name]) for row in data]
        elif kind == "cat":
            colors = [get_color_cat(var, row[name]) for row in data]
        self._colorvar = var
        self._model.colors = colors

    def _update_colors(self, colors):
        if colors and colors[0] == DEFAULT_COLOR:
            self._colorvar = None
        self._draw_hist(self._varhists, tag="hist")
        sel = self._model.selection
        self._update_selection(sel, sel)

    def _show_hover_label(self, label, xc, yc):
        c = self._canvas
        c.delete("hoverlabel")
        c.create_text(xc, yc, text=label, font=HIGHLIGHT_FONT, tag="hoverlabel")

    def _hide_hover_label(self):
        self._canvas.delete("hoverlabel")


def main():

    model = Model()

    root = tk.Tk()
    root.winfo_toplevel().title("Roam GUI")

    plotframe = tk.LabelFrame(root, text="Plot Area")
    plot = MapView(plotframe, model, size=450)
    
    plotframe2 = tk.LabelFrame(root, text="Plot Area")
    plot2 = MapView(plotframe2, model, size=450)

    varframe = tk.LabelFrame(root, text="Variables")
    vartable = VarTable(varframe, model)

    buttonframe = tk.Frame(root)

    radio_val = tk.IntVar()
    radio1 = tk.Radiobutton(buttonframe,
                   text="PCA",
                   variable=radio_val,
                   value=1)
    radio2 = tk.Radiobutton(buttonframe,
                   text="t-SNE",
                   variable=radio_val,
                   value=2)
    # set default selection
    radio_val.set(1)

    filterbtn = tk.Button(buttonframe, text="Apply Filter")
    clearfiltbtn = tk.Button(buttonframe, text="Clear Filter")
    clearselbtn = tk.Button(buttonframe, text="Clear Selection")

    buttonframe2 = tk.Frame(root)
    updatebtn = tk.Button(buttonframe2, text="Update")
    clearcolorbtn = tk.Button(buttonframe2, text="Reset Colors")

    radio1.pack(side=tk.LEFT)
    radio2.pack(side=tk.LEFT)
    filterbtn.pack(side=tk.LEFT)
    clearfiltbtn.pack(side=tk.LEFT)
    clearselbtn.pack(side=tk.LEFT)

    updatebtn.pack(side=tk.LEFT)
    clearcolorbtn.pack(side=tk.LEFT)

    plotframe.grid(row=1, column=0)
    plotframe2.grid(row=1, column=1)
    varframe.grid(row=1, column=2)
    buttonframe.grid(row=0, column=0)
    buttonframe2.grid(row=0, column=1)

    model.vars = get_vars()
    model.weights = get_weights()
    model.ypos = get_ypos()
    model.filter = get_filter()
    model.data = get_data()

    def switch_type():
        model.type = radio_val.get()
        model.ypos = do_type_swap(model.type)
        print(model.type)

    radio1.config(command=switch_type)
    radio2.config(command=switch_type)

    def filter():
        indices = model.selection
        model.filter = indices
        model.clear_selection()
        model.ypos = do_filter(list(indices))

    filterbtn.config(command=filter)

    def clear_filter():
        model.clear_filter()
        indices = model.filter
        model.ypos = do_filter(indices)

    clearfiltbtn.config(command=clear_filter)

    clearselbtn.config(command=model.clear_selection)

    def update():
        weights = vartable.get_user_weights()
        model.ypos = do_update(weights)
        model.weights = get_weights()

    updatebtn.config(command=update)

    clearcolorbtn.config(command=model.reset_colors)

    tk.mainloop()


if __name__ == "__main__":
    main()

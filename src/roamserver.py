#!/usr/bin/env python3
# Roam - Copyright 2018 David J. C. Beach; see LICENSE in project root

import sys
import json
from io import BytesIO

import numpy as np
import pandas

import matplotlib.pyplot as plt

import bottle
from munch import Munch

import mapalgo as MAP


class AppState(object):

    def __init__(self, df):
        colinfo = get_colinfo(df)
        procdf = preprocess(df, colinfo)
        idpos = [i for i, info in enumerate(colinfo) if info.kind == "id"]
        weights = np.ones(len(colinfo), dtype=float)

        self._origdf = df
        self._colinfo = colinfo
        self._procdf = procdf
        self._idpos = idpos
        self._filter = np.arange(len(df))
        self._type = 1

        # use setter to normalize weights
        self.weights = weights

        self._map = None
        self.update()

    @property
    def origdf(self):
        return self._origdf

    @property
    def procdf(self):
        return self._procdf

    @property
    def colinfo(self):
        return self._colinfo

    @property
    def weights(self):
        return self._weights

    @property
    def wdata(self):
        return self._wdata

    @property
    def map(self):
        return self._map

    @property
    def filter(self):
        return self._filter

    @filter.setter
    def filter(self, indices):
        if not indices:
            indices = np.arange(len(self._origdf))
        else:
            indices = np.asarray(indices)
        indices.sort()
        self._filter = indices

    @weights.setter
    def weights(self, w):
        # zero any ID weights
        w = np.array(w, dtype=float)
        assert len(w) == len(self._colinfo), "incorrect number of weights"
        w[w < 0] = 0.0
        w[self._idpos] = 0.0
        wsum = np.sum(w)
        if wsum < 1.0e-6:
            # handle case of zero-sum weights
            w[:] = 1.0
            w[self._idpos] = 0.0
            wsum = np.sum(w)
        w /= wsum
        self._weights = w

        # compute weighted version of data
        mappedw = np.zeros(len(self._procdf.columns))
        for i, info in enumerate(self._colinfo):
            for pos in info.idxpos:
                mappedw[pos] = w[i]
        self._wdata = self._procdf.as_matrix() * mappedw

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        self._type = type

    def update(self):
        prior = self._map
        filt = self._filter
        newmap = MAP.create_map_from_data("data", self._wdata[filt], type=self._type, labels=filt, prior=prior)
        self._map = newmap


def get_colinfo_1(df, i, catmax=10):
    colname = df.columns[i]
    dtype = df.dtypes[i]
    count = df[[colname]].apply(pandas.Series.nunique)[0]

    mean = None
    sd = None

    colinfo = Munch(
        pos=i,
        name=colname,
        nvals=count
    )

    if dtype == np.dtype('O'):
        # presume string
        if count <= catmax:
            colinfo.kind = "cat"
            cats = list(df[colname].dropna().unique())
            cats.sort()
            colinfo.cats = cats
        else:
            colinfo.kind = "id"
    else:
        colinfo.kind = "scale"
        colinfo.mean = df[colname].mean()
        colinfo.sd = df[colname].std(ddof=0)

    return colinfo


def get_colinfo(df, catmax=10):
    return [get_colinfo_1(df, i, catmax=catmax) for i in range(len(df.columns))]


def preprocess(df, colinfo):

    catcols = [info.name for info in colinfo if info.kind == "cat"]
    scalecols = [info.name for info in colinfo if info.kind == "scale"]

    cats = pandas.get_dummies(df[catcols], columns=catcols)
    cats = cats.fillna(0)

    vals = df[scalecols]
    vals = (vals - vals.mean()) / vals.std(ddof=0)
    vals = vals.fillna(vals.mean())

    merged = cats.join(vals)

    # create mapping from colinfo names onto merged dataframe
    for info in colinfo:
        idxpos = []
        for i, colname in enumerate(merged.columns):
            if colname.startswith(info.name):
                idxpos.append(i)
        info.idxpos = idxpos

    return merged


# NumpyJSONEncoder class adapted from:
# https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)


def jsonout(callback):
    def wrapper(*args, **kw):
        result = callback(*args, **kw)
        if isinstance(result, pandas.DataFrame):
            return result.to_json(orient='records')
        return json.dumps(result, cls=NumpyJSONEncoder)
    return wrapper


app = bottle.Bottle()

appState = None

@app.get("/vars", apply=jsonout)
def vars():
    return appState.colinfo

@app.get("/data", apply=jsonout)
def data():
    return appState.origdf

@app.get("/weights", apply=jsonout)
def weights():
    return {info.name: w for info, w in zip(appState.colinfo, appState.weights)}

@app.get("/ypos", apply=jsonout)
def ypos():
    N = len(appState.origdf)
    ypos = np.zeros((N, 2))
    ypos[appState.filter] = appState.map.y
    return ypos

@app.get("/prob", apply=jsonout)
def prob():
    return appState.map.prob

@app.post("/update", apply=jsonout)
def update():
    wdict = bottle.request.json
    assert wdict, "no weights provided"
    weights = [wdict.get(info.name, 0.0) for info in appState.colinfo]
    appState.weights = weights
    appState.update()
    return ypos()

@app.get("/filter", apply=jsonout)
def get_filter():
    return appState.filter

@app.post("/filter", apply=jsonout)
def update_filter():
    indices = bottle.request.json
    appState.filter = indices
    appState.update()
    return ypos()


@app.post("/type", apply=jsonout)
def update_filter():
    appState.type = bottle.request.json
    appState.update()
    print("AppState: {}".format(appState.type))
    return ypos()


COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]

@app.get("/graph")
def graph():
    colors = [COLORS[0]] * len(appState.map.y)

    colorattr = bottle.request.params.get("c")
    coldict = {info.name: info for info in appState.colinfo}
    if colorattr and coldict.get(colorattr):        
        origdf = appState.origdf
        kind = coldict[colorattr].kind
        if kind == "cat":
            vals = origdf[colorattr].unique()
            mapping = dict(zip(vals, range(len(vals))))
            colors = [COLORS[mapping[c] % len(COLORS)] for c in origdf[colorattr]]
        elif kind == "scale":
            # LATER!
            colors = [COLORS[0]] * len(appState.map.y)

    io = BytesIO()
    y = appState.map.y
    plt.figure(figsize=(15,15))
    plt.scatter(y[:,0], y[:,1], s=50, color=colors, alpha=0.6)
    plt.savefig(io, format='png')
    bottle.response.set_header("Content-Type", "image/png")
    return io.getvalue()


# MY_DIR = os.path.abspath(os.getcwd())
# STATIC_DIR = os.path.join(MY_DIR, "web")

# @app.route("/static/<path:path>")
# def client_files(path):
#     print(f"fetching path {path}")
#     return bottle.static_file(path, root=STATIC_DIR)


if __name__ == '__main__':
    fname = sys.argv[1]
    df = pandas.read_csv(fname, na_values='?')
    print(df.describe())
    appState = AppState(df)
    bottle.run(app, host='localhost', port=8123)


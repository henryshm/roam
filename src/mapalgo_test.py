# Roam - Copyright 2018 David J. C. Beach; see LICENSE in project root

import numpy as np

import mapalgo


def test_create_map_unif():
    P = np.ones((3, 3)) - np.eye(3)
    m = mapalgo.create_map("test", P)
    print(m.y)


def test_create_map_zeros():
    P = np.ones((4, 4)) - np.eye(4)
    P[:, 0] = 0.0
    P[0, :] = 0.0
    m = mapalgo.create_map("test", P)
    print(m.y)


def test_create_map_with_prior():
    P1 = np.random.uniform(0.0, 1.0, (4, 4))
    P2 = np.random.uniform(0.0, 1.0, (4, 4))

    m1 = mapalgo.create_map("map1", P1)
    m2 = mapalgo.create_map("map2", P2, prior=m1)

    print(m1)
    print(m2)


def test_create_map_incomplete_prior():
    P1 = np.random.uniform(0.0, 1.0, (4, 4))
    P1[:, 0] = 0.0
    P1[0, :] = 0.0
    P2 = np.random.uniform(0.0, 1.0, (4, 4))

    m1 = mapalgo.create_map("map1", P1)
    m2 = mapalgo.create_map("map2", P2, prior=m1)

    print(m1)
    print(m2)


def test_translate_labels():
    src = np.array(["x", "y", "z", "a"])
    dest = np.array(["z", "x", "y", "w"])

    remap, mask = mapalgo.translate_labels(src, dest)

    print(remap)
    print(mask)

    assert np.all(remap == np.array([2, 0, 1]))
    assert np.all(mask == np.array([True, True, True, False]))
    assert np.all(src[remap] == dest[mask])


def test_create_blended_map():
    P1 = np.random.uniform(0.0, 1.0, (4, 4))
    P2 = np.random.uniform(0.0, 1.0, (4, 4))
    labels1 = np.array([0, 1, 2, 3])
    labels2 = np.array([2, 3, 4, 5])
    map1 = mapalgo.create_map("map1", P1, labels=labels1)
    map2 = mapalgo.create_map("map2", P2, labels=labels2)
    blend = mapalgo.create_blended_map("blend", [map1, map2], [0.5, 0.5])


def test_weights_to_condprob():
    weights = np.array([
        [0., 3., 5.],
        [5., 0., 5.],
        [4., 4., 4.],
    ])
    probs = mapalgo.weights_to_condprob(weights)
    expected = np.array([
        [0., 3. / 8., 5. / 8.],
        [5. / 10., 0., 5. / 10.],
        [4. / 8., 4. / 8., 0.],
    ])
    diffs = probs - expected
    print(diffs)
    assert np.all(np.abs(diffs) < 1e-6)


def test_propagate_with_decay():
    condprobs = np.array([
        [0.0, 0.25, 0.75],
        [0.5, 0.0, 0.5],
        [0.1, 0.9, 0.0],
    ])

    expected = (condprobs * (2. / 3.) + np.dot(condprobs, condprobs) * (1. / 3.))

    result = mapalgo.propagate_with_decay(condprobs, n=2, gamma=0.5)

    diff = result - expected

    print(expected)
    print(result)
    print(diff)

    assert np.all(np.abs(diff) < 1e-6)


def test_stabilize_assignments():
    assign1 = np.array([0, 0, 0, 1, 1, 2, 2])
    assign2 = np.array([1, 1, 1, 2, 2, 0, 1])

    mapping = mapalgo.stabilize_assignments(assign1, assign2)

    print(mapping)
    print(assign2)
    print(mapping[assign1])

    assert np.all(mapping == np.array([1, 2, 0]))


def test_zscore():
    X = np.array([
        [3., 3., 3., 3.],
        [0., 0., 1., 1.],
        [1., 2., 2., 3.]
    ]).T
    q = np.sqrt(2.)
    expected = np.array([
        [0., 0., 0., 0.],
        [-1., -1., 1., 1.],
        [-q, 0., 0., q],
    ]).T
    Y = mapalgo.zscore(X)
    assert np.all(np.abs(Y - expected) < 1e-6)


def test_map_mean_sse():
    labels = np.arange(3)
    y1 = np.array([[0.0, -0.0], [1.0, 0.0], [-1.0, 0.0]])
    y2 = np.array([[0.0, -1.0], [0.0, 0.0], [-1.0, 1.0]])

    map1 = mapalgo.Map2D("map1", labels, None, y1, None, None)
    map2 = mapalgo.Map2D("map2", labels, None, y2, None, None)

    assert abs(mapalgo.map_mean_sse(map1, map1)) < 1e-6
    assert abs(mapalgo.map_mean_sse(map2, map2)) < 1e-6
    assert abs(mapalgo.map_mean_sse(map1, map2) - 1.0) < 1e-6
    assert abs(mapalgo.map_mean_sse(map2, map1) - 1.0) < 1e-6

import matplotlib.pyplot as plt

from macrel import graphs
from macrel import viewmap
from macrel import vast11data

topo = vast11data.get_vast11_topology()


prob = graphs.get_prob_matrix(topo)

topomap = graphs.get_map(prob)

graph_props = vast11data.VAST11GraphProps({node.id: 10 for node in vast11data.NODES})

viewmap.view_maps([topomap], props=graph_props)
plt.show()


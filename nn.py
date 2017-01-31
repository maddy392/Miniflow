"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from miniflow import *

#x, y, z = Input(), Input(), Input()
inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)
#f = Add(x, y,z)
#g = Mul(x,y,z)

feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)
#output2 = forward_pass(g, sorted_nodes)

# NOTE: because topological_sort set the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
#print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y],feed_dict[z], output))
#print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y],feed_dict[z], output2))
print (output)
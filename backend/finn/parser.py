from logging import getLogger
import os
import json
from venv import logger

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from samo.model import Partition

from .network import FinnNetworkWrapper
from .node import FinnNodeWrapper

def parse(model, platform, freq):
    # create the computation graph
    reference = Partition()
    reference.platform = platform
    reference.freq = freq

    edges = []
    ## add nodes
    for finn_node in model.graph.node:
        size_in  = model.get_tensor_shape(finn_node.input[0])
        size_out = model.get_tensor_shape(finn_node.output[0])
        reference.add_node(finn_node.name, hw=FinnNodeWrapper(getCustomOp(finn_node), size_in, size_out))

        for i in finn_node.input:
            prev_node = model.find_producer(i)
            if prev_node != None:
                edges.append((prev_node.name, finn_node.name))

    ## add edges
    for edge in edges:
        reference.add_edge(*edge)

    for layer in reference.nodes:
        if reference.nodes[layer]['hw'].finn_node.onnx_node.op_type in ["MVAU_hls", "MVAU_rtl"] and reference.out_degree(layer) > 0:
            next_node = list(reference.successors(layer))[0]
            if reference.nodes[next_node]['hw'].finn_node.onnx_node.op_type == "AddStreams_hls":
                for prev_node in reference.predecessors(next_node):
                    reference.nodes[prev_node]['hw'].constraints["matching_inter_folding"] = True


    # create network from reference design
    network = FinnNetworkWrapper(reference)
    network.batch_size = 1
    
    return network
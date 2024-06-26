from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import get_by_name

def export(network, model):
    if len(network.partitions) > 1:
        return 

    for finn_node in model.graph.node:
        node = finn_node.name
        layer = network.partitions[0].nodes[node]["hw"]
        finn_node = getCustomOp(finn_node)

        if get_by_name(finn_node.onnx_node.attribute, "SIMD") is not None:
            finn_node.set_nodeattr("SIMD", layer.channel_in_folding)

        if get_by_name(finn_node.onnx_node.attribute, "PE") is not None:
            finn_node.set_nodeattr("PE", layer.channel_out_folding)

    return model
"""
Pipeline orchestrator. Chains L0-L3 and the fusion layer in a fixed order.

Design notes:
  - Every message flows through all layers in full; it is not "classified"
    into a single layer
  - Supports layer trimming (for ablation experiments: L1-only / L1+L2 / all layers)
"""
from layers.l0_gatekeeper import Gatekeeper
from layers.l1_hard_limits import HardLimits
from layers.l2_context import ContextLayer
from layers.l3_models import L3Layer
from fusion import AlertFusion


class Pipeline:
    def __init__(self, config, layers=None):
        self.config = config
        self.node_memory = {}                    # used by L0, persisted across messages
        self.active_alerts_memory = {}            # used by the fusion layer

        all_layers = {
            "L0": Gatekeeper(config, self.node_memory),
            "L1": HardLimits(config),
            "L2": ContextLayer(config),
            "L3": L3Layer(config),
        }
        # layers param is for ablation experiments, e.g. layers=["L0","L1"] stops after L1
        self.active_layer_names = layers or ["L0", "L1", "L2", "L3"]
        self.layer_objs = [all_layers[name] for name in self.active_layer_names]
        self.fusion = AlertFusion(config, self.active_alerts_memory)

    def run(self, snapshot):
        for layer in self.layer_objs:
            snapshot = layer.process(snapshot)
        return self.fusion.merge(snapshot)

"""
Pipeline orchestrator. Chains the active layers and the fusion layer in a
fixed order.

L0 (data gatekeeper) and L1 (hard limits) have been removed: the hardware
side now handles data-quality gatekeeping and hard-limit alerting itself
(see TB_Data_Reference_for_ML.md), so this pipeline only ever runs L2
(context/scenario rules) and L3 (ML anomaly detection).

Design notes:
  - Every message flows through all active layers in full; it is not
    "classified" into a single layer
  - Supports layer trimming (layers=["L2"] to run L2 alone, etc.) for
    diagnostics / manual training filtering / ablation-style comparisons
"""
from layers.l2_context import ContextLayer
from layers.l3_models import L3Layer
from fusion import AlertFusion


class Pipeline:
    def __init__(self, config, layers=None):
        self.config = config
        self.active_alerts_memory = {}            # used by the fusion layer

        all_layers = {
            "L2": ContextLayer(config),
            "L3": L3Layer(config),
        }
        # layers param is for e.g. running L2 alone (diagnostics/training filter)
        self.active_layer_names = layers or ["L2", "L3"]
        self.layer_objs = [all_layers[name] for name in self.active_layer_names]
        self.fusion = AlertFusion(config, self.active_alerts_memory)

    def run(self, snapshot):
        for layer in self.layer_objs:
            snapshot = layer.process(snapshot)
        return self.fusion.merge(snapshot)

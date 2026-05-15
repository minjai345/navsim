"""TransFuser agent for MAN TruckScenes.

Mirrors navsim/agents/transfuser/ structure (config / model / loss /
features / agent / callback / backbone) so the truck variant looks like
any other navsim agent. Differences from the vanilla transfuser agent are
TruckScenes-specific: 4-camera input layout, trailer trajectory head,
no-HD-map status feature composition.

Long-term plan: this agent becomes the canonical truck training stack
once the navsim-side environment (nuplan-devkit + numpy) becomes
B200-compatible. Until then the agent code is in place but
transfuser-truckscenes/train.py is what actually runs training; the
conformance test (project task #18) keeps the two implementations in
lockstep.
"""

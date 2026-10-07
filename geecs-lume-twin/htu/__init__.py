"""HTU beamline description for the transport twin.

Lattice geometry, magnet calibrations, and camera geometry ported from the
two prior digital-twin efforts (numbers, not architecture):

- BLAST-AI-ML/bella_htu_digital_twin (BSD-3-LBNL) — original Cheetah/ImpactX
  lattice + ECS-Live-Dump camera geometry extraction
- BeamTransportDigitalTwin (local, ImpactX) — refined lattice: measured EMQ
  calibrations, kicker integrated fields, chicane r56 parameterization
"""

from htu.model import build_htu_model  # noqa: F401

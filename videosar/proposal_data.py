"""Legacy ``proposal_data.py`` removed during the v2.0 Video-SAR refactor.

Why this module was retired:
- The legacy implementation mixed noise-free ``source=geometry`` 2D mask shortcuts (2/3 of
  the dataset) with ``source=sar`` scenes in a single evaluation split, masking a 0/169
  real-world transfer failure on Sandia National Laboratories (SNL) blind tests.

Migration path:
- Real benchmarks: use ``videosar.datasets.VSMBDataset`` or ``videosar.datasets.SNLDataset``.
- Physical simulation: use ``videosar.simulation.VideoSARSimulator`` (>=256x256,
  real-background shadow injection or RadarSimPy backend).
"""

from __future__ import annotations

import warnings


class LegacyProposalDataRemovedError(RuntimeError):
    """Raised when legacy mixed geometry/SAR shortcut functions are invoked."""


def load_proposal_dataset(*args, **kwargs):  # pragma: no cover - explicit guard
    warnings.warn(
        "videosar.proposal_data has been retired in v2.0 because mixing 2D geometry "
        "masks with SAR scenes invalidated domain-transfer evaluation.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise LegacyProposalDataRemovedError(
        "Legacy mixed geometry/SAR shortcut is disabled. Use "
        "videosar.datasets.VSMBDataset / SNLDataset for real benchmarks, or "
        "videosar.simulation.VideoSARSimulator for physical simulation."
    )

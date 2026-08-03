"""shearWaveProcessing (swp): unified cardiac shear-wave elastography pipeline.

Three modular stages:

1. :mod:`swp.acquisition` - convert a Verasonics ``.mat`` to zea ``.hdf5``,
   beamform RF->IQ, make B-mode GIFs, and save per-measurement shear-wave IQ
   (including the ARF push location) for downstream processing.
2. :mod:`swp.mline` - interactive (or reused) M-line selection on a B-mode frame.
3. :mod:`swp.viz` - IQ -> shear-wave space-time plots + speed (ported from iq2sws),
   for active (buffer 2 + buffer 5) and passive (buffer 4) SWE.

See ``run.py`` for the end-to-end driver.
"""

__version__ = "0.1.0"

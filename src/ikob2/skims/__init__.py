"""
Skims: origin x destination travel times (and later distances, fares).

  store   - lazily read, resumable on-disk matrices with a near/far
            destination scheme that is combined on the fly,
  zones   - zone points in WGS84 and coarse destination cells,
  walk    - walking times from zone geometry (no routing),
  router  - the routing interface and its r5py implementation,
  build   - fills a store block by block from a router.
"""

from ikob2.skims.store import SkimStore

__all__ = ["SkimStore"]

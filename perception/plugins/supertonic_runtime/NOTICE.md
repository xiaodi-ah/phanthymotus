`vendor.py` is the unmodified official Supertonic Python helper, copied from
https://github.com/supertone-oss-archive/supertonic/blob/main/py/helper.py.
SHA256: `b14b1da8a018aa4db6ca8cbb927c32590bfc889b49116ea344c56e8351d94279`.
Its MIT license is included in LICENSE. The adapter subclasses only latent sampling
to preserve the validated RNG sequence without global RNG interference and enforce
the certified TensorRT profile bound.

Model: `supertone-oss-archive/supertonic-3`, revision
`aafc6e32416a594460b32413efc49d7fe4ce6d46`, model license OpenRAIL-M.
Weights and Orin-specific engines are external assets, not source-controlled here.

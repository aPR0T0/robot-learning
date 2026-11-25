"""Small GIFs of rollout videos for the docs: python scripts/dream/make_gifs.py OUT_DIR VIDEO.mp4[=name] ...

Every second frame, scaled to ``WIDTH`` px (env), looped.
"""

import os
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
width = int(os.environ.get("WIDTH", 480))
for arg in sys.argv[2:]:
    src, _, name = arg.partition("=")
    name = name or Path(src).stem
    frames = []
    for i, f in enumerate(iio.get_reader(src)):
        if i % 2:
            continue
        im = Image.fromarray(np.asarray(f))
        h = int(im.height * width / im.width)
        frames.append(np.asarray(im.resize((width, h), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE,
                                                                             colors=128).convert("RGB")))
    iio.mimsave(out / f"{name}.gif", frames, duration=0.1, loop=0)
    print(name, len(frames), "frames", round((out / f"{name}.gif").stat().st_size / 1e6, 2), "MB")

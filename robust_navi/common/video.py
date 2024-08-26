# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path
import imageio
import numpy as np


class VideoRecorder:
    def __init__(self,
                 root_dir: tp.Optional[tp.Union[str, Path]],
                 render_size: int = 256,
                 fps: int = 20) -> None:
        self.save_dir: tp.Optional[Path] = None
        if root_dir is not None:
            self.save_dir = Path(root_dir) / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        self.enabled = False
        self.render_size = render_size
        self.fps = fps
        self.frames: tp.List[np.ndarray] = []

    def init(self, env, enabled: bool = True) -> None:
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env) -> None:
        if self.enabled:
            frame = env.render()
            self.frames.append(frame)

    def save(self, file_name: str) -> None:
        if self.enabled:
            assert self.save_dir is not None
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)  # type: ignore


class TrainVideoRecorder:
    def __init__(self,
                 root_dir: tp.Optional[tp.Union[str, Path]],
                 render_size: int = 256,
                 fps: int = 20) -> None:
        self.save_dir: tp.Optional[Path] = None
        if root_dir is not None:
            self.save_dir = Path(root_dir) / 'train_video'
            self.save_dir.mkdir(exist_ok=True)

        self.enabled = False
        self.render_size = render_size
        self.fps = fps
        self.frames: tp.List[np.ndarray] = []
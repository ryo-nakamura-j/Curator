# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
from pathlib import Path
from typing import Final, Optional, Union

import torch
from loguru import logger
from PIL import Image

from nemo_curator.utils.hf_download_utils import download_model_from_hf

from .base import ModelInterface

# Add FIBO to Python path
FIBO_PATH = Path("/home/ryonakamura/FIBO")
if str(FIBO_PATH) not in sys.path:
    sys.path.insert(0, str(FIBO_PATH))

from src.fibo_inference.prompt_to_json import get_json_prompt, load_engine
from src.fibo_inference.vlm.common import DEFAULT_SAMPLING, SamplingConfig

_FIBO_VLM_MODEL_ID: Final = "briaai/FIBO-vlm"
_FIBO_VLM_REVISION: Final = "main"


class FIBOVLMCaptioner(ModelInterface):
    """Interface for generating structured JSON captions from images using FIBO VLM.

    This model uses the FIBO Vision-Language Model to analyze images and generate
    detailed, structured JSON captions containing information about:
    - Subject and scene descriptions
    - Lighting (type, direction, quality, time of day)
    - Camera settings (angle, focal length, depth of field)
    - Composition details
    - Color palette
    - Artistic style and medium
    - Aesthetic scores

    Args:
        model_dir: Directory where model weights are stored
        model_mode: Either "local" (use local FIBO-VLM) or "gemini" (use Gemini API)
        task_mode: One of "inspire" (image only), "generate" (prompt only), or "refine" (image + prompt)
        prompt: Optional text prompt to guide caption generation (used in "generate" and "refine" modes)
        temperature: Sampling temperature for VLM generation (default: 0.2)
        top_p: Top-p sampling parameter (default: 0.9)
        max_tokens: Maximum tokens to generate (default: 4096)
    """

    def __init__(
        self,
        model_dir: str,
        model_mode: str = "local",
        task_mode: str = "inspire",
        prompt: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize the FIBOVLMCaptioner model."""
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dir = model_dir
        self.model_mode = model_mode
        self.task_mode = task_mode
        self.prompt = prompt

        # VLM sampling configuration
        self.sampling_config = SamplingConfig(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=DEFAULT_SAMPLING.stop,
        )

        # Will be initialized in setup()
        self.engine = None

    @property
    def model_id_names(self) -> list[str]:
        """Get the model ID names.

        Returns:
            A list of model IDs used by this model.
        """
        if self.model_mode == "local":
            return [_FIBO_VLM_MODEL_ID]
        else:
            # Gemini API doesn't need local weights
            return []

    def setup(self) -> None:
        """Set up the FIBO VLM model."""
        if self.model_mode == "gemini":
            # For Gemini, just verify API key is set
            import os

            if not os.getenv("GOOGLE_API_KEY"):
                logger.warning(
                    "GOOGLE_API_KEY not set. Set it with: export GOOGLE_API_KEY=your_api_key"
                )

        # Determine model name/path
        if self.model_mode == "local":
            # Use downloaded weights from model_dir
            model_path = str(Path(self.model_dir) / _FIBO_VLM_MODEL_ID)
            if not Path(model_path).exists():
                # Fall back to HuggingFace model ID if local path doesn't exist
                model_path = _FIBO_VLM_MODEL_ID
            model_name = model_path
        else:
            model_name = _FIBO_VLM_MODEL_ID

        # Load the VLM engine
        self.engine = load_engine(model_mode=self.model_mode, model_name=model_name)

        # Move model to GPU if using local mode
        if self.model_mode == "local" and torch.cuda.is_available():
            self.engine.model.to(self.device)

        logger.info(
            f"FIBO VLM initialized in {self.model_mode} mode (task_mode={self.task_mode})"
        )

    @torch.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, str, Path],
        prompt: Optional[str] = None,
        structured_prompt: Optional[str] = None,
    ) -> dict:
        """Generate structured JSON caption for an image.

        Args:
            image: PIL Image, or path to image file
            prompt: Optional text prompt to guide generation (overrides instance prompt)
            structured_prompt: Optional existing structured prompt for refinement

        Returns:
            Dictionary containing the structured JSON caption
        """
        if self.engine is None:
            msg = "FIBOVLMCaptioner model not initialized. Call setup() first."
            raise RuntimeError(msg)

        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Use provided prompt or instance prompt
        prompt_to_use = prompt if prompt is not None else self.prompt

        # Determine inputs based on task mode
        if self.task_mode == "inspire":
            # Image only - extract structured caption from image
            image_input = image
            prompt_input = None
        elif self.task_mode == "generate":
            # Prompt only - generate structured caption from text
            image_input = None
            prompt_input = prompt_to_use or "Generate a detailed description"
        elif self.task_mode == "refine":
            # Image + prompt - refine caption with additional instructions
            image_input = image
            prompt_input = prompt_to_use
        else:
            raise ValueError(f"Invalid task_mode: {self.task_mode}")

        # Generate structured JSON caption using FIBO VLM
        json_caption = get_json_prompt(
            engine=self.engine,
            model_mode=self.model_mode,
            image=image_input,
            prompt=prompt_input,
            structured_prompt=structured_prompt,
            sampling_config=self.sampling_config,
        )

        # Ensure we return a dict
        if isinstance(json_caption, str):
            try:
                json_caption = json.loads(json_caption)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON caption, returning as-is: {json_caption}")
                return {"raw_caption": json_caption}

        return json_caption  # type: ignore[return-value]

    @classmethod
    def download_weights_on_node(cls, model_dir: str) -> None:
        """Download the weights for the FIBO VLM model on the node.

        Note: This only downloads weights for local mode. Gemini API mode doesn't need weights.
        """
        model_dir_path = Path(model_dir) / _FIBO_VLM_MODEL_ID

        # Check if already downloaded
        if model_dir_path.exists() and any(model_dir_path.glob("*.safetensors")):
            logger.info(f"FIBO VLM weights already exist at: {model_dir_path}")
            return

        model_dir_path.mkdir(parents=True, exist_ok=True)

        try:
            download_model_from_hf(
                model_id=_FIBO_VLM_MODEL_ID,
                local_dir=model_dir_path,
                revision=_FIBO_VLM_REVISION,
                ignore_patterns=["*.msgpack", "*.bin", "*.ot", "*.h5", "*.gz"],
            )
            logger.info(f"FIBO VLM weights downloaded to: {model_dir_path}")
        except Exception as e:
            logger.warning(
                f"Failed to download FIBO VLM weights: {e}. "
                f"Model will be downloaded on first use by HuggingFace."
            )

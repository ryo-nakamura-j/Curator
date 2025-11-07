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
from dataclasses import dataclass
from pathlib import Path
import torch
from loguru import logger
from PIL import Image


from collections.abc import Generator

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.models.fibo_vlm import FIBOVLMCaptioner
from nemo_curator.stages.image.captioning.base import BaseCaptioningStage
from nemo_curator.tasks import ImageBatch
from nemo_curator.stages.resources import Resources

@dataclass
class FIBOJsonCaptioningStage(BaseCaptioningStage):
    """Stage for generating structured JSON captions from images using FIBO VLM.

    This stage processes image batches through the FIBO Vision-Language Model to generate
    detailed, structured JSON captions containing information about composition, lighting,
    camera settings, subjects, and more.

    Attributes:
        model_dir: Path to the FIBO VLM model directory
        model_mode: Either "local" to use local FIBO-VLM or "gemini" to use Gemini API (default: "local")
        num_gpus_per_worker: GPU allocation per worker (default: 0.25)
        model_inference_batch_size: Number of images to process at once (default: 1 for VLM)
        temperature: Sampling temperature for VLM generation (default: 0.2)
        top_p: Top-p sampling parameter (default: 0.9)
        max_tokens: Maximum tokens to generate (default: 4096)
        verbose: Enable detailed logging
        prompt: Optional text prompt to guide caption generation
        task_mode: One of "inspire" (image only), "generate" (prompt only), or "refine" (image + prompt)
    """

    model_dir: str = None
    model_mode: str = "local"  # "local" or "gemini"
    num_gpus_per_worker: float = 0.25
    model_inference_batch_size: int = 32  # VLM processes one image at a time
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 4096
    verbose: bool = False
    prompt: str = None  # Optional prompt to guide generation
    task_mode: str = "inspire"  # "inspire", "generate", or "refine"
    _name: str = "fibo_json_captioning"

    def __post_init__(self) -> None:
        if torch.cuda.is_available():
            self._resources = Resources(gpus=self.num_gpus_per_worker)
        else:
            self._resources = Resources()
    def setup_on_node(
        self, _node_info: NodeInfo | None = None, _worker_metadata: WorkerMetadata | None = None
    ) -> None:
        """Setup FIBO VLM model on node (download weights if needed)."""
        if self.model_mode == "local":
            FIBOVLMCaptioner.download_weights_on_node(self.model_dir)
        else:
            # For Gemini API mode, no download needed
            import os

            if not os.getenv("GOOGLE_API_KEY"):
                logger.warning(
                    "GOOGLE_API_KEY not set. Set it with: export GOOGLE_API_KEY=your_api_key"
                )

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Initialize the FIBO VLM model."""
        self.model = FIBOVLMCaptioner(
            model_dir=self.model_dir,
            model_mode=self.model_mode,
            task_mode=self.task_mode,
            prompt=self.prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        self.model.setup()

        if self.verbose:
            logger.info(
                f"Initialized FIBO VLM in {self.model_mode} mode (task_mode={self.task_mode})"
            )

    def process(self, task: ImageBatch) -> ImageBatch:
        """Process an image batch to generate structured JSON captions.

        Args:
            task: ImageBatch containing list of ImageObject instances with image data

        Returns:
            ImageBatch with JSON captions stored in ImageObject.json_caption field
        """
        processed_count = 0
        error_count = 0

        for batch in self.yield_next_batch(task):
            for image_obj in batch:
                try:
                    # Use image_data if available, otherwise fall back to image_path
                    image_path = None
                    if image_obj.image_data is not None:
                        # Convert numpy array to PIL Image
                        # image_data is in RGB format (H, W, C)
                        image = Image.fromarray(image_obj.image_data)
                    else:
                        # Fall back to loading from path
                        image_path = Path(image_obj.image_path)
                        if not image_path.exists():
                            logger.error(f"Image not found: {image_path}")
                            image_obj.json_caption = None
                            error_count += 1
                            continue
                        image = image_path

                    # Generate structured JSON caption using FIBO VLM
                    json_caption = self.model(image=image)

                    # Store the JSON caption in the json_caption field
                    image_obj.json_caption = json_caption

                    processed_count += 1

                    if self.verbose:
                        image_name = image_path.name if image_path else image_obj.image_id
                        logger.info(
                            f"Generated JSON caption for {image_obj.image_id} ({image_name})"
                        )
                        if isinstance(json_caption, dict) and "short_description" in json_caption:
                            logger.info(f"  Short description: {json_caption['short_description']}")

                except Exception as e:
                    logger.error(f"Error processing image {image_obj.image_id}: {e}")
                    image_obj.json_caption = None
                    error_count += 1

            if self.verbose:
                logger.info(
                    f"Generated captions for {len(batch)} images."
                )

        if self.verbose:
            logger.info(
                f"FIBO JSON captioning complete: {processed_count}/{len(task.data)} images processed, "
                f"{error_count} errors"
            )

        return task

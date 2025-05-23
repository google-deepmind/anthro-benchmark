# Copyright 2025 The Anthropomorphism Benchmark Project Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LLM client initialization and management for dialogue generation.
"""

import dataclasses


@dataclasses.dataclass
class LLMClient:
    """Base class for LLM clients."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 1000

    def generate(self, messages: list) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters to pass to the LLM

        Returns:
            Generated text response
        """
        import litellm

        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        response_text = response.choices[0].message.content
        return response_text

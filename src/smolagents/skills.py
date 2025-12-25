#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""
Skills module for smolagents.

Skills are higher-level abstractions that can bundle tools, behaviors, and capabilities.
They provide a way to organize and manage agent capabilities in a structured manner.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .tools import BaseTool
from .utils import is_valid_name


logger = logging.getLogger(__name__)


__all__ = ["BaseSkill", "Skill", "SkillRegistry"]


class BaseSkill(ABC):
    """
    Base class for agent skills.

    A skill is a higher-level abstraction that represents a capability or set of capabilities
    that can be enabled or disabled. Skills can provide tools, modify agent behavior, or add
    new functionality.
    """

    name: str

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


class Skill(BaseSkill):
    """
    A skill represents a higher-level capability that can be enabled for an agent.

    Skills can provide tools, configure behaviors, or add new capabilities to agents.
    Each skill has a name, description, and optional category for organization.

    Attributes:
        name (str): Unique identifier for the skill
        description (str): Description of what the skill does
        category (str): Optional category for organizing skills (e.g., "web", "data", "media")
        enabled (bool): Whether this skill is currently enabled
        tools (list[BaseTool]): List of tools provided by this skill
    """

    def __init__(
        self,
        name: str,
        description: str,
        tools: list[BaseTool] | None = None,
        category: str | None = None,
        enabled: bool = False,
    ):
        """
        Initialize a skill.

        Args:
            name: Unique identifier for the skill
            description: Description of what the skill does
            tools: Optional list of tools provided by this skill
            category: Optional category for organizing skills
            enabled: Whether the skill is enabled by default (default: False)
        """
        if not is_valid_name(name):
            raise ValueError(f"Skill name '{name}' must be a valid Python identifier and not a reserved keyword.")

        self.name = name
        self.description = description
        self.category = category
        self.enabled = enabled
        self.tools = tools or []

        logger.debug(f"Initialized skill: {self.name} (enabled={self.enabled})")

    def enable(self):
        """Enable this skill."""
        self.enabled = True
        logger.debug(f"Enabled skill: {self.name}")

    def disable(self):
        """Disable this skill."""
        self.enabled = False
        logger.debug(f"Disabled skill: {self.name}")

    def get_tools(self) -> list[BaseTool]:
        """
        Get the tools provided by this skill.

        Returns:
            List of tools if skill is enabled, empty list otherwise
        """
        if not self.enabled:
            return []
        return self.tools

    def __call__(self, *args, **kwargs) -> Any:
        """
        Invoke the skill.

        By default, this is a no-op. Subclasses can override to add custom behavior.
        """
        if not self.enabled:
            raise RuntimeError(f"Skill '{self.name}' is not enabled")
        return None

    def __repr__(self):
        status = "enabled" if self.enabled else "disabled"
        category = f", category={self.category}" if self.category else ""
        return f"Skill(name={self.name}, {status}{category}, tools={len(self.tools)})"


class SkillRegistry:
    """
    Registry for managing agent skills.

    The registry provides centralized management of skills, including registration,
    enabling/disabling, and retrieval of skill tools.
    """

    def __init__(self):
        """Initialize an empty skill registry."""
        self._skills: dict[str, Skill] = {}
        logger.debug("Initialized SkillRegistry")

    def register(self, skill: Skill):
        """
        Register a skill with the registry.

        Args:
            skill: The skill to register

        Raises:
            ValueError: If a skill with the same name is already registered
        """
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' is already registered")

        self._skills[skill.name] = skill
        logger.debug(f"Registered skill: {skill.name}")

    def unregister(self, skill_name: str):
        """
        Unregister a skill from the registry.

        Args:
            skill_name: Name of the skill to unregister

        Raises:
            KeyError: If the skill is not found
        """
        if skill_name not in self._skills:
            raise KeyError(f"Skill '{skill_name}' not found in registry")

        del self._skills[skill_name]
        logger.debug(f"Unregistered skill: {skill_name}")

    def get_skill(self, skill_name: str) -> Skill:
        """
        Get a skill by name.

        Args:
            skill_name: Name of the skill to retrieve

        Returns:
            The requested skill

        Raises:
            KeyError: If the skill is not found
        """
        if skill_name not in self._skills:
            raise KeyError(f"Skill '{skill_name}' not found in registry")
        return self._skills[skill_name]

    def enable_skill(self, skill_name: str):
        """
        Enable a skill by name.

        Args:
            skill_name: Name of the skill to enable

        Raises:
            KeyError: If the skill is not found
        """
        skill = self.get_skill(skill_name)
        skill.enable()

    def disable_skill(self, skill_name: str):
        """
        Disable a skill by name.

        Args:
            skill_name: Name of the skill to disable

        Raises:
            KeyError: If the skill is not found
        """
        skill = self.get_skill(skill_name)
        skill.disable()

    def get_enabled_skills(self) -> list[Skill]:
        """
        Get all enabled skills.

        Returns:
            List of enabled skills
        """
        return [skill for skill in self._skills.values() if skill.enabled]

    def get_all_tools(self) -> list[BaseTool]:
        """
        Get all tools from enabled skills.

        Returns:
            List of tools from all enabled skills
        """
        tools = []
        for skill in self.get_enabled_skills():
            tools.extend(skill.get_tools())
        return tools

    def list_skills(self) -> list[str]:
        """
        List all registered skill names.

        Returns:
            List of skill names
        """
        return list(self._skills.keys())

    def __len__(self):
        """Return the number of registered skills."""
        return len(self._skills)

    def __repr__(self):
        enabled_count = len(self.get_enabled_skills())
        return f"SkillRegistry(total={len(self._skills)}, enabled={enabled_count})"

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
Agent Skills support for smolagents.

Skills are a lightweight, open format for extending AI agent capabilities with specialized
knowledge and workflows. A skill is a folder containing a SKILL.md file with metadata and
instructions that tell an agent how to perform a specific task.

This module provides:
- Skill: A class representing a single skill loaded from a SKILL.md file
- SkillCollection: A class for managing multiple skills

Skills use progressive disclosure to manage context efficiently:
1. Discovery: At startup, agents load only the name and description of each available skill
2. Activation: When a task matches a skill's description, the agent reads the full SKILL.md instructions
3. Execution: The agent follows the instructions, optionally loading referenced files as needed
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


logger = logging.getLogger(__name__)


class SkillParseError(Exception):
    """Raised when a SKILL.md file cannot be parsed."""

    pass


class SkillValidationError(Exception):
    """Raised when a skill fails validation."""

    pass


@dataclass
class Skill:
    """
    Represents an Agent Skill loaded from a SKILL.md file.

    A skill contains metadata (name, description) and instructions that tell an agent
    how to perform a specific task. Skills can also bundle scripts, templates, and
    reference materials in their directory.

    Attributes:
        name: A short identifier for the skill (lowercase, hyphens allowed, max 64 chars).
        description: A description of what the skill does and when to use it (max 1024 chars).
        instructions: The full Markdown instructions from the SKILL.md body.
        path: The absolute path to the skill directory.
        license: Optional license information.
        compatibility: Optional environment requirements description.
        metadata: Optional arbitrary key-value mapping for additional metadata.
        allowed_tools: Optional space-delimited list of pre-approved tools (experimental).
    """

    name: str
    description: str
    instructions: str
    path: Path
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    allowed_tools: str | None = None

    def __post_init__(self):
        """Validate skill attributes after initialization."""
        self._validate()

    def _validate(self):
        """Validate that the skill meets the specification requirements."""
        # Validate name
        if not self.name:
            raise SkillValidationError("Skill name is required")
        if len(self.name) > 64:
            raise SkillValidationError(f"Skill name must be at most 64 characters, got {len(self.name)}")
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", self.name):
            raise SkillValidationError(
                f"Skill name '{self.name}' must be lowercase alphanumeric with single hyphens, "
                "not starting or ending with a hyphen"
            )
        if "--" in self.name:
            raise SkillValidationError(f"Skill name '{self.name}' must not contain consecutive hyphens")

        # Validate description
        if not self.description:
            raise SkillValidationError("Skill description is required")
        if len(self.description) > 1024:
            raise SkillValidationError(
                f"Skill description must be at most 1024 characters, got {len(self.description)}"
            )

        # Validate compatibility if provided
        if self.compatibility and len(self.compatibility) > 500:
            raise SkillValidationError(
                f"Skill compatibility must be at most 500 characters, got {len(self.compatibility)}"
            )

        # Validate that directory name matches skill name
        if self.path and self.path.name != self.name:
            raise SkillValidationError(
                f"Skill directory name '{self.path.name}' must match skill name '{self.name}'"
            )

    @classmethod
    def from_directory(cls, skill_dir: str | Path) -> "Skill":
        """
        Load a skill from a directory containing a SKILL.md file.

        Args:
            skill_dir: Path to the skill directory.

        Returns:
            A Skill instance.

        Raises:
            SkillParseError: If the SKILL.md file cannot be found or parsed.
            SkillValidationError: If the skill fails validation.
        """
        skill_dir = Path(skill_dir).resolve()

        if not skill_dir.is_dir():
            raise SkillParseError(f"Skill directory not found: {skill_dir}")

        # Find SKILL.md file (case-insensitive)
        skill_file = _find_skill_md(skill_dir)
        if skill_file is None:
            raise SkillParseError(f"No SKILL.md file found in {skill_dir}")

        content = skill_file.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(content)

        return cls(
            name=frontmatter.get("name", ""),
            description=frontmatter.get("description", ""),
            instructions=body.strip(),
            path=skill_dir,
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
            metadata=frontmatter.get("metadata", {}),
            allowed_tools=frontmatter.get("allowed-tools"),
        )

    def to_prompt_metadata(self) -> str:
        """
        Generate XML metadata for inclusion in an agent's system prompt.

        This provides the skill name, description, and location for the agent
        to decide when to activate the skill.

        Returns:
            XML string with skill metadata.
        """
        return f"""  <skill>
    <name>{self.name}</name>
    <description>{self.description}</description>
    <location>{self.path / 'SKILL.md'}</location>
  </skill>"""

    def get_full_instructions(self) -> str:
        """
        Get the full skill instructions for when the skill is activated.

        Returns:
            The complete instructions from the SKILL.md body.
        """
        return self.instructions


def _find_skill_md(skill_dir: Path) -> Path | None:
    """
    Find the SKILL.md file in a skill directory.

    Args:
        skill_dir: Path to the skill directory.

    Returns:
        Path to the SKILL.md file, or None if not found.
    """
    # Check for exact SKILL.md first, then case-insensitive
    for name in ["SKILL.md", "skill.md"]:
        skill_file = skill_dir / name
        if skill_file.is_file():
            return skill_file
    return None


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """
    Parse YAML frontmatter from a SKILL.md file.

    Args:
        content: The full content of the SKILL.md file.

    Returns:
        A tuple of (frontmatter_dict, body_content).

    Raises:
        SkillParseError: If the frontmatter is missing or invalid.
    """
    # Match YAML frontmatter between --- delimiters
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        raise SkillParseError(
            "SKILL.md must start with YAML frontmatter between '---' delimiters. "
            "Example:\n---\nname: my-skill\ndescription: My skill description\n---"
        )

    frontmatter_yaml = match.group(1)
    body = match.group(2)

    try:
        frontmatter = yaml.safe_load(frontmatter_yaml)
        if not isinstance(frontmatter, dict):
            raise SkillParseError("SKILL.md frontmatter must be a YAML dictionary")
    except yaml.YAMLError as e:
        raise SkillParseError(f"Invalid YAML in SKILL.md frontmatter: {e}")

    return frontmatter, body


class SkillCollection:
    """
    A collection of skills that can be used by an agent.

    SkillCollection provides methods to load skills from directories and generate
    prompt content for agent system prompts.

    Example:
        >>> skills = SkillCollection.from_directories(["/path/to/skills/pdf-processing", "/path/to/skills/data-analysis"])
        >>> agent = CodeAgent(tools=[], skills=skills)
    """

    def __init__(self, skills: list[Skill] | None = None):
        """
        Initialize a SkillCollection.

        Args:
            skills: Optional list of Skill instances.
        """
        self._skills: dict[str, Skill] = {}
        if skills:
            for skill in skills:
                self.add(skill)

    def add(self, skill: Skill) -> None:
        """
        Add a skill to the collection.

        Args:
            skill: The Skill to add.

        Raises:
            ValueError: If a skill with the same name already exists.
        """
        if skill.name in self._skills:
            raise ValueError(f"Skill with name '{skill.name}' already exists in collection")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """
        Get a skill by name.

        Args:
            name: The name of the skill.

        Returns:
            The Skill, or None if not found.
        """
        return self._skills.get(name)

    def __iter__(self):
        """Iterate over skills in the collection."""
        return iter(self._skills.values())

    def __len__(self) -> int:
        """Return the number of skills in the collection."""
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        """Check if a skill with the given name exists."""
        return name in self._skills

    @property
    def skills(self) -> list[Skill]:
        """Get all skills in the collection as a list."""
        return list(self._skills.values())

    @classmethod
    def from_directory(cls, base_dir: str | Path) -> "SkillCollection":
        """
        Load all skills from subdirectories of a base directory.

        Each subdirectory containing a SKILL.md file will be loaded as a skill.

        Args:
            base_dir: Path to the base directory containing skill subdirectories.

        Returns:
            A SkillCollection with all discovered skills.

        Example:
            >>> # Given directory structure:
            >>> # /skills/
            >>> #   pdf-processing/
            >>> #     SKILL.md
            >>> #   data-analysis/
            >>> #     SKILL.md
            >>> collection = SkillCollection.from_directory("/skills")
        """
        base_dir = Path(base_dir).resolve()
        if not base_dir.is_dir():
            raise SkillParseError(f"Skills directory not found: {base_dir}")

        skills = []
        for item in base_dir.iterdir():
            if item.is_dir():
                skill_file = _find_skill_md(item)
                if skill_file:
                    try:
                        skill = Skill.from_directory(item)
                        skills.append(skill)
                    except (SkillParseError, SkillValidationError) as e:
                        logger.warning(f"Skipping invalid skill in {item}: {e}")

        return cls(skills)

    @classmethod
    def from_directories(cls, skill_dirs: list[str | Path]) -> "SkillCollection":
        """
        Load skills from a list of skill directories.

        Args:
            skill_dirs: List of paths to skill directories, each containing a SKILL.md file.

        Returns:
            A SkillCollection with the loaded skills.
        """
        skills = []
        for skill_dir in skill_dirs:
            try:
                skill = Skill.from_directory(skill_dir)
                skills.append(skill)
            except (SkillParseError, SkillValidationError) as e:
                logger.warning(f"Skipping invalid skill in {skill_dir}: {e}")

        return cls(skills)

    def to_prompt(self) -> str:
        """
        Generate the <available_skills> XML block for an agent's system prompt.

        This XML contains the name, description, and location of each skill,
        allowing the agent to decide when to activate a skill.

        Returns:
            XML string with all skill metadata, or empty string if no skills.
        """
        if not self._skills:
            return ""

        skill_entries = "\n".join(skill.to_prompt_metadata() for skill in self._skills.values())
        return f"""<available_skills>
{skill_entries}
</available_skills>"""

    def validate(self, skill_dir: str | Path) -> list[str]:
        """
        Validate a skill directory and return a list of errors.

        Args:
            skill_dir: Path to the skill directory to validate.

        Returns:
            List of error messages. Empty list means the skill is valid.
        """
        errors = []
        skill_dir = Path(skill_dir).resolve()

        if not skill_dir.is_dir():
            errors.append(f"Path is not a directory: {skill_dir}")
            return errors

        skill_file = _find_skill_md(skill_dir)
        if skill_file is None:
            errors.append(f"No SKILL.md file found in {skill_dir}")
            return errors

        try:
            Skill.from_directory(skill_dir)
        except SkillParseError as e:
            errors.append(f"Parse error: {e}")
        except SkillValidationError as e:
            errors.append(f"Validation error: {e}")

        return errors


__all__ = [
    "Skill",
    "SkillCollection",
    "SkillParseError",
    "SkillValidationError",
]

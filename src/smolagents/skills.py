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

Agent Skills are a lightweight, open format for extending AI agent capabilities
with specialized knowledge and workflows. A skill is a folder containing a SKILL.md
file with metadata and instructions.

For more information, see: https://github.com/agentskills/agentskills
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# Regex pattern for valid skill names (lowercase alphanumeric and hyphens)
SKILL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")

# Maximum field lengths per specification
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_COMPATIBILITY_LENGTH = 500


@dataclass
class Skill:
    """Represents a loaded Agent Skill.

    A skill contains metadata (name, description) and instructions loaded from
    a SKILL.md file. Skills can optionally include scripts, references, and assets.

    Attributes:
        name: Short identifier (1-64 chars, lowercase alphanumeric and hyphens).
        description: Description of what the skill does and when to use it (1-1024 chars).
        path: Path to the skill directory.
        instructions: Full Markdown instructions (loaded on demand).
        license: Optional license information.
        compatibility: Optional environment requirements.
        metadata: Optional additional key-value metadata.
        allowed_tools: Optional space-delimited list of pre-approved tools.
    """

    name: str
    description: str
    path: Path | str
    instructions: str | None = None
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: str | None = None

    def __post_init__(self):
        # Ensure path is a Path object
        if isinstance(self.path, str):
            self.path = Path(self.path)
        self._validate()

    def _validate(self):
        """Validate skill fields according to the Agent Skills specification."""
        # Validate name
        if not self.name or len(self.name) > MAX_NAME_LENGTH:
            raise ValueError(f"Skill name must be 1-{MAX_NAME_LENGTH} characters, got {len(self.name) if self.name else 0}")
        if not SKILL_NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Skill name '{self.name}' is invalid. Must be lowercase letters, numbers, and hyphens only, "
                "not starting or ending with hyphen, no consecutive hyphens."
            )

        # Validate description
        if not self.description or len(self.description) > MAX_DESCRIPTION_LENGTH:
            raise ValueError(
                f"Skill description must be 1-{MAX_DESCRIPTION_LENGTH} characters, got {len(self.description) if self.description else 0}"
            )

        # Validate optional compatibility
        if self.compatibility and len(self.compatibility) > MAX_COMPATIBILITY_LENGTH:
            raise ValueError(
                f"Skill compatibility must be at most {MAX_COMPATIBILITY_LENGTH} characters, got {len(self.compatibility)}"
            )

    @property
    def skill_md_path(self) -> Path:
        """Path to the SKILL.md file."""
        return self.path / "SKILL.md"

    @property
    def scripts_path(self) -> Path | None:
        """Path to the scripts directory, if it exists."""
        scripts = self.path / "scripts"
        return scripts if scripts.is_dir() else None

    @property
    def references_path(self) -> Path | None:
        """Path to the references directory, if it exists."""
        refs = self.path / "references"
        return refs if refs.is_dir() else None

    @property
    def assets_path(self) -> Path | None:
        """Path to the assets directory, if it exists."""
        assets = self.path / "assets"
        return assets if assets.is_dir() else None

    def load_instructions(self) -> str:
        """Load full instructions from SKILL.md file.

        Returns:
            The full Markdown content of the SKILL.md file (including frontmatter).
        """
        if self.instructions is None:
            self.instructions = self.skill_md_path.read_text(encoding="utf-8")
        return self.instructions

    def to_prompt_xml(self, include_location: bool = True) -> str:
        """Generate XML representation for system prompt injection.

        Args:
            include_location: Whether to include the path location element.

        Returns:
            XML string for the skill suitable for agent system prompts.
        """
        parts = [
            "  <skill>",
            f"    <name>{_escape_xml(self.name)}</name>",
            f"    <description>{_escape_xml(self.description)}</description>",
        ]
        if include_location:
            parts.append(f"    <location>{_escape_xml(str(self.skill_md_path))}</location>")
        parts.append("  </skill>")
        return "\n".join(parts)


def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _parse_yaml_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from Markdown content.

    Args:
        content: The full content of a SKILL.md file.

    Returns:
        Tuple of (frontmatter dict, remaining markdown content).

    Raises:
        ValueError: If the frontmatter is missing or malformed.
    """
    # Check for frontmatter delimiters
    if not content.startswith("---"):
        raise ValueError("SKILL.md must start with YAML frontmatter (---)")

    # Find the closing delimiter
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        raise ValueError("SKILL.md frontmatter is not properly closed with ---")

    frontmatter_text = content[3 : 3 + end_match.start()]
    body = content[3 + end_match.end() :]

    # Parse YAML manually (simple key: value parsing to avoid yaml dependency in core)
    frontmatter: dict[str, Any] = {}
    current_key = None
    current_value_lines: list[str] = []
    in_metadata = False
    metadata: dict[str, str] = {}

    for line in frontmatter_text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Check for metadata block
        if stripped == "metadata:":
            in_metadata = True
            continue

        if in_metadata:
            # Check if we're exiting metadata (non-indented line)
            if not line.startswith(" ") and not line.startswith("\t") and ":" in stripped:
                in_metadata = False
            else:
                # Parse metadata key-value pairs
                if ":" in stripped:
                    k, v = stripped.split(":", 1)
                    metadata[k.strip()] = v.strip().strip('"').strip("'")
                continue

        # Parse top-level key-value
        if ":" in stripped:
            # Save previous key-value if any
            if current_key:
                frontmatter[current_key] = "\n".join(current_value_lines).strip().strip('"').strip("'")

            k, v = stripped.split(":", 1)
            current_key = k.strip()
            v = v.strip()
            if v:
                current_value_lines = [v.strip('"').strip("'")]
            else:
                current_value_lines = []
        elif current_key:
            current_value_lines.append(stripped)

    # Don't forget the last key
    if current_key:
        frontmatter[current_key] = "\n".join(current_value_lines).strip().strip('"').strip("'")

    if metadata:
        frontmatter["metadata"] = metadata

    return frontmatter, body


def load_skill(skill_path: str | Path) -> Skill:
    """Load a skill from a directory.

    Args:
        skill_path: Path to the skill directory containing SKILL.md.

    Returns:
        A Skill instance with metadata loaded.

    Raises:
        FileNotFoundError: If the skill directory or SKILL.md doesn't exist.
        ValueError: If the SKILL.md is malformed or missing required fields.
    """
    skill_path = Path(skill_path)

    if not skill_path.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {skill_path}")

    skill_md = skill_path / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"SKILL.md not found in: {skill_path}")

    content = skill_md.read_text(encoding="utf-8")
    frontmatter, body = _parse_yaml_frontmatter(content)

    # Validate required fields
    if "name" not in frontmatter:
        raise ValueError(f"SKILL.md in {skill_path} is missing required 'name' field")
    if "description" not in frontmatter:
        raise ValueError(f"SKILL.md in {skill_path} is missing required 'description' field")

    # Validate name matches directory name
    if frontmatter["name"] != skill_path.name:
        raise ValueError(
            f"Skill name '{frontmatter['name']}' must match directory name '{skill_path.name}'"
        )

    return Skill(
        name=frontmatter["name"],
        description=frontmatter["description"],
        path=skill_path,
        instructions=content,  # Store full content for later use
        license=frontmatter.get("license"),
        compatibility=frontmatter.get("compatibility"),
        metadata=frontmatter.get("metadata", {}),
        allowed_tools=frontmatter.get("allowed-tools"),
    )


def discover_skills(search_paths: list[str | Path]) -> list[Skill]:
    """Discover skills in the given directories.

    Scans each directory for subdirectories containing a SKILL.md file.

    Args:
        search_paths: List of directories to search for skills.

    Returns:
        List of discovered Skill instances.
    """
    skills = []

    for search_path in search_paths:
        search_path = Path(search_path)
        if not search_path.is_dir():
            logger.warning(f"Skills search path does not exist: {search_path}")
            continue

        for item in search_path.iterdir():
            if item.is_dir() and (item / "SKILL.md").is_file():
                try:
                    skill = load_skill(item)
                    skills.append(skill)
                    logger.debug(f"Discovered skill: {skill.name} at {item}")
                except Exception as e:
                    logger.warning(f"Failed to load skill from {item}: {e}")

    return skills


def generate_skills_prompt(skills: list[Skill], include_location: bool = True) -> str:
    """Generate the system prompt fragment for available skills.

    Creates an XML block describing available skills for injection into
    the agent's system prompt.

    Args:
        skills: List of skills to include in the prompt.
        include_location: Whether to include file paths in the output.

    Returns:
        XML-formatted string for available skills, or empty string if no skills.
    """
    if not skills:
        return ""

    skill_xml_parts = [skill.to_prompt_xml(include_location=include_location) for skill in skills]

    return "\n".join(
        [
            "<available_skills>",
            *skill_xml_parts,
            "</available_skills>",
        ]
    )


@dataclass
class SkillsConfig:
    """Configuration for skills support in an agent.

    Attributes:
        enabled: Whether skills support is enabled (default: False, opt-in).
        search_paths: List of directories to search for skills.
        include_location: Whether to include file paths in prompts.
    """

    enabled: bool = False
    search_paths: list[str | Path] = field(default_factory=list)
    include_location: bool = True

    def __post_init__(self):
        # Convert string paths to Path objects
        self.search_paths = [Path(p) for p in self.search_paths]


class SkillsManager:
    """Manages skills discovery and loading for an agent.

    This class handles the lifecycle of skills including discovery, loading,
    and generating prompts. It implements progressive disclosure by loading
    only metadata at startup and full instructions on demand.

    Args:
        config: Skills configuration. If None, skills are disabled.
    """

    def __init__(self, config: SkillsConfig | None = None):
        self._config = config or SkillsConfig()
        self._skills: list[Skill] = []
        self._loaded = False

    @property
    def enabled(self) -> bool:
        """Whether skills support is enabled."""
        return self._config.enabled

    @property
    def skills(self) -> list[Skill]:
        """List of discovered skills."""
        if not self._loaded and self.enabled:
            self._discover()
        return self._skills

    def _discover(self):
        """Discover and load skills from configured search paths."""
        if not self._config.search_paths:
            logger.debug("No skills search paths configured")
            self._loaded = True
            return

        self._skills = discover_skills(self._config.search_paths)
        self._loaded = True
        logger.info(f"Discovered {len(self._skills)} skills")

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name.

        Args:
            name: The skill name to look up.

        Returns:
            The Skill instance, or None if not found.
        """
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def get_skill_instructions(self, name: str) -> str | None:
        """Get the full instructions for a skill.

        This activates the skill by loading its full SKILL.md content.

        Args:
            name: The skill name.

        Returns:
            The full instructions, or None if skill not found.
        """
        skill = self.get_skill(name)
        if skill:
            return skill.load_instructions()
        return None

    def generate_prompt(self) -> str:
        """Generate the skills prompt fragment for system prompt injection.

        Returns:
            XML-formatted string of available skills, or empty string if disabled.
        """
        if not self.enabled:
            return ""
        return generate_skills_prompt(self.skills, include_location=self._config.include_location)

    def reload(self):
        """Reload skills from configured paths."""
        self._loaded = False
        self._skills = []
        self._discover()


__all__ = [
    "Skill",
    "SkillsConfig",
    "SkillsManager",
    "discover_skills",
    "generate_skills_prompt",
    "load_skill",
]

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

Skills are directory-based packages containing specialized instructions and optional scripts
that extend an agent's capabilities. Unlike tools (which are executable functions), skills
provide natural language instructions that guide the agent on how to approach specific tasks.

A skill is defined by a SKILL.md file with YAML frontmatter containing metadata and
Markdown body containing instructions.

Example SKILL.md:
```
---
name: code-review
description: Provides guidance for reviewing code for quality, security, and best practices.
---

# Code Review Skill

When reviewing code, follow these steps:
1. Check for security vulnerabilities
2. Verify error handling
3. Review code style and conventions
...
```

Skills support progressive disclosure:
- Discovery: Only frontmatter (name, description) is loaded initially (~100 tokens per skill)
- Activation: Full content is loaded when the agent decides to use a skill (~500-5000 tokens)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class SkillScope(Enum):
    """Scope where a skill was discovered."""

    USER = "user"  # User's home directory (~/.smolagents/skills/)
    WORKSPACE = "workspace"  # Project workspace (.smolagents/skills/)
    SYSTEM = "system"  # System-wide skills
    CUSTOM = "custom"  # Custom path provided by user


@dataclass
class SkillMetadata:
    """
    Metadata for a skill, parsed from SKILL.md frontmatter.

    Attributes:
        name: Unique identifier for the skill (1-64 chars, lowercase + hyphens).
        description: What the skill does and when to use it (1-1024 chars).
        path: Path to the SKILL.md file.
        scope: Where the skill was discovered.
        enabled: Whether the skill is currently enabled.
        license: Optional license information.
        version: Optional version string.
        tags: Optional list of tags for categorization.
        metadata: Optional custom key-value metadata.
    """

    name: str
    description: str
    path: Path
    scope: SkillScope = SkillScope.CUSTOM
    enabled: bool = True
    license: str | None = None
    version: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate skill metadata after initialization."""
        self._validate_name()
        self._validate_description()

    def _validate_name(self):
        """Validate skill name format."""
        if not self.name:
            raise ValueError("Skill name cannot be empty")
        if len(self.name) > 64:
            raise ValueError(f"Skill name '{self.name}' exceeds 64 characters")
        if not re.match(r"^[a-z][a-z0-9-]*$", self.name):
            raise ValueError(
                f"Skill name '{self.name}' must start with lowercase letter and contain only "
                "lowercase letters, numbers, and hyphens"
            )

    def _validate_description(self):
        """Validate skill description."""
        if not self.description:
            raise ValueError(f"Skill '{self.name}' must have a description")
        if len(self.description) > 1024:
            raise ValueError(f"Skill '{self.name}' description exceeds 1024 characters")


@dataclass
class Skill:
    """
    A skill that can be loaded and activated by an agent.

    Skills provide specialized instructions that guide agent behavior for specific tasks.
    They support progressive disclosure - metadata is loaded at discovery, full content
    only when activated.

    Attributes:
        metadata: Skill metadata from SKILL.md frontmatter.
        instructions: Full skill instructions (loaded on activation).
        is_activated: Whether the skill content has been loaded.
    """

    metadata: SkillMetadata
    instructions: str | None = None
    is_activated: bool = False

    @property
    def name(self) -> str:
        """Return the skill name."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Return the skill description."""
        return self.metadata.description

    @property
    def path(self) -> Path:
        """Return the path to the SKILL.md file."""
        return self.metadata.path

    @property
    def enabled(self) -> bool:
        """Return whether the skill is enabled."""
        return self.metadata.enabled

    @enabled.setter
    def enabled(self, value: bool):
        """Set whether the skill is enabled."""
        self.metadata.enabled = value

    def activate(self) -> str:
        """
        Activate the skill by loading its full content.

        Returns:
            The full skill instructions.

        Raises:
            FileNotFoundError: If the SKILL.md file doesn't exist.
            ValueError: If the skill content is invalid.
        """
        if self.is_activated and self.instructions is not None:
            return self.instructions

        content = self.path.read_text(encoding="utf-8")
        _, body = _parse_skill_file(content)
        self.instructions = body
        self.is_activated = True

        logger.debug(f"Activated skill '{self.name}' ({len(self.instructions)} chars)")
        return self.instructions

    def deactivate(self):
        """Deactivate the skill, freeing memory from full instructions."""
        self.instructions = None
        self.is_activated = False

    def to_discovery_prompt(self) -> str:
        """
        Return a minimal prompt for skill discovery (injected into system prompt).

        This provides just enough information for the agent to know about the skill
        without loading full content.
        """
        return f"- **{self.name}**: {self.description}"

    def to_activation_prompt(self) -> str:
        """
        Return the full activation prompt with instructions.

        Should only be called after activate().
        """
        if not self.is_activated or self.instructions is None:
            self.activate()

        return f"""## Skill: {self.name}

{self.instructions}"""

    def dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the skill."""
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "scope": self.metadata.scope.value,
            "enabled": self.enabled,
            "is_activated": self.is_activated,
            "license": self.metadata.license,
            "version": self.metadata.version,
            "tags": self.metadata.tags,
            "metadata": self.metadata.metadata,
        }


def _parse_skill_file(content: str) -> tuple[dict[str, Any], str]:
    """
    Parse a SKILL.md file into frontmatter and body.

    Args:
        content: Raw content of the SKILL.md file.

    Returns:
        Tuple of (frontmatter dict, body markdown string).

    Raises:
        ValueError: If the file format is invalid.
    """
    # Check for YAML frontmatter (starts with ---)
    if not content.startswith("---"):
        raise ValueError("SKILL.md must start with YAML frontmatter (---)")

    # Find the end of frontmatter
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        raise ValueError("SKILL.md frontmatter must end with ---")

    frontmatter_end = end_match.end() + 3  # Account for initial ---
    frontmatter_str = content[3 : end_match.start() + 3]
    body = content[frontmatter_end:].strip()

    # Parse YAML frontmatter
    try:
        import yaml

        frontmatter = yaml.safe_load(frontmatter_str)
    except Exception as e:
        raise ValueError(f"Invalid YAML frontmatter: {e}") from e

    if not isinstance(frontmatter, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML dictionary")

    return frontmatter, body


def load_skill_from_file(path: Path, scope: SkillScope = SkillScope.CUSTOM) -> Skill:
    """
    Load a skill from a SKILL.md file.

    Only loads metadata initially (progressive disclosure).
    Call skill.activate() to load full content.

    Args:
        path: Path to the SKILL.md file.
        scope: Scope where the skill was discovered.

    Returns:
        Skill instance with metadata loaded.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")

    content = path.read_text(encoding="utf-8")
    frontmatter, _ = _parse_skill_file(content)

    # Extract required fields
    name = frontmatter.get("name")
    if not name:
        raise ValueError(f"Skill at {path} missing required 'name' field")

    description = frontmatter.get("description")
    if not description:
        raise ValueError(f"Skill at {path} missing required 'description' field")

    # Extract optional fields
    metadata = SkillMetadata(
        name=name,
        description=description,
        path=path,
        scope=scope,
        enabled=frontmatter.get("enabled", True),
        license=frontmatter.get("license"),
        version=frontmatter.get("version"),
        tags=frontmatter.get("tags", []),
        metadata=frontmatter.get("metadata", {}),
    )

    return Skill(metadata=metadata)


def load_skill_from_directory(directory: Path, scope: SkillScope = SkillScope.CUSTOM) -> Skill:
    """
    Load a skill from a directory containing SKILL.md.

    Args:
        directory: Path to the skill directory.
        scope: Scope where the skill was discovered.

    Returns:
        Skill instance.

    Raises:
        FileNotFoundError: If SKILL.md doesn't exist in the directory.
        ValueError: If the skill format is invalid.
    """
    skill_file = directory / "SKILL.md"
    return load_skill_from_file(skill_file, scope)


@dataclass
class SkillLoadOutcome:
    """Result of loading skills from a directory tree."""

    skills: list[Skill]
    errors: list[tuple[Path, Exception]]

    @property
    def success_count(self) -> int:
        """Number of successfully loaded skills."""
        return len(self.skills)

    @property
    def error_count(self) -> int:
        """Number of skills that failed to load."""
        return len(self.errors)


def discover_skills(
    roots: list[Path] | None = None,
    include_user_skills: bool = True,
    include_workspace_skills: bool = True,
    workspace_path: Path | None = None,
) -> SkillLoadOutcome:
    """
    Discover skills from standard locations and custom paths.

    Standard locations (checked if corresponding flags are True):
    - User skills: ~/.smolagents/skills/
    - Workspace skills: .smolagents/skills/ (relative to workspace_path or cwd)

    Skills from later roots override earlier ones with the same name
    (workspace > user > system).

    Args:
        roots: Additional custom paths to search for skills.
        include_user_skills: Whether to search user home directory.
        include_workspace_skills: Whether to search workspace directory.
        workspace_path: Workspace root (defaults to current directory).

    Returns:
        SkillLoadOutcome with discovered skills and any errors.
    """
    search_paths: list[tuple[Path, SkillScope]] = []

    # System skills (future: package-bundled skills)
    # Currently not implemented

    # User skills
    if include_user_skills:
        user_skills_path = Path.home() / ".smolagents" / "skills"
        if user_skills_path.exists():
            search_paths.append((user_skills_path, SkillScope.USER))

    # Workspace skills
    if include_workspace_skills:
        workspace = workspace_path or Path.cwd()
        workspace_skills_path = workspace / ".smolagents" / "skills"
        if workspace_skills_path.exists():
            search_paths.append((workspace_skills_path, SkillScope.WORKSPACE))

    # Custom paths
    if roots:
        for root in roots:
            if root.exists():
                search_paths.append((root, SkillScope.CUSTOM))

    # Discover skills from all paths
    skills_by_name: dict[str, Skill] = {}
    errors: list[tuple[Path, Exception]] = []

    for search_path, scope in search_paths:
        for skill_dir in search_path.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue

            try:
                skill = load_skill_from_file(skill_file, scope)
                # Later scopes override earlier (workspace > user)
                skills_by_name[skill.name] = skill
                logger.debug(f"Discovered skill '{skill.name}' at {skill_file}")
            except Exception as e:
                errors.append((skill_file, e))
                logger.warning(f"Failed to load skill at {skill_file}: {e}")

    return SkillLoadOutcome(
        skills=list(skills_by_name.values()),
        errors=errors,
    )


class SkillRegistry:
    """
    Registry for managing skills available to an agent.

    The registry handles:
    - Skill discovery and loading
    - Enable/disable skills
    - Activation (loading full content) when needed
    - Generating prompts for agent system prompt injection

    Example:
        ```python
        registry = SkillRegistry()
        registry.discover()  # Load from standard locations

        # Or load from custom path
        registry.load_skill("/path/to/my-skill/SKILL.md")

        # Enable/disable skills
        registry.disable_skill("code-review")
        registry.enable_skill("code-review")

        # Get discovery prompt for system prompt
        discovery_prompt = registry.get_discovery_prompt()

        # Activate a skill (load full content)
        instructions = registry.activate_skill("code-review")
        ```
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._discovery_errors: list[tuple[Path, Exception]] = []

    @property
    def skills(self) -> dict[str, Skill]:
        """Return all registered skills by name."""
        return self._skills.copy()

    @property
    def enabled_skills(self) -> dict[str, Skill]:
        """Return only enabled skills."""
        return {name: skill for name, skill in self._skills.items() if skill.enabled}

    @property
    def discovery_errors(self) -> list[tuple[Path, Exception]]:
        """Return errors encountered during skill discovery."""
        return self._discovery_errors.copy()

    def discover(
        self,
        roots: list[Path] | None = None,
        include_user_skills: bool = True,
        include_workspace_skills: bool = True,
        workspace_path: Path | None = None,
    ) -> SkillLoadOutcome:
        """
        Discover and load skills from standard locations.

        Args:
            roots: Additional custom paths to search.
            include_user_skills: Whether to search ~/.smolagents/skills/.
            include_workspace_skills: Whether to search .smolagents/skills/.
            workspace_path: Workspace root for workspace skills.

        Returns:
            SkillLoadOutcome with results.
        """
        outcome = discover_skills(
            roots=roots,
            include_user_skills=include_user_skills,
            include_workspace_skills=include_workspace_skills,
            workspace_path=workspace_path,
        )

        for skill in outcome.skills:
            self._skills[skill.name] = skill

        self._discovery_errors.extend(outcome.errors)

        logger.info(f"Discovered {outcome.success_count} skills, {outcome.error_count} errors")
        return outcome

    def register_skill(self, skill: Skill):
        """
        Register a skill in the registry.

        Args:
            skill: Skill to register.
        """
        self._skills[skill.name] = skill
        logger.debug(f"Registered skill '{skill.name}'")

    def load_skill(
        self,
        path: str | Path,
        scope: SkillScope = SkillScope.CUSTOM,
    ) -> Skill:
        """
        Load and register a skill from a path.

        Args:
            path: Path to SKILL.md file or directory containing it.
            scope: Scope to assign to the skill.

        Returns:
            The loaded skill.

        Raises:
            FileNotFoundError: If the skill file doesn't exist.
            ValueError: If the skill format is invalid.
        """
        path = Path(path)

        if path.is_dir():
            skill = load_skill_from_directory(path, scope)
        else:
            skill = load_skill_from_file(path, scope)

        self.register_skill(skill)
        return skill

    def get_skill(self, name: str) -> Skill | None:
        """
        Get a skill by name.

        Args:
            name: Skill name.

        Returns:
            The skill, or None if not found.
        """
        return self._skills.get(name)

    def enable_skill(self, name: str) -> bool:
        """
        Enable a skill.

        Args:
            name: Skill name.

        Returns:
            True if the skill was found and enabled.
        """
        skill = self._skills.get(name)
        if skill:
            skill.enabled = True
            return True
        return False

    def disable_skill(self, name: str) -> bool:
        """
        Disable a skill.

        Args:
            name: Skill name.

        Returns:
            True if the skill was found and disabled.
        """
        skill = self._skills.get(name)
        if skill:
            skill.enabled = False
            return True
        return False

    def activate_skill(self, name: str) -> str:
        """
        Activate a skill and return its full instructions.

        Args:
            name: Skill name.

        Returns:
            Full skill instructions.

        Raises:
            KeyError: If the skill is not found.
            ValueError: If the skill is disabled.
        """
        skill = self._skills.get(name)
        if not skill:
            raise KeyError(f"Skill '{name}' not found")
        if not skill.enabled:
            raise ValueError(f"Skill '{name}' is disabled")

        return skill.activate()

    def deactivate_skill(self, name: str) -> bool:
        """
        Deactivate a skill (free memory from full instructions).

        Args:
            name: Skill name.

        Returns:
            True if the skill was found and deactivated.
        """
        skill = self._skills.get(name)
        if skill:
            skill.deactivate()
            return True
        return False

    def get_discovery_prompt(self) -> str:
        """
        Generate a discovery prompt listing all enabled skills.

        This is injected into the agent's system prompt to let it know
        what skills are available.

        Returns:
            Formatted prompt string listing available skills.
        """
        enabled = self.enabled_skills
        if not enabled:
            return ""

        lines = [
            "## Available Skills",
            "",
            "You have access to the following skills. When you need specialized guidance,",
            "you can activate a skill to receive detailed instructions.",
            "",
        ]

        for skill in enabled.values():
            lines.append(skill.to_discovery_prompt())

        lines.append("")
        lines.append(
            "To use a skill, indicate that you want to activate it, and the full "
            "instructions will be provided."
        )

        return "\n".join(lines)

    def get_activated_skills_prompt(self) -> str:
        """
        Generate a prompt with all currently activated skills' instructions.

        Returns:
            Formatted prompt string with active skill instructions.
        """
        activated = [skill for skill in self._skills.values() if skill.is_activated and skill.enabled]

        if not activated:
            return ""

        lines = ["## Active Skill Instructions", ""]

        for skill in activated:
            lines.append(skill.to_activation_prompt())
            lines.append("")

        return "\n".join(lines)

    def reset(self):
        """Clear all registered skills."""
        self._skills.clear()
        self._discovery_errors.clear()

    def list_skills(self) -> list[dict[str, Any]]:
        """
        List all registered skills with their metadata.

        Returns:
            List of skill dictionaries.
        """
        return [skill.dict() for skill in self._skills.values()]


__all__ = [
    "Skill",
    "SkillMetadata",
    "SkillRegistry",
    "SkillScope",
    "SkillLoadOutcome",
    "discover_skills",
    "load_skill_from_file",
    "load_skill_from_directory",
]

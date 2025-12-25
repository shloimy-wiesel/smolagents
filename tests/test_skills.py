# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import tempfile
from pathlib import Path

import pytest

from smolagents.skills import (
    Skill,
    SkillCollection,
    SkillParseError,
    SkillValidationError,
)


@pytest.fixture
def test_skill_dir():
    """Path to the test skill directory fixture."""
    return Path(__file__).parent / "fixtures" / "test_skill"


@pytest.fixture
def pdf_processing_skill_dir(test_skill_dir):
    """Path to the pdf-processing skill fixture."""
    return test_skill_dir / "pdf-processing"


@pytest.fixture
def data_analysis_skill_dir(test_skill_dir):
    """Path to the data-analysis skill fixture."""
    return test_skill_dir / "data-analysis"


class TestSkill:
    """Tests for the Skill class."""

    def test_skill_from_directory(self, pdf_processing_skill_dir):
        """Test loading a skill from a directory."""
        skill = Skill.from_directory(pdf_processing_skill_dir)

        assert skill.name == "pdf-processing"
        assert "Extract text and tables from PDF files" in skill.description
        assert skill.license == "MIT"
        assert skill.compatibility == "Requires pdfplumber and PyPDF2 packages"
        assert skill.metadata == {"author": "test-org", "version": "1.0"}
        assert "# PDF Processing" in skill.instructions
        assert skill.path == pdf_processing_skill_dir.resolve()

    def test_skill_from_directory_minimal(self, data_analysis_skill_dir):
        """Test loading a skill with minimal frontmatter."""
        skill = Skill.from_directory(data_analysis_skill_dir)

        assert skill.name == "data-analysis"
        assert "Analyzes datasets" in skill.description
        assert skill.license is None
        assert skill.compatibility is None
        assert skill.metadata == {}

    def test_skill_from_directory_not_found(self):
        """Test error when skill directory doesn't exist."""
        with pytest.raises(SkillParseError, match="Skill directory not found"):
            Skill.from_directory("/nonexistent/path")

    def test_skill_from_directory_no_skill_md(self, tmp_path):
        """Test error when SKILL.md file is missing."""
        with pytest.raises(SkillParseError, match="No SKILL.md file found"):
            Skill.from_directory(tmp_path)

    def test_skill_name_validation_uppercase(self, tmp_path):
        """Test that uppercase names are rejected."""
        skill_dir = tmp_path / "Invalid-Name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: Invalid-Name\ndescription: Test\n---\nInstructions"
        )
        with pytest.raises(SkillValidationError, match="lowercase alphanumeric"):
            Skill.from_directory(skill_dir)

    def test_skill_name_validation_consecutive_hyphens(self, tmp_path):
        """Test that consecutive hyphens in names are rejected."""
        skill_dir = tmp_path / "invalid--name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: invalid--name\ndescription: Test\n---\nInstructions"
        )
        with pytest.raises(SkillValidationError, match="single hyphens"):
            Skill.from_directory(skill_dir)

    def test_skill_name_validation_starting_hyphen(self, tmp_path):
        """Test that names starting with hyphen are rejected."""
        skill_dir = tmp_path / "-invalid"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: -invalid\ndescription: Test\n---\nInstructions"
        )
        with pytest.raises(SkillValidationError, match="lowercase alphanumeric"):
            Skill.from_directory(skill_dir)

    def test_skill_name_validation_too_long(self, tmp_path):
        """Test that names longer than 64 characters are rejected."""
        long_name = "a" * 65
        skill_dir = tmp_path / long_name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {long_name}\ndescription: Test\n---\nInstructions"
        )
        with pytest.raises(SkillValidationError, match="at most 64 characters"):
            Skill.from_directory(skill_dir)

    def test_skill_name_directory_mismatch(self, tmp_path):
        """Test that skill name must match directory name."""
        skill_dir = tmp_path / "wrong-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: correct-name\ndescription: Test\n---\nInstructions"
        )
        with pytest.raises(SkillValidationError, match="must match skill name"):
            Skill.from_directory(skill_dir)

    def test_skill_missing_name(self, tmp_path):
        """Test error when name is missing from frontmatter."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: Test description\n---\nInstructions"
        )
        with pytest.raises(SkillValidationError, match="name is required"):
            Skill.from_directory(skill_dir)

    def test_skill_missing_description(self, tmp_path):
        """Test error when description is missing from frontmatter."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\n---\nInstructions"
        )
        with pytest.raises(SkillValidationError, match="description is required"):
            Skill.from_directory(skill_dir)

    def test_skill_missing_frontmatter(self, tmp_path):
        """Test error when frontmatter is missing."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Just instructions without frontmatter")
        with pytest.raises(SkillParseError, match="must start with YAML frontmatter"):
            Skill.from_directory(skill_dir)

    def test_skill_invalid_yaml_frontmatter(self, tmp_path):
        """Test error when frontmatter has invalid YAML."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: [invalid yaml\n---\nInstructions"
        )
        with pytest.raises(SkillParseError, match="Invalid YAML"):
            Skill.from_directory(skill_dir)

    def test_skill_to_prompt_metadata(self, pdf_processing_skill_dir):
        """Test generating XML metadata for system prompt."""
        skill = Skill.from_directory(pdf_processing_skill_dir)
        metadata = skill.to_prompt_metadata()

        assert "<skill>" in metadata
        assert "<name>pdf-processing</name>" in metadata
        assert "Extract text and tables" in metadata
        assert "<location>" in metadata
        assert "SKILL.md</location>" in metadata

    def test_skill_get_full_instructions(self, pdf_processing_skill_dir):
        """Test getting full instructions."""
        skill = Skill.from_directory(pdf_processing_skill_dir)
        instructions = skill.get_full_instructions()

        assert "# PDF Processing" in instructions
        assert "How to extract text" in instructions
        assert "How to fill forms" in instructions


class TestSkillCollection:
    """Tests for the SkillCollection class."""

    def test_collection_from_directory(self, test_skill_dir):
        """Test loading all skills from a base directory."""
        collection = SkillCollection.from_directory(test_skill_dir)

        assert len(collection) == 2
        assert "pdf-processing" in collection
        assert "data-analysis" in collection

    def test_collection_from_directories(self, pdf_processing_skill_dir, data_analysis_skill_dir):
        """Test loading skills from a list of directories."""
        collection = SkillCollection.from_directories([
            pdf_processing_skill_dir,
            data_analysis_skill_dir,
        ])

        assert len(collection) == 2
        assert "pdf-processing" in collection
        assert "data-analysis" in collection

    def test_collection_add_skill(self, pdf_processing_skill_dir):
        """Test adding a skill to a collection."""
        collection = SkillCollection()
        skill = Skill.from_directory(pdf_processing_skill_dir)
        collection.add(skill)

        assert len(collection) == 1
        assert "pdf-processing" in collection

    def test_collection_add_duplicate_raises(self, pdf_processing_skill_dir):
        """Test that adding a duplicate skill raises an error."""
        collection = SkillCollection()
        skill = Skill.from_directory(pdf_processing_skill_dir)
        collection.add(skill)

        with pytest.raises(ValueError, match="already exists"):
            collection.add(skill)

    def test_collection_get_skill(self, pdf_processing_skill_dir):
        """Test getting a skill by name."""
        collection = SkillCollection()
        skill = Skill.from_directory(pdf_processing_skill_dir)
        collection.add(skill)

        retrieved = collection.get("pdf-processing")
        assert retrieved == skill

        assert collection.get("nonexistent") is None

    def test_collection_iterate(self, test_skill_dir):
        """Test iterating over skills in a collection."""
        collection = SkillCollection.from_directory(test_skill_dir)

        skill_names = [skill.name for skill in collection]
        assert set(skill_names) == {"pdf-processing", "data-analysis"}

    def test_collection_skills_property(self, test_skill_dir):
        """Test the skills property returns a list."""
        collection = SkillCollection.from_directory(test_skill_dir)
        skills = collection.skills

        assert isinstance(skills, list)
        assert len(skills) == 2

    def test_collection_to_prompt(self, test_skill_dir):
        """Test generating the available_skills XML block."""
        collection = SkillCollection.from_directory(test_skill_dir)
        prompt = collection.to_prompt()

        assert "<available_skills>" in prompt
        assert "</available_skills>" in prompt
        assert "<skill>" in prompt
        assert "<name>pdf-processing</name>" in prompt
        assert "<name>data-analysis</name>" in prompt

    def test_empty_collection_to_prompt(self):
        """Test that empty collection returns empty string."""
        collection = SkillCollection()
        assert collection.to_prompt() == ""

    def test_collection_validate_valid_skill(self, pdf_processing_skill_dir):
        """Test validating a valid skill directory."""
        collection = SkillCollection()
        errors = collection.validate(pdf_processing_skill_dir)
        assert errors == []

    def test_collection_validate_invalid_skill(self, tmp_path):
        """Test validating an invalid skill directory."""
        collection = SkillCollection()
        errors = collection.validate(tmp_path)
        assert len(errors) > 0
        assert "No SKILL.md file found" in errors[0]

    def test_collection_from_directory_skips_invalid(self, tmp_path):
        """Test that invalid skill directories are skipped with warning."""
        # Create a valid skill
        valid_dir = tmp_path / "valid-skill"
        valid_dir.mkdir()
        (valid_dir / "SKILL.md").write_text(
            "---\nname: valid-skill\ndescription: A valid skill\n---\nInstructions"
        )

        # Create an invalid skill (no SKILL.md)
        invalid_dir = tmp_path / "invalid-skill"
        invalid_dir.mkdir()

        collection = SkillCollection.from_directory(tmp_path)

        # Should only contain the valid skill
        assert len(collection) == 1
        assert "valid-skill" in collection

    def test_collection_from_nonexistent_directory(self):
        """Test error when base directory doesn't exist."""
        with pytest.raises(SkillParseError, match="Skills directory not found"):
            SkillCollection.from_directory("/nonexistent/path")


class TestSkillIntegrationWithAgent:
    """Tests for skill integration with agents."""

    def test_agent_with_no_skills(self):
        """Test that agents work without skills (default behavior)."""
        from unittest.mock import MagicMock

        from smolagents import CodeAgent

        mock_model = MagicMock()
        agent = CodeAgent(tools=[], model=mock_model)

        # Skills should be an empty collection by default
        assert isinstance(agent.skills, SkillCollection)
        assert len(agent.skills) == 0

        # System prompt should not contain skills section
        system_prompt = agent.system_prompt
        assert "<available_skills>" not in system_prompt

    def test_agent_with_skill_collection(self, test_skill_dir):
        """Test that agents can be initialized with a SkillCollection."""
        from unittest.mock import MagicMock

        from smolagents import CodeAgent

        mock_model = MagicMock()
        skills = SkillCollection.from_directory(test_skill_dir)
        agent = CodeAgent(tools=[], model=mock_model, skills=skills)

        assert len(agent.skills) == 2

        # System prompt should contain skills section
        system_prompt = agent.system_prompt
        assert "<available_skills>" in system_prompt
        assert "pdf-processing" in system_prompt
        assert "data-analysis" in system_prompt

    def test_agent_with_skill_list(self, pdf_processing_skill_dir, data_analysis_skill_dir):
        """Test that agents can be initialized with a list of Skill objects."""
        from unittest.mock import MagicMock

        from smolagents import CodeAgent

        mock_model = MagicMock()
        skill1 = Skill.from_directory(pdf_processing_skill_dir)
        skill2 = Skill.from_directory(data_analysis_skill_dir)
        agent = CodeAgent(tools=[], model=mock_model, skills=[skill1, skill2])

        assert len(agent.skills) == 2
        assert "pdf-processing" in agent.skills
        assert "data-analysis" in agent.skills

    def test_agent_skills_invalid_type(self):
        """Test that invalid skills type raises error."""
        from unittest.mock import MagicMock

        from smolagents import CodeAgent

        mock_model = MagicMock()
        with pytest.raises(TypeError, match="skills must be"):
            CodeAgent(tools=[], model=mock_model, skills="invalid")

    def test_tool_calling_agent_with_skills(self, test_skill_dir):
        """Test that ToolCallingAgent also supports skills."""
        from unittest.mock import MagicMock

        from smolagents import ToolCallingAgent

        mock_model = MagicMock()
        skills = SkillCollection.from_directory(test_skill_dir)
        agent = ToolCallingAgent(tools=[], model=mock_model, skills=skills)

        assert len(agent.skills) == 2

        # System prompt should contain skills section
        system_prompt = agent.system_prompt
        assert "<available_skills>" in system_prompt

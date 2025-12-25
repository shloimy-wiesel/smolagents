"""
Example: Using Skills with smolagents

This example demonstrates how to use skills to provide specialized guidance
to your agent. Skills are directory-based packages containing instructions
that help the agent approach specific types of tasks.

Skills are:
- Opt-in by default (disabled unless explicitly enabled)
- Progressively loaded (only metadata at discovery, full content when activated)
- Composable with tools and other agent capabilities

To run this example, create a skill directory with a SKILL.md file, then
run this script pointing to that directory.
"""

from pathlib import Path
import tempfile

from smolagents import CodeAgent, InferenceClientModel, tool


# Define a simple tool
@tool
def analyze_code(code: str) -> str:
    """Analyze code and return metrics.

    Args:
        code: The code to analyze.

    Returns:
        A string with analysis results.
    """
    lines = code.strip().split("\n")
    return f"Lines of code: {len(lines)}, Characters: {len(code)}"


def create_sample_skill(directory: Path):
    """Create a sample code-review skill for demonstration."""
    skill_dir = directory / "code-review"
    skill_dir.mkdir(exist_ok=True)

    (skill_dir / "SKILL.md").write_text("""---
name: code-review
description: Provides comprehensive guidance for reviewing Python code for quality, security, and best practices.
version: "1.0.0"
tags:
  - code
  - review
  - python
  - quality
---

# Code Review Skill

When reviewing Python code, follow this structured approach:

## 1. Security Review
- Check for hardcoded credentials or API keys
- Look for SQL injection vulnerabilities
- Verify input validation and sanitization
- Review file operations for path traversal risks

## 2. Code Quality
- Verify proper error handling with specific exceptions
- Check for resource cleanup (files, connections)
- Look for potential memory leaks
- Ensure functions are not too long (< 50 lines recommended)

## 3. Style and Conventions
- Verify PEP 8 compliance
- Check docstring completeness
- Ensure meaningful variable names
- Look for consistent formatting

## 4. Performance
- Identify N+1 query patterns
- Check for unnecessary loops
- Look for opportunities to use comprehensions
- Verify appropriate data structures

## Response Format
After reviewing code, structure your response as:
1. **Summary**: Overall assessment
2. **Issues Found**: List with severity (Critical/Major/Minor)
3. **Recommendations**: Specific improvements
""")

    return skill_dir


def main():
    # Create a temporary directory for our sample skill
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / "skills"
        skills_dir.mkdir()

        # Create the sample skill
        skill_path = create_sample_skill(skills_dir)
        print(f"Created sample skill at: {skill_path}")

        # Initialize the model
        # Note: Requires HF_TOKEN environment variable or huggingface-cli login
        try:
            model = InferenceClientModel()
        except Exception as e:
            print(f"Could not initialize InferenceClientModel: {e}")
            print("Using a mock model for demonstration...")

            # Mock model for demonstration
            class MockModel:
                model_id = "mock-model"

                def generate(self, messages, **kwargs):
                    from smolagents.models import ChatMessage

                    return ChatMessage(
                        role="assistant",
                        content=(
                            "Thought: I will analyze the code using my code review skill.\n"
                            "<code>\n"
                            "result = analyze_code('def hello(): pass')\n"
                            "final_answer(result)\n"
                            "</code>"
                        ),
                    )

            model = MockModel()

        # Create agent WITHOUT skills (default behavior)
        print("\n--- Creating agent without skills (default) ---")
        agent_no_skills = CodeAgent(
            tools=[analyze_code],
            model=model,
        )
        print(f"Skills enabled: {agent_no_skills.skills_enabled}")
        print(f"Number of skills: {len(agent_no_skills.skill_registry.skills)}")

        # Create agent WITH skills enabled
        print("\n--- Creating agent with skills enabled ---")
        agent_with_skills = CodeAgent(
            tools=[analyze_code],
            model=model,
            skills_enabled=True,  # Opt-in to enable skills
            skills_paths=[skill_path],  # Load skills from this path
        )
        print(f"Skills enabled: {agent_with_skills.skills_enabled}")
        print(f"Number of skills: {len(agent_with_skills.skill_registry.skills)}")

        # List available skills
        print("\n--- Available Skills ---")
        for skill_info in agent_with_skills.list_skills():
            print(f"  - {skill_info['name']}: {skill_info['description'][:60]}...")
            print(f"    Enabled: {skill_info['enabled']}, Activated: {skill_info['is_activated']}")

        # The skills are included in the system prompt
        print("\n--- Skills in System Prompt ---")
        if "Available Skills" in agent_with_skills.system_prompt:
            print("✓ Skills section found in system prompt")
            # Find and print the skills section
            lines = agent_with_skills.system_prompt.split("\n")
            in_skills_section = False
            for line in lines:
                if "## Available Skills" in line:
                    in_skills_section = True
                if in_skills_section:
                    print(f"  {line}")
                    if line.strip() == "" and in_skills_section:
                        break
        else:
            print("✗ No skills section in system prompt")

        # Activate a skill to load full instructions
        print("\n--- Activating Skill ---")
        instructions = agent_with_skills.activate_skill("code-review")
        print(f"Loaded {len(instructions)} characters of instructions")
        print("First 200 chars:")
        print(instructions[:200])

        # Run the agent (with skills context available)
        print("\n--- Running Agent with Skills ---")
        try:
            result = agent_with_skills.run("Review this code: def hello(): pass")
            print(f"Result: {result}")
        except Exception as e:
            print(f"Note: Could not run agent (likely no API key): {type(e).__name__}")
            print("The skills functionality is working - the system prompt includes skills guidance.")

        # Demonstrate skill management
        print("\n--- Skill Management ---")
        agent_with_skills.disable_skill("code-review")
        print(f"After disable: enabled={agent_with_skills.skill_registry.get_skill('code-review').enabled}")

        agent_with_skills.enable_skill("code-review")
        print(f"After enable: enabled={agent_with_skills.skill_registry.get_skill('code-review').enabled}")


if __name__ == "__main__":
    main()

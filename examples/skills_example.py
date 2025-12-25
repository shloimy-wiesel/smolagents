#!/usr/bin/env python
"""
Example demonstrating Claude Skills support in smolagents.

This example shows how to:
1. Create skills with associated tools
2. Register skills with agents (disabled by default)
3. Enable/disable skills dynamically
4. Use skills to organize and manage agent capabilities
"""

from smolagents import CodeAgent, InferenceClientModel, Skill, tool


# Define tools that belong to different skills
@tool
def search_web(query: str) -> str:
    """
    Search the web for information.

    Args:
        query: The search query
    """
    # Mock implementation
    return f"Web search results for: {query}"


@tool
def browse_page(url: str) -> str:
    """
    Browse a webpage and extract content.

    Args:
        url: The URL to browse
    """
    # Mock implementation
    return f"Content from {url}"


@tool
def read_csv(filepath: str) -> str:
    """
    Read and analyze a CSV file.

    Args:
        filepath: Path to the CSV file
    """
    # Mock implementation
    return f"Data from {filepath}"


@tool
def plot_data(data: str) -> str:
    """
    Create a plot from data.

    Args:
        data: The data to plot
    """
    # Mock implementation
    return f"Plot created from: {data}"


# Create skills to organize tools by capability
web_skill = Skill(
    name="web_browsing",
    description="Capability to search and browse web content",
    tools=[search_web, browse_page],
    category="web",
    enabled=False,  # Disabled by default (opt-in)
)

data_skill = Skill(
    name="data_analysis",
    description="Capability to analyze and visualize data",
    tools=[read_csv, plot_data],
    category="data",
    enabled=False,  # Disabled by default (opt-in)
)

print("=== Claude Skills Example ===\n")

# Example 1: Agent without any skills
print("1. Create agent without skills")
model = InferenceClientModel()
agent = CodeAgent(tools=[], model=model, verbosity_level=0)
print(f"   Agent tools: {list(agent.tools.keys())}")
print(f"   Skills registered: {len(agent.skill_registry)}")
print()

# Example 2: Agent with skills registered but disabled
print("2. Agent with skills registered (disabled by default)")
agent = CodeAgent(tools=[], model=model, skills=[web_skill, data_skill], verbosity_level=0)
print(f"   Agent tools: {list(agent.tools.keys())}")
print(f"   Skills registered: {len(agent.skill_registry)}")
print(f"   Skills enabled: {len(agent.skill_registry.get_enabled_skills())}")
print()

# Example 3: Enable a skill to add its tools
print("3. Enable web_browsing skill")
# Create a fresh skill with enabled=True
web_skill_enabled = Skill(
    name="web_browsing",
    description="Capability to search and browse web content",
    tools=[search_web, browse_page],
    category="web",
    enabled=True,  # Enable at creation
)
agent = CodeAgent(tools=[], model=model, skills=[web_skill_enabled], verbosity_level=0)
print(f"   Agent tools: {list(agent.tools.keys())}")
print(f"   Skills enabled: {len(agent.skill_registry.get_enabled_skills())}")
print()

# Example 4: Use skill registry to manage skills dynamically
print("4. Skill registry management")
agent = CodeAgent(tools=[], model=model, skills=[web_skill, data_skill], verbosity_level=0)
print(f"   Initial tools: {list(agent.tools.keys())}")

# Enable a skill via registry (note: this won't add tools to existing agent)
print(f"   Registered skills: {agent.skill_registry.list_skills()}")
print(f"   Enabled skills: {[s.name for s in agent.skill_registry.get_enabled_skills()]}")
print()

print("=== Key Points ===")
print("✓ Skills are disabled by default (opt-in design)")
print("✓ Skills organize tools by capability/domain")
print("✓ Skills can be enabled at creation: Skill(..., enabled=True)")
print("✓ Backward compatible: agents work with or without skills")
print("✓ Skills provide a higher-level abstraction over tools")

# Note: To actually enable a skill and have its tools available,
# you need to pass it with enabled=True when creating the agent,
# or create a new agent after enabling skills via the registry.

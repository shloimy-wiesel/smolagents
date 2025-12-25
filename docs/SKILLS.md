# Skills Support

smolagents now supports **Skills** - a higher-level abstraction for organizing and managing agent capabilities.

## What are Skills?

Skills are composable capability packages that:
- Bundle related tools together by domain (e.g., web, data, media)
- Can be enabled or disabled independently
- Are **opt-in by default** (disabled unless explicitly enabled)
- Provide a cleaner way to organize agent capabilities

## Key Concepts

### Skill
A `Skill` is a named capability that can contain multiple tools and has:
- **name**: Unique identifier
- **description**: What the skill does
- **tools**: List of tools provided by the skill
- **category**: Optional grouping (e.g., "web", "data", "media")
- **enabled**: Whether the skill is active (default: False)

### SkillRegistry
A `SkillRegistry` manages skills for an agent, providing:
- Skill registration and unregistration
- Dynamic enabling/disabling of skills
- Tool aggregation from enabled skills

## Basic Usage

### Creating Skills

```python
from smolagents import Skill, tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Create a skill (disabled by default)
web_skill = Skill(
    name="web_browsing",
    description="Web search and browsing capability",
    tools=[search_web],
    category="web",
    enabled=False  # Default
)
```

### Using Skills with Agents

```python
from smolagents import CodeAgent, InferenceClientModel

model = InferenceClientModel()

# Skills are disabled by default - no tools added
agent = CodeAgent(
    tools=[],
    model=model,
    skills=[web_skill]  # Registered but disabled
)

# To use a skill, enable it at creation
enabled_skill = Skill(
    name="web_browsing",
    description="Web search capability",
    tools=[search_web],
    enabled=True  # Enable the skill
)

agent = CodeAgent(
    tools=[],
    model=model,
    skills=[enabled_skill]
)
# Now search_web tool is available to the agent
```

### Managing Skills via Registry

```python
# Access the skill registry
agent.skill_registry.list_skills()  # List all registered skills
agent.skill_registry.get_enabled_skills()  # Get enabled skills
agent.skill_registry.get_all_tools()  # Get tools from enabled skills
```

## Design Principles

1. **Opt-in by Default**: Skills are disabled unless explicitly enabled
2. **Backward Compatible**: Agents work identically with or without skills
3. **Composable**: Skills can be mixed and matched
4. **Organized**: Tools are grouped by capability domain
5. **Dynamic**: Skills can be enabled/disabled at runtime (though tools are added during initialization)

## Example: Organizing Tools with Skills

```python
from smolagents import CodeAgent, Skill, InferenceClientModel, tool

# Define tools for different capabilities
@tool
def read_csv(filepath: str) -> str:
    """Read a CSV file."""
    return f"Data from {filepath}"

@tool
def plot_data(data: str) -> str:
    """Create a plot from data."""
    return f"Plot of {data}"

@tool
def search_web(query: str) -> str:
    """Search the web."""
    return f"Results for {query}"

# Group tools into skills by domain
data_skill = Skill(
    name="data_analysis",
    description="Data reading and visualization",
    tools=[read_csv, plot_data],
    category="data",
    enabled=True  # Enable this skill
)

web_skill = Skill(
    name="web_browsing",
    description="Web search capability",
    tools=[search_web],
    category="web",
    enabled=False  # Keep this disabled
)

# Create agent with skills
model = InferenceClientModel()
agent = CodeAgent(
    tools=[],
    model=model,
    skills=[data_skill, web_skill]
)

# Only data_skill tools are available (read_csv, plot_data)
# web_skill tools are not available (search_web) because it's disabled
```

## Migration Guide

If you're not using skills, nothing changes:

```python
# Old way - still works exactly the same
agent = CodeAgent(tools=[my_tool], model=model)

# New way with skills - opt-in
agent = CodeAgent(
    tools=[my_tool],
    model=model,
    skills=[my_skill]  # Optional
)
```

## Benefits

- **Better Organization**: Group related tools by domain
- **Cleaner Code**: Separate tool definitions from agent configuration
- **Reusability**: Share skill definitions across multiple agents
- **Flexibility**: Enable/disable entire capability domains at once
- **Extensibility**: Easy to add new skill types without changing core agent code

## API Reference

### Skill

```python
Skill(
    name: str,
    description: str,
    tools: list[BaseTool] | None = None,
    category: str | None = None,
    enabled: bool = False,
)
```

### SkillRegistry

```python
registry = SkillRegistry()
registry.register(skill)
registry.enable_skill(skill_name)
registry.disable_skill(skill_name)
registry.get_enabled_skills()
registry.get_all_tools()
```

### Agent with Skills

```python
agent = CodeAgent(
    tools=[...],
    model=model,
    skills=[skill1, skill2],  # Optional
    ...
)
```

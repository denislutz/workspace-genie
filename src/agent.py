#!/usr/bin/env python3
"""
Simple Agent Loop with LangChain + Claude

This implements a ReAct (Reason + Act) agent that:
1. Takes a task from the user
2. Thinks about what to do
3. Calls tools (search, read files, etc.)
4. Observes the results
5. Repeats until task is complete

Architecture:
    User Input → Agent Loop → Tools → Qdrant/Files → Response
                     ↑___________|
                     (loop until done)
"""

import os
import logging
import requests
from typing import Any
from dotenv import load_dotenv
import requests

# Load environment variables from .env.local (in project root)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, ".env.local"))

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create debug logger for detailed flow tracking
debug_logger = logging.getLogger("debug")
debug_logger.setLevel(logging.DEBUG)
# Add handler only if not already added
if not debug_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("🔍 DEBUG: %(asctime)s - %(message)s"))
    debug_logger.addHandler(handler)
    # Prevent debug messages from propagating to root logger
    debug_logger.propagate = False

# =============================================================================
# IMPORTS
# =============================================================================

# LangChain core - for tool decorator and message types
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# LangChain Anthropic - Claude as the reasoning LLM
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from pydantic.v1 import tools

# LangGraph - provides the agent loop (ReAct pattern)

# Your existing code - reuse the vector store and search logic
from layer_config import classify_layer, load_default_layer_map
import vector_store_s
import workspace_content_s


# =============================================================================
# CONFIGURATION
# =============================================================================

# API key is loaded from .env.local (ignored by git)
# Run with: mise run agent
# Or manually: export ANTHROPIC_API_KEY="sk-ant-..." && python agent.py

# Default workspace to search (can be overridden per-query)
DEFAULT_WORKSPACE = "workspace-genie"


# =============================================================================
# SETUP: Embeddings and Vector Store Connection
# =============================================================================


# =============================================================================
# TOOLS: These are what the agent can use
# =============================================================================


@tool
def search_code(*, query: str) -> str:
    """
    Search codebase for relevant code snippets.

    This is the agent's main way to find code related to a task.
    Returns formatted code snippets with file paths.

    Args:
        query: What to search for (e.g., "authentication middleware")
        workspace: Which indexed workspace to search (uses environment if None)

    Returns:
        Formatted string with matching code snippets
    """
    workspace = os.getenv("WORKSPACE", DEFAULT_WORKSPACE)

    debug_logger.debug(
        f"🔧 TOOL EXECUTION: search_code called with query='{query}', workspace='{workspace}'"
    )
    logger.info(f"🔍 Searching codebase for: '{query}' in workspace: '{workspace}'")

    try:
        debug_logger.debug(f"🔧 VECTOR STORE: Attempting to connect to vector store...")
        vectorstore = vector_store_s.store_for_workspace(workspace)
        debug_logger.debug("✅ VECTOR STORE: Connection established")
        logger.info("✅ Vector store connected")

        debug_logger.debug(
            f"🔧 VECTOR SEARCH: Executing similarity search with query='{query}'"
        )
        results = vectorstore.similarity_search(query)
        debug_logger.debug(f"✅ VECTOR SEARCH: Found {len(results)} results")
        logger.info(f"✅ Found {len(results)} results")

        if not results:
            debug_logger.debug(
                "⚠️ VECTOR SEARCH: No results found, workspace may not be indexed"
            )
            return f"No results found for query: '{query}'. The workspace may not be indexed yet."

        debug_logger.debug("🔧 FORMATTING: Processing search results for output...")
        formatted_results = workspace_content_s.format_results(results)
        debug_logger.debug(
            f"✅ FORMATTING: Processed {len(formatted_results)} formatted results"
        )
        logger.info(f"✅ Formatted {len(formatted_results)} results")

        final_output = "\n".join([str(result) for result in formatted_results])
        debug_logger.debug(
            f"🔧 TOOL COMPLETE: search_code returning {len(final_output)} characters"
        )

        return final_output

    except ConnectionRefusedError as e:
        debug_logger.debug(f"❌ VECTOR STORE: Connection refused - {e}")
        logger.error(f"❌ Qdrant connection refused: {e}")
        return "ERROR: Qdrant vector database is not running. Start it with: mise run qdrant-start"
    except Exception as e:
        debug_logger.debug(f"❌ TOOL ERROR: search_code failed - {e}")
        logger.error(f"❌ Search failed: {e}")
        return f"Search error: {str(e)}. Make sure Qdrant is running and the workspace is indexed."


@tool
def find_patterns(*, task: str, current_file: str = "") -> dict[str, Any]:
    """
    Find code patterns in the same architectural layer.

    When implementing something new, this finds similar code
    in the same "layer" (services, components, utils, etc.)
    so you can follow existing patterns.

    Args:
        task: What you're trying to implement
        current_file: The file you're working in (for layer detection)
        workspace: Which workspace to search (uses environment if None)

    Returns:
        Dict with:
        - current_layer: detected layer of current_file
        - similar_patterns: list of {file, snippet} dicts

    Hint: Look at find_similar_patterns() in mcp_workspace.py
    """
    workspace = os.getenv("WORKSPACE", DEFAULT_WORKSPACE)

    debug_logger.debug(
        f"🔧 TOOL EXECUTION: find_patterns called with task='{task}', current_file='{current_file}', workspace='{workspace}'"
    )
    logger.info(f"🎯 Finding patterns for task: '{task}' in file: '{current_file}'")

    try:
        debug_logger.debug("🔧 LAYER DETECTION: Classifying architectural layer...")
        # Reuse your existing layer detection
        layer = classify_layer(current_file, load_default_layer_map())
        debug_logger.debug(f"✅ LAYER DETECTION: Classified as '{layer}'")
        logger.info(f"✅ Detected layer: '{layer}'")

        debug_logger.debug(
            f"🔧 PATTERN SEARCH: Searching within layer '{layer}' for task='{task}'"
        )
        # Search within that layer only
        results = vector_store_s.search_with_filter(
            workspace=workspace, query=task, filter=layer, k=8
        )
        debug_logger.debug(f"✅ PATTERN SEARCH: Found {len(results)} patterns in layer")
        logger.info(f"✅ Found {len(results)} patterns in layer")

        debug_logger.debug(
            "🔧 PATTERN PROCESSING: Extracting file snippets and metadata..."
        )
        similar_patters = [
            {
                "file": r.metadata.get("source", "unknown"),
                "snippet": r.page_content[:200],
            }
            for r in results
        ]

        result = {
            "current_layer": layer,
            "similar_patterns": similar_patters,
            "usage_hint": f"Follow the {layer} layer conventions shown above",
        }
        debug_logger.debug(
            f"✅ PATTERN PROCESSING: Created result with {len(similar_patters)} patterns"
        )
        logger.info("✅ Pattern analysis completed")

        return result

    except FileNotFoundError as e:
        debug_logger.debug(f"❌ LAYER CONFIG: File not found - {e}")
        logger.error(f"❌ Layer config file not found: {e}")
        return {"error": "Layer configuration file missing. Check default_layers.json"}
    except ConnectionRefusedError as e:
        debug_logger.debug(f"❌ VECTOR STORE: Connection refused - {e}")
        logger.error(f"❌ Qdrant connection refused: {e}")
        return {
            "error": "Qdrant vector database is not running. Start it with: mise run qdrant-start"
        }
    except Exception as e:
        debug_logger.debug(f"❌ TOOL ERROR: find_patterns failed - {e}")
        logger.error(f"❌ Pattern finding failed: {e}")
        return {
            "error": f"Pattern analysis failed: {str(e)}. Make sure Qdrant is running and workspace is indexed."
        }


@tool
def read_file(filepath: str) -> str:
    """
    Read the contents of a file.

    After searching, the agent might want to read
    full file contents to understand the code better.

    Args:
        filepath: Path to file (relative to workspace or absolute)

    Returns:
        File contents as string, or error message
    """
    debug_logger.debug(
        f"🔧 TOOL EXECUTION: read_file called with filepath='{filepath}'"
    )
    logger.info(f"📖 Reading file: '{filepath}'")

    try:
        debug_logger.debug(
            f"🔧 FILE ACCESS: Checking file existence and size for '{filepath}'"
        )
        # try to read file with given path
        is_not_too_large = os.path.getsize(filepath) < 100000  # 100KB
        debug_logger.debug(
            f"🔧 FILE ACCESS: Size check passed ({is_not_too_large}), checking existence..."
        )

        if os.path.exists(filepath) and is_not_too_large:
            debug_logger.debug(
                f"🔧 FILE READING: Opening file '{filepath}' for reading..."
            )
            with open(filepath, "r") as f:
                content = f.read()
                debug_logger.debug(
                    f"✅ FILE READING: Successfully read {len(content)} characters"
                )
                logger.info(f"✅ File read successfully ({len(content)} chars)")
                return content
        else:
            debug_logger.debug(
                f"⚠️ FILE ACCESS: File not found or too large: {filepath}"
            )
            logger.warning(f"⚠️ File not found or too large: {filepath}")
            return f"File not found: {filepath}"
    except Exception as e:
        debug_logger.debug(f"❌ TOOL ERROR: read_file failed - {e}")
        logger.error(f"❌ Error reading file: {e}")
        return f"Error reading file: {e}"


@tool
def list_files(directory: str = ".") -> str:
    """
    List files in a directory.

    Helps the agent explore the codebase structure.

    Args:
        directory: Path to directory to list

    Returns:
        Formatted list of files and subdirectories
    """
    debug_logger.debug(
        f"🔧 TOOL EXECUTION: list_files called with directory='{directory}'"
    )
    logger.info(f"📁 Listing files in directory: '{directory}'")

    try:
        debug_logger.debug(
            f"🔧 DIRECTORY ACCESS: Checking if directory '{directory}' exists..."
        )
        if not os.path.exists(directory):
            debug_logger.debug(f"⚠️ DIRECTORY ACCESS: Directory not found: {directory}")
            logger.warning(f"⚠️ Directory not found: {directory}")
            return f"Directory not found: {directory}"

        debug_logger.debug(
            f"🔧 DIRECTORY LISTING: Getting contents of '{directory}'..."
        )
        files = os.listdir(directory)
        debug_logger.debug(f"✅ DIRECTORY LISTING: Found {len(files)} items")
        logger.info(f"✅ Found {len(files)} items in directory")

        result = "\n".join(files)
        debug_logger.debug(
            f"🔧 TOOL COMPLETE: list_files returning {len(result)} characters"
        )
        logger.info("✅ Directory listing completed")

        return result

    except Exception as e:
        debug_logger.debug(f"❌ TOOL ERROR: list_files failed - {e}")
        logger.error(f"❌ Error listing directory: {e}")
        return f"Error listing directory: {e}"


# =============================================================================
# AGENT SETUP
# =============================================================================


# Function that tries open source first, falls back to Claude
def choose_and_create_llm():
    """Prompt user to select workspace and model, then create LLM."""
    logger.info("🤖 Creating LLM...")

    # Check what model to use (you can set this via environment variable)
    model_choice = os.getenv("LLM_MODEL")  # Default to Ollama
    logger.info(f"🔧 Using model: {model_choice}")

    # Prompt for workspace selection if not set
    workspace = os.getenv("WORKSPACE")
    if not workspace or workspace == DEFAULT_WORKSPACE:
        print(f"\n📁 Current workspace: {workspace}")
        print("💡 To work on a specific workspace, set WORKSPACE environment variable:")
        print("   export WORKSPACE=your-workspace-name")
        print("🔍 Available workspaces (run 'mise run list-collections' to see all):")
        print("   - Default: workspace-genie (current)")
        print("   - Add more by indexing: mise run index your-workspace-path")
        print()

    logger.info(f"🔧 Selected workspace: {workspace}")
    try:
        if model_choice == "ollama":
            logger.info("🔧 Initializing Ollama chat model...")
            llm = ChatOllama(
                model="llama3.2:3b"
            )  # Use lightweight llama3.2:3b for speed
            logger.info("✅ Ollama chat model initialized successfully")
            return llm

        elif model_choice == "ollama-coder":
            logger.info("🔧 Initializing Ollama coding model...")
            llm = ChatOllama(
                model="deepseek-coder-v2:16b"
            )  # Use deepseek for specialized coding
            logger.info("✅ Ollama coding model initialized successfully")
            return llm

        elif model_choice == "ollama-large":
            logger.info("🔧 Initializing larger Ollama model...")
            llm = ChatOllama(model="llama2:13b")  # More capable local model
            logger.info("✅ Ollama large model initialized successfully")
            return llm

        elif model_choice == "claude":
            logger.info("🔧 Initializing Claude model...")
            llm = ChatAnthropic(
                model_name="claude-sonnet-4-20250514", timeout=120, stop=None
            )
            logger.info("✅ Claude model initialized successfully")
            return llm

        else:
            logger.error(f"❌ Unknown model choice: {model_choice}")
            logger.info(
                "🔧 Available options: ollama, ollama-coder, ollama-large, claude"
            )
            raise ValueError(f"Unknown model: {model_choice}")

    except Exception as e:
        logger.error(f"❌ Failed to create LLM ({model_choice}): {e}")

        # Fallback to Claude if open source fails
        if model_choice != "claude":
            logger.info("🔄 Falling back to Claude...")
            try:
                llm = ChatAnthropic(
                    model_name="claude-sonnet-4-20250514", timeout=120, stop=None
                )
                logger.info("✅ Claude fallback initialized successfully")
                return llm
            except Exception as fallback_error:
                logger.error(f"❌ Claude fallback also failed: {fallback_error}")
                raise
        else:
            raise


def create_agent_all():
    """
    Create the ReAct agent with LLM and tools.
    """
    logger.info("🏗️  Creating agent...")

    # Initialize LLM using our new function
    logger.info("🔧 Step 1: Creating LLM...")
    llm = choose_and_create_llm()
    logger.info("✅ LLM created successfully")

    # Collect tools with the selected workspace
    logger.info("🔧 Step 2: Collecting tools...")
    tools = [
        search_code,
        find_patterns,
        read_file,
        list_files,
    ]
    logger.info(f"✅ Collected {len(tools)} tools: {[tool.name for tool in tools]}")

    # Create the agent with ReAct pattern
    logger.info("🔧 Step 3: Creating ReAct agent...")
    agent = create_agent(llm, tools)
    logger.info("✅ Agent created successfully")

    return agent


# =============================================================================
# AGENT LOOP: Run the agent
# =============================================================================


def run_agent(agent, task: str) -> str:
    """
    Run the agent on a task and return the final response.
    """
    debug_logger.debug(f"🚀 AGENT EXECUTION: Starting run_agent with task='{task}'")
    logger.info(f"🚀 Running agent on task: {task}")

    # Create input with task as a user message
    debug_logger.debug("🔧 INPUT: Creating message for agent...")
    logger.info("🔧 Creating input message...")
    inputs = {"messages": [HumanMessage(content=task)]}
    debug_logger.debug(f"✅ INPUT: Created message with {len(task)} characters")

    # Run the agent (this runs the full think→act→observe loop)
    debug_logger.debug("🔄 AGENT LOOP: Starting agent execution (ReAct pattern)...")
    logger.info("🔄 Starting agent execution (this may take a moment)...")
    try:
        debug_logger.debug("🔧 AGENT INVOKE: Calling agent.invoke()...")
        result = agent.invoke(inputs)
        debug_logger.debug("✅ AGENT INVOKE: Agent execution completed successfully")
        logger.info("✅ Agent execution completed")

        # Extract the final message content
        debug_logger.debug("🔧 RESPONSE: Extracting final message from result...")
        logger.info("🔧 Extracting final response...")
        final_response = result["messages"][-1].content
        debug_logger.debug(f"✅ RESPONSE: Extracted {len(final_response)} characters")
        logger.info("✅ Response extracted successfully")

        return final_response

    except Exception as e:
        debug_logger.debug(f"❌ AGENT ERROR: Execution failed - {e}")
        logger.error(f"❌ Agent execution failed: {e}")
        return f"Error: {str(e)}"


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main():
    """
    Simple CLI loop for interacting with the agent.
    """
    logger.info("🎯 Starting Workspace Genie Agent CLI...")

    # Check if debug logging is enabled
    debug_enabled = os.getenv("DEBUG_AGENT", "false").lower() == "true"
    if debug_enabled:
        debug_logger.info("🔍 DEBUG MODE: Enabled - detailed flow logging active")
        print("🔍 DEBUG MODE: Enabled - detailed flow logging active")
    else:
        print("💡 Enable debug logging with: export DEBUG_AGENT=true")

    # Check required services
    logger.info("🔍 Checking required services...")

    # Check Ollama
    try:

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        logger.info("✅ Ollama server is running")
    except:
        logger.warning("⚠️ Ollama server not running. Start it with: ollama serve &")
        print("⚠️ WARNING: Ollama server is not running. Start it with: ollama serve &")

    # Check Qdrant
    try:

        response = requests.get("http://localhost:6333/collections", timeout=2)
        logger.info("✅ Qdrant server is running")
    except:
        logger.warning(
            "⚠️ Qdrant server not running. Start it with: mise run qdrant-start"
        )
        print(
            "⚠️ WARNING: Qdrant server is not running. Start it with: mise run qdrant-start"
        )
        print(
            "💡 Without Qdrant, the agent cannot search code. Start it to enable full functionality."
        )

    print("=" * 60)
    print("Workspace Genie Agent")
    print("=" * 60)
    print("Commands: 'quit' to exit, 'help' for usage")
    print()

    # Create agent once
    logger.info("🔧 Creating agent (this may take a moment on first run)...")
    try:
        # Get workspace from environment or use default
        agent = create_agent_all()
        logger.info("✅ Agent created and ready!")
    except Exception as e:
        logger.error(f"❌ Failed to create agent: {e}")
        print(f"ERROR: Agent not configured. {e}")
        return

    if agent is None:
        print("ERROR: Agent not configured. Fill in create_agent() first.")
        return

    logger.info("🎮 Starting interactive loop...")
    while True:
        try:
            # Get user input
            task = input("> ").strip()

            # Handle commands
            if task.lower() in ("quit", "exit", "q"):
                logger.info("👋 Goodbye!")
                print("Goodbye!")
                break

            if task.lower() == "help":
                print(
                    """
Usage:
  Just type what you want to do, e.g.:
  > find authentication code in the codebase
  > how does the payment service work?
  > show me examples of error handling

Setup Commands:
  - Start Ollama: ollama serve &
  - Start Qdrant: mise run qdrant-start
  - Index workspace: mise run index

Debug Options:
  - Enable debug logging: export DEBUG_AGENT=true

The agent will search the codebase and answer based on what it finds.
                """
                )
                continue

            if not task:
                continue

            # Run the agent
            print("\nThinking...\n")
            logger.info(f"🤖 Processing user request: {task}")
            response = run_agent(agent, task)
            print(response)
            print()

        except KeyboardInterrupt:
            logger.info("👋 User interrupted, exiting...")
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

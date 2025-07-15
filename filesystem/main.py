import os
import json
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from typing import TypedDict, List

load_dotenv()

# Read config file
with open("config.json") as f:
    config = json.load(f)

# Extract MCP config
mcp_servers_config = config["mcp"]["servers"]

# State for workflow
class WorkflowState(TypedDict, total=False):
    file_list: List[str]
    file_content: str
    updated_content: str

# Node: List files using MCP
async def list_files_node(state: WorkflowState, tools):
    print("\n📂 Listing files in MCP folder...")

    list_tool = next((t for t in tools if t.name == "list_directory"), None)
    if not list_tool:
        raise ValueError("list_directory tool not found in Filesystem MCP tools.")

    target_file = "C:/Users/ayanr/OneDrive/Desktop/buisnessfetch/data"
    result = await list_tool.ainvoke({"path": target_file})
    print(f"✅ Files found: {result}")
    state["file_list"] = result
    return state

# Node: Read first file
async def read_file_node(state: WorkflowState, tools):
    print("\n📄 Reading the first file...")

    read_tool = next((t for t in tools if t.name == "read_file"), None)
    if not read_tool:
        raise ValueError("read_file tool not found in Filesystem MCP tools.")

    target_file = "C:/Users/ayanr/OneDrive/Desktop/buisnessfetch/data/example.txt"  # relative path only

    result = await read_tool.ainvoke({"path": target_file})
    print(f"✅ Contents of {target_file}:\n{result}")

    state["file_content"] = result
    return state

# Node: LLM edits file content
async def edit_with_llm_node(state: WorkflowState, llm):
    print("\n🤖 Editing content with LLM...")

    prompt = (
        "Please correct grammar and spelling in the following text:\n\n"
        f"{state['file_content']}\n\n"
        "Return only the corrected text."
    )

    response = await llm.ainvoke(prompt)
    edited_content = response.content

    print(f"✅ LLM Edited Content:\n{edited_content}")

    state["updated_content"] = edited_content
    return state

# Node: Write corrected content back
async def write_file_node(state: WorkflowState, tools):
    print("\n💾 Writing updated content to file...")

    write_tool = next((t for t in tools if t.name == "write_file"), None)
    if not write_tool:
        raise ValueError("write_file tool not found in Filesystem MCP tools.")

    target_file = "C:/Users/ayanr/OneDrive/Desktop/buisnessfetch/data/example.txt" # relative path only

    await write_tool.ainvoke({
        "path": target_file,
        "content": state["updated_content"]
    })

    print(f"✅ File '{target_file}' successfully updated.")
    return state

# Main workflow
async def run_workflow():
    model = ChatOpenAI(
        model_name="llama3-8b-8192",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1"
    )

    client = MultiServerMCPClient(mcp_servers_config)
    tools = await client.get_tools()

    print(f"🛠️ Available tools from MCP: {[t.name for t in tools]}")

    state: WorkflowState = {}

    state = await list_files_node(state, tools)
    state = await read_file_node(state, tools)
    state = await edit_with_llm_node(state, model)
    state = await write_file_node(state, tools)

    print(f"\n🏁 Final state:\n{state}")

if __name__ == "__main__":
    asyncio.run(run_workflow())

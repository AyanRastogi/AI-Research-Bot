import asyncio
from langgraph.graph import StateGraph, START, END
from langchain_mcp_adapters.tools.filesystem import create_filesystem_tool
from mcp.client.stdio import stdio_client

async def main():
    async with stdio_client(
        ["node", "dist/index.js", "C:/Users/ayanr/OneDrive/Desktop/notesagent/notes"],
        cwd="C:/Users/ayanr/OneDrive/Desktop/mcpservers/servers/src/filesystem"
    ) as (read, write):
        tool = await create_filesystem_tool(read, write)
        result = await tool.invoke({"path": "."})
        print(result)

asyncio.run(main())

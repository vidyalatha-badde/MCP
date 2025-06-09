"""
Simple chat example using MCPAgent with built-in conversation memory.

This example demonstrates how to use the MCPAgent with its built-in
conversation history capabilities for better contextual interactions.

Special thanks to https://github.com/microsoft/playwright-mcp for the server.
"""

import asyncio
import torch
import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from mcp_use.adapters.langchain_adapter import LangChainAdapter
#from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv

from mcp_use import MCPAgent, MCPClient


async def run_memory_chat():
    """Run a chat using MCPAgent's built-in conversation memory."""
    
    # Config file path - change this to your config file
    config_file = "tools.json"

    print("Initializing chat...")

    # Create MCP client and agent with memory enabled
    client = MCPClient.from_config_file(config_file)
    #login(token=HF_TOKEN)
    #llm = ChatGroq(model="llama-3.1-8b-instant")
#     llm = HuggingFacePipeline.from_model_id(
#     model_id="Qwen/Qwen2.5-1.5B-Instruct",
#     task="text-generation",
#     device_map="xpu",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     ),
    
# )
#     #llm=llm.to('xpu')
    
#     chat_model = ChatHuggingFace(llm=llm)
    
    model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct"
    llm = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    device_map="xpu",
    task="text-generation",
    pipeline_kwargs=dict(
                        max_new_tokens=512,
                        do_sample=False,
                        repetition_penalty=1.03,
                    ),
    ) 
#     llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
#     task="text-generation",
#     max_new_tokens=512,
#     temperature=0.7
# )
        # Tool names for instruction
    #adapter = LangChainAdapter()
    # Get LangChain tools with a single line
    #tools = await adapter.create_tools(client)
    
    # Create a custom LangChain agent
    #llm_with_tools = llm.bind_tools(tools)
    #result = await llm_with_tools.ainvoke("What tools do you have available ? ")
    #print(result)
    chat_model = ChatHuggingFace(llm=llm)

    # Create agent with memory_enabled=True
    agent = MCPAgent(
        llm=chat_model,
        client=client,
        max_steps=15,
        memory_enabled=False,  # Enable built-in conversation memory
    )

    print("\n===== Interactive MCP Chat =====")
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("==================================\n")

    try:
        # Main chat loop
        while True:
            # Get user input
            user_input = input("\nYou: ")

            # Check for exit command
            if user_input.lower() in ["exit", "quit"]:
                print("Ending conversation...")
                break

            # Check for clear history command
            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue

            # Get response from agent
            print("\nAssistant: ", end="", flush=True)

            try:
                # Run the agent with the user input (memory handling is automatic)
                response = await agent.run(user_input)
                print(response)

            except Exception as e:
                print(f"\nError: {e}")

    finally:
        # Clean up
        if client and client.sessions:
            await client.close_all_sessions()


if __name__ == "__main__":
    asyncio.run(run_memory_chat())

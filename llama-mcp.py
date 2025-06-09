"""
Simple chat example using MCPAgent with built-in conversation memory.

This example demonstrates how to use the MCPAgent with its built-in
conversation history capabilities for better contextual interactions.

Special thanks to https://github.com/microsoft/playwright-mcp for the server.
"""

import asyncio
import os
from langchain_community.chat_models import ChatLlamaCpp
from dotenv import load_dotenv

from mcp_use import MCPAgent, MCPClient


async def run_memory_chat():
    """Run a chat using MCPAgent's built-in conversation memory."""
    #HF_TOKEN = "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"
    # Config file path - change this to your config file
    config_file = "duckduckgo.json"

    print("Initializing chat...")

    # Create MCP client and agent with memory enabled
    client = MCPClient.from_config_file(config_file)

    
    
    local_model = "C:/Users/vbaddex/.cache/huggingface/hub/models--NousResearch--Nous-Hermes-2-Mistral-7B-DPO-GGUF/snapshots/eb85cf06e8663157611e8ee472e61b43f50ee49f/Nous-Hermes-2-Mistral-7B-DPO.Q2_K.gguf"
    llm = ChatLlamaCpp(
    temperature=0.5,
    model_path=local_model,
    n_ctx=10000,
    n_gpu_layers=-1,
    n_batch=300,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    max_tokens=512,
    repeat_penalty=1.5,
    top_p=0.5,
 )
    msg = """You are a helpful AI assistant.
        You have access to the following tools:

        {tool_descriptions}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of the available tools
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
        """
    prompt = """You are an useful AI assistant. You must answer the user queries by invoking one of the necessary.
    Give precise and accurate answers
    """
    # Create agent with memory_enabled=True
    agent = MCPAgent(
        llm=llm,
        client=client,
        #max_steps=15,
        system_prompt=prompt,
        system_prompt_template=msg,
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
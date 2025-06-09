# MCP

# Required packages
```
Requires-python = ">=3.12"
Dependencies: 
    "accelerate"
    "langchain-huggingface==0.1.2"
    "langchain-community"
    "llama-cpp-python"
    "mcp-use"
```
## Installation steps for ChatLlamaCpp chat model for GPU BE
```
conda create -n llama-cpp-env
conda activate llama-cpp-env
@call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 --force
set CMAKE_GENERATOR=Ninja
set CMAKE_C_COMPILER=cl
set CMAKE_CXX_COMPILER=icx
set CXX=icx
set CC=cl
set CMAKE_ARGS="-DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_CXX_COMPILER=icx -DCMAKE_C_COMPILER=cl"
pip install langchain-community llama-cpp-python
```
### Download the llama model
`huggingface-cli download NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF Nous-Hermes-2-Mistral-7B-DPO.Q2_K.ggu`

#### To run
`python <filename.py>`

    

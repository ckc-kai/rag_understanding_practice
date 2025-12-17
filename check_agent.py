from llama_index.core.agent import ReActAgent
import llama_index.core
print(f"Llama Index Version: {llama_index.core.__version__}")
print(f"ReActAgent type: {type(ReActAgent)}")
try:
    print(f"Has from_tools: {hasattr(ReActAgent, 'from_tools')}")
    print(dir(ReActAgent))
except Exception as e:
    print(e)

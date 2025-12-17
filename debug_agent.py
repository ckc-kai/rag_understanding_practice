from llama_index.core.agent import ReActAgent
print(f"ReActAgent type: {type(ReActAgent)}")
print(f"Has from_tools: {hasattr(ReActAgent, 'from_tools')}")
try:
    print(dir(ReActAgent))
except:
    pass

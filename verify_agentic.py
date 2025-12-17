
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("/Users/ckc/Desktop/AI_Projects/llm"))

print("Importing advanced_rag_agentic...")
try:
    from advanced_rag_agentic import Librarian, AgentOrchestrator, AgenticMemory
    print("Import successful!")
except ImportError as e:
    traceback.print_exc()
    print(f"Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during import: {e}")
    sys.exit(1)

print("Checking classes...")
# Basic instantiation checks (mocking dependencies if needed, but just checking class existence here)
assert Librarian
assert AgentOrchestrator
assert AgenticMemory

print("Verification passed!")

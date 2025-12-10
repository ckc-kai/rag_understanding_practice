import ollama 

MODEL = "deepseek-r1:7b"
def ask(prompt):
    stream = ollama.chat(
        model = MODEL,
        messages = [
            {"role": "user", "content": prompt}
        ],
        stream = True
    )
    full_response = ""
    for part in stream:
        delta = part['message']['content']
        print(delta, end="", flush=True)
        full_response += delta
    return full_response

if __name__ == "__main__":
    print("running DeepSeek ")
    answer = ask("what is sentence-window retrieval?")
    print("answer: ", answer)
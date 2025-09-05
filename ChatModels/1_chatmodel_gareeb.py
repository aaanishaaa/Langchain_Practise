from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # chat-tuned model
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("Explain the theory of relativity in simple terms.")
print(result.content)

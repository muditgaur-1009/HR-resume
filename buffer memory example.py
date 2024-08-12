import json
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# Initialize the LLM
llm = ChatOllama(
    model="llama2",
    temperature=0.7,
    max_tokens=50  # Adjusting max_tokens to allow for longer responses
)

# Initialize BufferMemoryWindow to keep track of the conversation history
memory = ConversationBufferWindowMemory(k=20)  # Adjust 'k' to the number of past interactions to remember

# Create a prompt template
template = """You are a helpful assistant. The following is a conversation with a user:
{history}
User: {input}
Assistant: Given the user's input, perform the following tasks:
1. Expand and elaborate the query to be more understandable and expressive. Use the format: [EXPAND = <expanded_query>]
2. Identify the overall topic and format it as [TOPIC: category - subcategory].
3. Answer the question precisely and in fewer words.

Provide the response in the following format:
User = <original_input> | [EXPAND = <expanded_query> | [TOPIC: <topic>]
Bot => <answer> ] 

Examples:
- Input: "Who is the pm of India?"
  Output: User = Who is the pm of India? [TOPIC: Politics - India]
          Bot => The prime minister of India is Narendra Modi | [EXPAND = Who is the prime minister of India?] [TOPIC: Politics - India]
- Input: "what are his duties"
  Output: User = what are his duties [TOPIC: Politics - India]
          Bot => The duties of the prime minister of India include... | [EXPAND = what are the duties of the prime minister of India?] [TOPIC: Politics - India]

Provide the expanded query, labeled topic, and your answer:
"""

# Create the PromptTemplate instance
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# Create the LLMChain instance with the LLM, prompt, and memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)

# Function to append conversation history to a list
def append_history_to_list(conversation_list, user_input, response):
    expanded_query = response.split('[EXPAND = ')[1].split(']')[0]
    topic = response.split('[TOPIC: ')[1].split(']')[0]
    answer = response.split('Bot => ')[1].split(' | ')[0].strip()

    conversation_list.append({
        "instruction": "Answer the user's query",
        "input": f"{user_input}",
        "output": f"User = {user_input} | [EXPAND = {expanded_query}] [TOPIC: {topic}]\nBot => {answer}"
    })

# Function to write the conversation history to a JSON file
def append_history_to_json(filename, conversation_list):
    with open(filename, 'w') as file:
        json.dump(conversation_list, file, indent=4)

# Function to interact with the bot
def chat_with_bot(user_input, conversation_list):
    response = chain.run({"input": user_input})
    
    # Append the entry to the conversation list
    append_history_to_list(conversation_list, user_input, response)
    
    return response

# Main function to run the chat loop
if __name__ == "__main__":
    conversation_list = []
    json_file = 'conversation_history.json'
    print("Start chatting with the bot (type 'exit' to stop):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chat. Goodbye!")
            append_history_to_json(json_file, conversation_list)
            break
        response = chat_with_bot(user_input, conversation_list)
        print(f"Bot: {response}")

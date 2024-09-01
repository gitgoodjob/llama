import streamlit as st
from streamlit_chat import message
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the chat history dictionary
chat_history = {}

# Load the pre-trained LLaMA model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

def generate_bot_response(user_message):
    inputs = tokenizer(user_message, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=50, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    st.title("LLaMA Chatbot")
    st.header("Converse with me!")

    # Set up the text input field and button
    user_input = st.text_input("Type a message...")
    send_button = st.button("Send")

    # Initialize chat_id if not set
    if "chat_id" not in st.session_state:
        st.session_state["chat_id"] = 0

    # Handle the chat interaction
    if send_button:
        user_message = user_input.strip()
        if user_message:
            # Add the user's message to the chat history
            chat_id = st.session_state["chat_id"]
            chat_history[chat_id] = {"user": user_message, "bot": None}

            # Generate a response from LLaMA
            bot_response = generate_bot_response(user_message)
            chat_history[chat_id]["bot"] = bot_response

            # Update the chat history with the new message
            message.update(chat_history)

            # Increment chat_id for the next conversation
            st.session_state["chat_id"] += 1

    # Display the chat history and generated responses
    for i, (chat_id, messages) in enumerate(chat_history.items()):
        st.write(f"**Chat {i+1}**")
        st.write(f"> **User**: {messages['user']}")
        st.write(f"< **Bot**: {messages['bot']}")

if __name__ == "__main__":
    main()

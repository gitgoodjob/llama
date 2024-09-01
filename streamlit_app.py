import streamlit as st
from streamlit_chat import message
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the chat history dictionary
chat_history = {}

# Load the pre-trained OPT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
    return tokenizer, model

tokenizer, model = load_model()

def generate_bot_response(user_message):
    inputs = tokenizer(user_message, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=50, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    st.title("AI Chatbot")
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

            # Generate a response from the model
            bot_response = generate_bot_response(user_message)
            chat_history[chat_id]["bot"] = bot_response

            # Increment chat_id for the next conversation
            st.session_state["chat_id"] += 1

    # Display the chat history and generated responses
    for i, (chat_id, messages) in enumerate(chat_history.items()):
        message(messages['user'], is_user=True, key=f"user_{i}")
        if messages['bot']:
            message(messages['bot'], is_user=False, key=f"bot_{i}")

if __name__ == "__main__":
    main()

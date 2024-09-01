import streamlit as st
from streamlit_chat import message
import nltk
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import random

print("Starting the app...")

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

print("NLTK data downloaded.")

# Initialize the chat history dictionary
chat_history = {}

# Create a simple language model
def create_language_model(text, n=2):
    print("Creating language model...")
    tokens = word_tokenize(text.lower())
    train_data, padded_sents = padded_everygram_pipeline(n, [tokens])
    model = MLE(n)
    model.fit(train_data, padded_sents)
    print("Language model created.")
    return model

# Generate a response using the language model
def generate_bot_response(model, user_message, max_length=20):
    print(f"Generating response for: {user_message}")
    context = word_tokenize(user_message.lower())[-model.order+1:]
    response = []
    for _ in range(max_length):
        next_word = model.generate(1, context)
        if next_word is None:
            break
        response.append(next_word)
        context = context[1:] + [next_word]
    response_text = ' '.join(response)
    print(f"Generated response: {response_text}")
    return response_text

# Load the model (this function is cached)
@st.cache_resource
def load_model():
    print("Loading model...")
    # Sample text for training the model
    sample_text = """
    Hello! How are you? I'm an AI assistant. I can help you with various tasks.
    What would you like to know? I can provide information on many topics.
    Let me know if you have any questions. I'm here to assist you.
    """
    model = create_language_model(sample_text)
    print("Model loaded.")
    return model

print("About to load model...")
model = load_model()
print("Model loaded successfully.")

def main():
    print("Entering main function...")
    st.title("Simple AI Chatbot")
    st.header("Converse with me!")

    # Set up the text input field and button
    user_input = st.text_input("Type a message...")
    send_button = st.button("Send")

    # Initialize chat_id if not set
    if "chat_id" not in st.session_state:
        st.session_state["chat_id"] = 0

    # Handle the chat interaction
    if send_button:
        print("Send button clicked.")
        user_message = user_input.strip()
        if user_message:
            print(f"Processing user message: {user_message}")
            # Add the user's message to the chat history
            chat_id = st.session_state["chat_id"]
            chat_history[chat_id] = {"user": user_message, "bot": None}

            # Generate a response
            bot_response = generate_bot_response(model, user_message)
            chat_history[chat_id]["bot"] = bot_response

            # Increment chat_id for the next conversation
            st.session_state["chat_id"] += 1

    # Display the chat history and generated responses
    print("Displaying chat history...")
    for i, (chat_id, messages) in enumerate(chat_history.items()):
        message(messages['user'], is_user=True, key=f"user_{i}")
        if messages['bot']:
            message(messages['bot'], is_user=False, key=f"bot_{i}")

    print("Main function completed.")

if __name__ == "__main__":
    main()
    

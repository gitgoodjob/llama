import streamlit as st
from transformers import LLaMAForConditionalGeneration, LLaMATokenizer

# Load the LLaMA model and tokenizer
model = LLaMAForConditionalGeneration.from_pretrained("facebook/llama-7b")
tokenizer = LLaMATokenizer.from_pretrained("facebook/llama-7b")

# Initialize the chat history
chat_history = []

def chatbot_response(input_text):
    # Preprocess the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate a response
    outputs = model.generate(**inputs, max_length=512)

    # Convert the response to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

def app():
    st.title("LLaMA Chatbot")

    # Chat input field
    user_input = st.text_input("Type a message:", "")

    # Send button
    if st.button("Send"):
        # Add the user input to the chat history
        chat_history.append(f"User: {user_input}")

        # Get the chatbot response
        response = chatbot_response(user_input)

        # Add the chatbot response to the chat history
        chat_history.append(f"LLaMA: {response}")

    # Display the chat history
    st.write("Chat History:")
    for message in chat_history:
        st.write(message)

if __name__ == "__main__":
    app()

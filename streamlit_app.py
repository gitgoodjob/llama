import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load a smaller model with caching disabled
model_name = "t5-small"
try:
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
except ImportError as e:
    st.error(f"Error loading model: {e}")
    st.write("Please install the required backend libraries (torch, torchvision, numpy, scipy, Pillow)")
    st.write("or upgrade the transformers library to the latest version")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Check if model and tokenizer are defined
if'model' not in locals() or 'tokenizer' not in locals():
    st.error("Model and tokenizer are not defined")
else:
    # Initialize the chat history
    chat_history = []

    def chatbot_response(input_text):
        try:
            # Preprocess the input text
            inputs = tokenizer(input_text, return_tensors="pt")

            # Generate a response
            outputs = model.generate(**inputs, max_length=512)

            # Convert the response to text
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return response
        except Exception as e:
            return f"Error: {e}"

    def app():
        st.title("Chatbot")

        # Chat input field
        user_input = st.text_input("Type a message:", "")

        # Send button
        if st.button("Send"):
            # Add the user input to the chat history
            try:
                chat_history.append(f"User: {user_input}")
            except Exception as e:
                st.error(f"Error appending to chat history: {e}")

            # Get the chatbot response
            response = chatbot_response(user_input)

            # Add the chatbot response to the chat history
            try:
                chat_history.append(f"Chatbot: {response}")
            except Exception as e:
                st.error(f"Error appending to chat history: {e}")

        # Display the chat history
        st.write("Chat History:")
        try:
            for message in chat_history:
                st.write(message)
        except Exception as e:
            st.error(f"Error displaying chat history: {e}")

    if __name__ == "__main__":
        app()

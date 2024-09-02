import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load a smaller model with caching disabled
model_name = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_cache=False)

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
        return f"Error: {str(e)}"

def app():
    st.title("Chatbot")

    # Chat input field
    user_input = st.text_input("Type a message:", "")

    # Send button
    if st.button("Send"):
        # Add the user input to the chat history
        chat_history.append(f"User: {user_input}")

        # Get the chatbot response
        response = chatbot_response(user_input)

        # Add the chatbot response to the chat history
        chat_history.append(f"Chatbot: {response}")

    # Display the chat history
    st.write("Chat History:")
    for message in chat_history:
        st.write(message)

if __name__ == "__main__":
    app()

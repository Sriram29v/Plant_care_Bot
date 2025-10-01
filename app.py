import streamlit as st
from dotenv import load_dotenv  # Load .env for API key
from chatbot import PlantBot  # Import your bot
from PIL import Image  # For image handling
import os  # For debug

# Load .env early
load_dotenv()

# Debug: Check API key (remove this line after testing)
st.sidebar.write(f"API Key Loaded: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")

# Streamlit page config
st.set_page_config(page_title="🌿 Plant Care Assistant", page_icon="🌿", layout="centered")

# Initialize bot in session state
if "bot" not in st.session_state:
    st.session_state.bot = PlantBot(openai_api_key=None)  # Auto-loads from .env

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("🌿 Plant Care Assistant Bot")
st.markdown("Ask me about your plants—care tips, troubleshooting, and more!")

# Image upload section
uploaded_file = st.file_uploader("📸 Upload a photo of your plant for diagnosis", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the image (fixed: omit width to avoid deprecation error)
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Plant Photo")
    
    # Generate prompt for vision analysis
    prompt = f"Analyze this plant photo for issues like diseases, pests, or care problems. Suggest fixes based on common houseplants. Keep it helpful and concise."
    
    # Add user message (with image note)
    st.session_state.messages.append({"role": "user", "content": f"{prompt} (Photo uploaded)"})
    with st.chat_message("user"):
        st.markdown(f"{prompt} (Photo uploaded)")
    
    # Get bot response (pass image file)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing image..."):
            response = st.session_state.bot.get_response(prompt, image=uploaded_file)
        st.markdown(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history (text-only for now)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input (text-only)
if prompt := st.chat_input("What's up with your plant?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.bot.get_response(prompt)
        st.markdown(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar for tips
with st.sidebar:
    st.header("Quick Tips")
    st.markdown("""
    - **Watering**: Let soil dry between waterings.
    - **Light**: Bright, indirect for most houseplants.
    - **Pro Tip**: Upload plant pics for AI diagnosis!
    """)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
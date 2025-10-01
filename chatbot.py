import json
import os
from dotenv import load_dotenv  # Load .env file
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory  # Corrected import path
import base64
from io import BytesIO
from PIL import Image  # For image handling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle  # For saving/loading model

# Load .env early
load_dotenv()

# Your plant data (same as before)
plant_data = {
    "monstera": {
        "watering": "Every 1-2 weeks, let soil dry out",
        "light": "Bright indirect",
        "issues": {"yellow leaves": "Overwatering or low light"}
    },
    # Add more plants...
}

# Sample training data for ML classifier
symptom_data = {
    "texts": [
        "yellow leaves soggy soil",
        "yellow leaves dry soil",
        "brown tips dry air",
        "wilting leaves too much sun",
        "spots on leaves high humidity"
    ],
    "labels": [
        "overwatering",
        "underwatering",
        "low humidity",
        "sunburn",
        "fungal infection"
    ]
}

# Train and save model (run once)
def train_ml_model():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(symptom_data["texts"])
    clf = MultinomialNB().fit(X, symptom_data["labels"])
    
    # Save for reuse
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('clf.pkl', 'wb') as f:
        pickle.dump(clf, f)
    return vectorizer, clf

# Load model (if exists, else train)
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('clf.pkl', 'rb') as f:
        clf = pickle.load(f)
except FileNotFoundError:
    vectorizer, clf = train_ml_model()

class PlantBot:
    def __init__(self, openai_api_key=None):
        self.plant_data = plant_data
        self.chat_histories: dict[str, ChatMessageHistory] = {}  # Session-based history
        self.api_key = None  # Store API key for direct OpenAI use
        self.vectorizer = vectorizer  # ML vectorizer
        self.clf = clf  # ML classifier
        
        if openai_api_key is None:
            openai_api_key = os.getenv('OPENAI_API_KEY')  # Pull from env if not provided
        
        if openai_api_key:
            self.api_key = openai_api_key
            self.llm = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
            
            # Prompt template with history and plant knowledge
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful plant care assistant. Use this knowledge: {plant_knowledge}."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])
            
            chain = (
                RunnablePassthrough.assign(
                    plant_knowledge=lambda x: json.dumps(self.plant_data)
                )
                | prompt
                | self.llm
            )
            
            # Define callable for history (returns ChatMessageHistory)
            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in self.chat_histories:
                    self.chat_histories[session_id] = ChatMessageHistory()
                return self.chat_histories[session_id]
            
            # Wrap with history (no set_session_history needed)
            self.chain = RunnableWithMessageHistory(
                runnable=chain,
                get_session_history=get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )
        else:
            self.chain = None
            self.llm = None
    
    def get_history(self, session_id: str) -> ChatMessageHistory:
        """Get the history object for manual adds (rules/fallbacks)."""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = ChatMessageHistory()
        return self.chat_histories[session_id]
    
    def get_response(self, user_input: str, session_id: str = "plant_bot_session", image=None):
        user_lower = user_input.lower()
        history = self.get_history(session_id)
        
        # ML symptom classification (for text queries with symptoms) - BEFORE rules for better flow
        symptom_keywords = ["leaves", "wilting", "spots", "yellow", "brown", "tips"]
        if any(k in user_lower for k in symptom_keywords) and not image:
            symptom_text = user_input
            X_new = self.vectorizer.transform([symptom_text])
            prediction = self.clf.predict(X_new)[0]
            confidence = self.clf.predict_proba(X_new).max()
            ml_response = f"Based on your description, this sounds like {prediction} (confidence: {confidence:.2f}). "
            if confidence > 0.5:
                history.add_user_message(user_input)
                history.add_ai_message(ml_response)
                return ml_response
            else:
                # Low confidence: Fall to AI
                pass
        
        # Improved rule-based check: Broader matching for yellow leaves
        if "yellow" in user_lower and "leaves" in user_lower:
            response = "Sounds like overwatering! Check soil moisture."
            history.add_user_message(user_input)
            history.add_ai_message(response)
            return response
        
        # Fallback without API (only if no rules match)
        if not self.chain:
            response = "Tell me more about your plant issue!"
            history.add_user_message(user_input)
            history.add_ai_message(response)
            return response
        
        # If image provided, use vision chain
        if image:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)  # Use stored api_key
            
            # Encode image to base64
            buffered = BytesIO()
            image_obj = Image.open(image)
            image_obj.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Vision prompt with history context
            full_prompt = f"{user_input}\n\nPrevious context: {' '.join([m.content for m in history.messages[-4:]])}"  # Last 2 turns
            
            response_msg = client.chat.completions.create(
                model="gpt-4o-mini",  # Vision-capable, cheap
                messages=[
                    {"role": "system", "content": f"You are a plant care expert. Use this knowledge: {json.dumps(self.plant_data)}. Analyze the image for issues."},
                    {"role": "user", "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]}
                ],
                max_tokens=300
            )
            ai_response = response_msg.choices[0].message.content
            history.add_user_message(user_input)
            history.add_ai_message(ai_response)
            return ai_response
        
        # Use the chain with history (auto-adds messages)
        response = self.chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content

# Test it (text-only for console)
bot = PlantBot(openai_api_key=None)  # Will auto-pull from .env
print(bot.get_response("Hi, I'm Bob and I have a monstera."))
print(bot.get_response("My leaves are turning yellow."))
print(bot.get_response("My leaves are yellow and soil is soggy."))  # Now tests ML first

# Optional: Print history to verify
print("\n--- Conversation History ---")
history = bot.get_history("plant_bot_session")
for msg in history.messages:
    role = "User" if isinstance(msg, HumanMessage) else "Bot"
    print(f"{role}: {msg.content}")
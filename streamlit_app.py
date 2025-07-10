import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
import numpy as np
import random
import datetime

# UI Styling
st.markdown("""
    <style>
    .chat-box {
        background: white;
        border-radius: 20px;
        max-height: 65vh;
        padding: 20px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 12px;
        box-shadow: 0 15px 30px rgba(0,0,0,0.2);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #222;
        margin-bottom: 20px;
    }
    .bot-message {
        background: #ff8400;
        color: #222;
        padding: 12px 16px;
        border-radius: 20px;
        max-width: 80%;
        align-self: flex-start;
    }
    .user-message {
        background: #2575fc;
        color: white;
        padding: 12px 16px;
        border-radius: 20px;
        max-width: 80%;
        align-self: flex-end;
    }
    .chat-header {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
        padding: 15px;
        border-radius: 20px 20px 0 0;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 0px;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom right, #6a11cb, #2575fc);
        background-attachment: fixed;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .stTextInput > div > input {
        border-radius: 10px;
        padding: 10px;
        font-size: 1rem;
    }
    .custom-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #d1e8ff;
        text-align: center;
        margin-bottom: 1rem;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Force color on label for the text input */
    div[data-testid="stTextInput"] label {
        color: #add8e6 !important;  /* Light blue */
        font-weight: bold;
        font-size: 1.05rem;
    }

    button[kind="primary"] {
        background-color: #ff8400 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: bold;
        border: none !important;
    }    
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="custom-title">Sentiment Analysis Chatbot by Team Engineers</div>', unsafe_allow_html=True)
#st.title("Sentiment Analysis Chatbot by Team Engineers")

# Load fine-tuned model & tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = TFDistilBertForSequenceClassification.from_pretrained("./sentiment_yelp_model")
    tokenizer = AutoTokenizer.from_pretrained("./sentiment_yelp_model")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Response templates
positive_responses = [
    "😊 That's great to hear! You might also enjoy checking out our latest product deals: [Amazon Deals](https://www.amazon.com/gp/goldbox)",
    "👍 Thanks for your positive feedback! Would you like me to recommend similar products?",
    "🎉 We’re glad you're happy! Don’t forget to check your wishlist for exclusive offers: [Wishlist](https://www.amazon.com/hz/wishlist)",
]

negative_responses = [
    "😟 I'm sorry to hear that. Would you like to visit our [Help Center](https://www.amazon.com/help)?",
    "It seems you're facing issues. I can help with returns, refunds, or contact support if you need.",
    "🙁 That doesn't sound good. Do you want to open a support ticket or see troubleshooting tips?",
]

# State for conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = [
        {"role": "bot", "text": "Hi there! 👋 How can I assist you today?"}
    ]

if "last_prompt_was_suggestion" not in st.session_state:
    st.session_state.last_prompt_was_suggestion = False

def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=1).numpy()
    label = np.argmax(probs)
    confidence = probs[0][label]
    sentiment = "Positive" if label == 1 else "Negative"
    return sentiment, confidence

def respond_to_user(user_input):
    if st.session_state.last_prompt_was_suggestion and user_input.lower() in ["yes", "yeah", "yep", "sure"]:
        st.session_state.last_prompt_was_suggestion = False
        return {"text": "Great! Redirecting you to helpful resources: [Help Center](https://www.amazon.com/help)"}

    sentiment, confidence = classify_sentiment(user_input)
    st.session_state.conversation.append({"role": "user", "text": user_input})
    st.session_state.conversation.append({
        "role": "meta",
        "text": f"Detected sentiment: {sentiment} ({confidence*100:.1f}%)"
    })

    if sentiment == "Positive":
        st.session_state.last_prompt_was_suggestion = False
        return {"text": random.choice(positive_responses)}
    else:
        st.session_state.last_prompt_was_suggestion = True
        return {"text": random.choice(negative_responses)}

# Header
st.markdown('<div class="chat-header">🤖 Sentiment ChatBot</div>', unsafe_allow_html=True)

# Chat History HTML Construction
chat_history_html = '<div class="chat-box">'
for msg in st.session_state.conversation:
    css_class = "user-message" if msg["role"] == "user" else "bot-message"
    chat_history_html += f'<div class="{css_class}">{msg["text"]}</div>'
chat_history_html += '</div>'

st.markdown(chat_history_html, unsafe_allow_html=True)

# User Input
user_input = st.text_input("Type your message...", placeholder="Ask me anything...", key="input_text")

if st.button("Send ➤") and user_input:
    bot_reply = respond_to_user(user_input)
    st.session_state.conversation.append({"role": "bot", "text": bot_reply["text"]})
    st.rerun()

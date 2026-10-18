# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 04:31:55 2026

@author: Mayank
"""

# -*- coding: utf-8 -*-

import os

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from GenAI_Cap import get_llm


# Load environment variables
load_dotenv()


app = Flask(__name__)


# Configure folder path relative to application directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_PATH = os.getenv(
    "DOCS_PATH",
    os.path.join(BASE_DIR, "documents")
)


# Ensure documents folder exists
os.makedirs(DOCS_PATH, exist_ok=True)


# Initialize RAG chatbot pipeline
chatbot = get_llm(DOCS_PATH)


# Store conversation history
chat_history = []


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Chat endpoint
@app.route("/get_answer", methods=["POST"])
def get_answer():

    # Handle JSON input
    if request.is_json:
        data = request.get_json(silent=True) or {}
        user_input = data.get("question", "").strip()

    # Handle form input
    else:
        user_input = request.form.get("question", "").strip()

    # Validate question
    if not user_input:
        return jsonify({
            "error": "Empty question provided"
        }), 400

    try:

        # Send current question AND previous conversation
        response = chatbot.invoke({
            "question": user_input,
            "chat_history": chat_history
        })

        # Extract answer
        answer = response.get(
            "answer",
            "No answer could be derived."
        )

        # Update conversation history
        chat_history.append(
            (user_input, answer)
        )

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "error": "An error occurred while processing your request."
        }), 500


# Optional endpoint to clear conversation
@app.route("/clear_chat", methods=["POST"])
def clear_chat():

    chat_history.clear()

    return jsonify({
        "message": "Conversation history cleared."
    })


# Run Flask application
if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=False
    )
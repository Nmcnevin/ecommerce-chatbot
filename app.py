from flask import Flask, request, jsonify
from chatbot import EcommerceHFChatbot
# Initialize chatbot
chatbot = EcommerceHFChatbot("microsoft/DialoGPT-medium")
# Create Flask app
app =Flask(__name__)
@app.route("/")
def home():
 return " Ecommerce Chatbot is running on Render!"
@app.route("/chat", methods=["POST"])
def chat():
 data = request.json
 user_input = data.get("message", "")
 response = chatbot.chat(user_input)
 return jsonify({"reply": response})
if __name__ == "__main__":
 app.run(host="0.0.0.0", port=5000)

 
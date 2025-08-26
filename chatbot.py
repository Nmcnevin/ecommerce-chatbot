import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import json

class EcommerceHFChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        print(f"Loading model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None

        # Mock product database
        self.products = {
            "101": {"name": "Running Shoes", "price": "₹1999", "stock": 5},
            "102": {"name": "Leather Wallet", "price": "₹799", "stock": 12},
            "103": {"name": "Denim Jacket", "price": "₹2499", "stock": 3},
        }

        # Predefined helpdesk fallback responses
        self.fallback_responses = {
            "order": "Please provide your order ID so I can check the status for you 🛒.",
            "return": "You can return/exchange items within 7 days. Would you like me to start a return request?",
            "payment": "Sorry about the issue 💳. Could you share your order ID and payment method?",
            "product": "Please provide the product ID or name, and I'll fetch the details for you 📦.",
            "greeting": "Hello! Welcome to Sidewalks Store 🛍️. How can I help you today?",
            "goodbye": "Thank you for shopping with us 💖. Goodbye!",
            "thanks": "You're welcome! 😊 Glad I could help.",
            "faq": "📦 Standard delivery takes 3-5 business days. Yes, Cash on Delivery is available."
        }

    def detect_intent(self, text):
        """Simple keyword-based intent detection"""
        text = text.lower()
        if any(word in text for word in ["order", "track", "status", "shipping"]):
            return "order"
        elif any(word in text for word in ["return", "refund", "exchange"]):
            return "return"
        elif any(word in text for word in ["payment", "paid", "charged"]):
            return "payment"
        elif any(word in text for word in ["product", "detail", "size", "available", "price"]):
            return "product"
        elif any(word in text for word in ["hi", "hello", "hey"]):
            return "greeting"
        elif any(word in text for word in ["bye", "goodbye", "exit", "quit"]):
            return "goodbye"
        elif any(word in text for word in ["thanks", "thank you"]):
            return "thanks"
        elif any(word in text for word in ["delivery", "cod", "cash on delivery"]):
            return "faq"
        elif "reset" in text:
            return "reset"
        return None

    def get_order_status(self, order_id):
        """Mock order status lookup"""
        statuses = ["Processing", "Shipped", "Out for Delivery", "Delivered"]
        return f"Order {order_id} is currently **{random.choice(statuses)}** 🚚."

    def get_product_details(self, product_id):
        """Fetch product details from mock DB"""
        product = self.products.get(product_id)
        if product:
            return f"{product['name']} is available at {product['price']}.\nStock left: {product['stock']} 📦."
        else:
            return "Sorry, I couldn’t find that product. Please check the product ID."

    def reset_session(self):
        """Clears chat history"""
        self.chat_history_ids = None
        return "Chat history has been reset 🔄. Let's start fresh!"

    def chat(self, user_input):
        # First check intent
        intent = self.detect_intent(user_input)
        if intent:
            if intent == "order":
                # Try to find an order ID in input
                order_id = "".join(filter(str.isdigit, user_input))
                if order_id:
                    return self.get_order_status(order_id)
                return self.fallback_responses["order"]

            elif intent == "product":
                product_id = "".join(filter(str.isdigit, user_input))
                if product_id:
                    return self.get_product_details(product_id)
                return self.fallback_responses["product"]

            elif intent == "reset":
                return self.reset_session()

            return self.fallback_responses[intent]

        # Otherwise, use Hugging Face model
        new_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1) if self.chat_history_ids is not None else new_input_ids

        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        bot_reply = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return bot_reply


# 🚀 Run chatbot
def run_chatbot():
    chatbot = EcommerceHFChatbot("microsoft/DialoGPT-medium")
    print("🤖 Helpdesk Chatbot is ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Bot: Goodbye 👋")
            break

        response = chatbot.chat(user_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    run_chatbot()

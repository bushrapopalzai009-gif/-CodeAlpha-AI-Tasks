"""
AI FAQ Chatbot
==============
NLP-powered chatbot that matches user questions to FAQs using
tokenization, stop-word removal, TF-IDF, and cosine similarity.
"""

import re
import math
import random
from collections import Counter


# ========== FAQ KNOWLEDGE BASE ==========
FAQ_DATA = [
    {
        "questions": [
            "what are your hours", "when are you open", "business hours",
            "opening hours", "what time do you open", "what time do you close",
            "are you open today", "store hours"
        ],
        "answer": "We're open Monday–Friday 9 AM–6 PM, Saturday 10 AM–4 PM. Closed Sundays and public holidays."
    },
    {
        "questions": [
            "how do i reset my password", "forgot password", "change password",
            "password reset", "i can't log in", "login issue", "account locked"
        ],
        "answer": "Click 'Forgot Password' on the login page, enter your email, and we'll send a reset link valid for 24 hours. Check spam if you don't see it."
    },
    {
        "questions": [
            "do you offer refunds", "refund policy", "can i get a refund",
            "money back", "return policy", "how to return", "cancel order"
        ],
        "answer": "Yes, 30-day money-back guarantee. Items must be unused in original packaging. Refunds process in 5–7 business days."
    },
    {
        "questions": [
            "how to contact support", "customer service", "help desk",
            "support email", "phone number", "talk to human", "live chat"
        ],
        "answer": "Email: support@example.com | Phone: +1 (555) 123-4567 | Live chat available during business hours. Response time: ~24 hours."
    },
    {
        "questions": [
            "what payment methods do you accept", "can i pay with paypal",
            "credit card", "payment options", "installments", "buy now pay later"
        ],
        "answer": "We accept Visa, MasterCard, Amex, PayPal, Apple Pay, Google Pay, and bank transfers. Klarna installments available for orders over $50."
    },
    {
        "questions": [
            "shipping time", "how long does delivery take", "when will my order arrive",
            "delivery estimate", "express shipping", "track my order"
        ],
        "answer": "Standard: 5–7 business days. Express: 2–3 days ($9.99). Free shipping on orders over $75. Tracking link sent via email."
    },
    {
        "questions": [
            "do you ship internationally", "international shipping",
            "overseas delivery", "ship to europe", "customs fees"
        ],
        "answer": "Yes, we ship to 50+ countries. Delivery: 10–20 business days. Recipient pays customs duties and taxes."
    },
    {
        "questions": [
            "how to create an account", "sign up", "register",
            "new account", "join", "become a member"
        ],
        "answer": "Click 'Sign Up', enter email and password, then verify via the email link. Takes under a minute!"
    },
    {
        "questions": [
            "is my data secure", "privacy policy", "data protection",
            "gdpr", "do you sell my data", "encryption"
        ],
        "answer": "Absolutely. 256-bit SSL encryption, fully GDPR compliant. We never sell your data. See our Privacy Policy for details."
    },
    {
        "questions": [
            "what is your pricing", "how much does it cost", "plans",
            "subscription", "free trial", "discount", "coupon code"
        ],
        "answer": "Free tier available. Pro: $9.99/mo. Enterprise: $29.99/mo. 14-day free trial on all paid plans. Use WELCOME20 for 20% off your first year."
    }
]


# ========== NLP PREPROCESSING ==========
STOP_WORDS = {
    'a','an','the','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','must','shall',
    'can','need','dare','ought','used','to','of','in','for','on','with','at','by',
    'from','as','into','through','during','before','after','above','below','between',
    'under','again','further','then','once','here','there','when','where','why','how',
    'all','each','few','more','most','other','some','such','no','nor','not','only',
    'own','same','so','than','too','very','just','and','but','if','or','because',
    'until','while','what','which','who','whom','this','that','these','those','i',
    'me','my','myself','we','our','ours','ourselves','you','your','yours','yourself',
    'yourselves','he','him','his','himself','she','her','hers','herself','it','its',
    'itself','they','them','their','theirs','themselves','am','about','out','up','down',
    'off','over','under','again','further','then','once'
}


def tokenize(text):
    """Lowercase, remove punctuation, tokenize, remove stop words."""
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    tokens = [w for w in text.split() if w and w not in STOP_WORDS and len(w) > 1]
    return tokens


def compute_tf(tokens):
    """Compute term frequency."""
    count = Counter(tokens)
    total = len(tokens)
    return {term: freq / total for term, freq in count.items()} if total else {}


def compute_idf(documents):
    """Compute inverse document frequency."""
    N = len(documents)
    idf = {}
    all_terms = set()
    for doc in documents:
        all_terms.update(doc)
    for term in all_terms:
        doc_count = sum(1 for doc in documents if term in doc)
        idf[term] = math.log(N / (doc_count or 1)) + 1
    return idf


def vectorize(tokens, vocab, idf):
    """Convert tokens to TF-IDF vector."""
    tf = compute_tf(tokens)
    vec = []
    for term in vocab:
        val = tf.get(term, 0) * idf.get(term, 1)
        vec.append(val)
    return vec


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class FAQChatbot:
    def __init__(self):
        self.faq_data = FAQ_DATA
        self.vocabulary = []
        self.idf = {}
        self.doc_vectors = []
        self._build_index()

    def _build_index(self):
        """Preprocess all FAQ questions and build TF-IDF vectors."""
        all_docs = []
        for faq in self.faq_data:
            for q in faq["questions"]:
                tokens = tokenize(q)
                all_docs.append(tokens)

        # Build vocabulary
        vocab_set = set()
        for doc in all_docs:
            vocab_set.update(doc)
        self.vocabulary = sorted(vocab_set)

        # Compute IDF
        self.idf = compute_idf(all_docs)

        # Build document vectors
        self.doc_vectors = []
        for faq_idx, faq in enumerate(self.faq_data):
            for q in faq["questions"]:
                tokens = tokenize(q)
                vec = vectorize(tokens, self.vocabulary, self.idf)
                self.doc_vectors.append({
                    "vector": vec,
                    "faq_index": faq_idx
                })

    def get_response(self, user_input):
        """Find best matching FAQ answer."""
        user_tokens = tokenize(user_input)
        if not user_tokens:
            return "I didn't catch that. Could you rephrase?", 0.0

        user_vec = vectorize(user_tokens, self.vocabulary, self.idf)

        best_score = -1
        best_faq_index = -1

        for doc in self.doc_vectors:
            score = cosine_similarity(user_vec, doc["vector"])
            if score > best_score:
                best_score = score
                best_faq_index = doc["faq_index"]

        confidence = round(best_score * 100, 1)

        if best_score < 0.15:
            return ("I'm not sure I understood that. Try asking about: hours, "
                    "passwords, refunds, shipping, pricing, or support."), confidence

        return self.faq_data[best_faq_index]["answer"], confidence


def run_chatbot():
    bot = FAQChatbot()

    print("=" * 50)
    print("🤖  AI FAQ Chatbot")
    print("=" * 50)
    print("Type your question or 'exit' to quit.\n")

    greetings = ["Hello! How can I help you today?",
                 "Hi there! Ask me anything about our services.",
                 "Welcome! I'm here to answer your questions."]

    print(f"Bot: {random.choice(greetings)}\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye! Have a great day! 👋")
            break
        if not user_input:
            continue

        response, conf = bot.get_response(user_input)
        print(f"Bot: {response}")
        print(f"      [Match Confidence: {conf}%]\n")


if __name__ == "__main__":
    run_chatbot()
from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize Flask application
app = Flask(__name__)

# Route to serve the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Initialize the GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Define route to handle user messages and generate chatbot response
@app.route('/get-response', methods=['POST'])
def get_response():
    # Get the user message from the request
    user_message = request.json.get('message')

    # Process the user message and generate the chatbot response
    chatbot_response = generate_chatbot_response(user_message)

    # Print the chatbot response to the terminal
    print("Chatbot Response:", chatbot_response)

    # Return the chatbot response as JSON
    return jsonify({'response': chatbot_response})

# Function to generate chatbot response
def generate_chatbot_response(user_message):
    # Tokenize the user message
    input_ids = tokenizer.encode(user_message, return_tensors="pt")

    # Generate response using the GPT-2 model
    chatbot_output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    
    # Decode the generated response
    chatbot_response = tokenizer.decode(chatbot_output[0], skip_special_tokens=True)
    
    return chatbot_response

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

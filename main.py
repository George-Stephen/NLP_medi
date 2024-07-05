from flask import Flask, request, jsonify
import jsonlines

app = Flask(__name__)

# Load the medical questions and answers from JSONL
def load_medical_data(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            data.append(obj)
    return data

data = load_medical_data('medical_qa_data.jsonl')

def find_most_relevant_context(question, data):
    # Find the most relevant context based on simple string matching or other heuristics
    best_match = None
    max_overlap = 0
    for item in data:
        context = item['answer']
        overlap = len(set(question.split()).intersection(set(context.split())))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = context
    return best_match

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json['question']
    context = find_most_relevant_context(question, data)  # Find the most relevant context

    if not context:
        return jsonify({'answer': 'No relevant context found.'})

    # Log the input and output for debugging
    print(f"Question: {question}")
    print(f"Context: {context}")

    return jsonify({'answer': context})

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random

app = Flask(__name__)

def extract_sections_from_docx(file_path):
    doc = Document(file_path)
    sections = []
    current_heading = None
    current_content = []

    for paragraph in doc.paragraphs:
        if paragraph.style.name.startswith('Heading'):
            if current_heading:
                sections.append((current_heading, '\n'.join(current_content)))
            current_heading = paragraph.text
            current_content = []
        else:
            current_content.append(paragraph.text)

    if current_heading:
        sections.append((current_heading, '\n'.join(current_content)))

    return sections

# Extract sections from the Word document
sections = extract_sections_from_docx('handbuch.docx')

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
section_headings = [section[0] for section in sections]
tfidf_matrix = vectorizer.fit_transform(section_headings)

def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())


def find_answer(question):
    processed_question = preprocess_text(question)
    question_vector = vectorizer.transform([processed_question])

    similarities = cosine_similarity(question_vector, tfidf_matrix)
    most_similar_index = similarities.argmax()

    heading, content = sections[most_similar_index]

    if not content.strip():
        content = "Leider konnte keine spezifische Information zu deiner Frage gefunden werden. Hier ist ein allgemeiner Hinweis: Bitte präzisiere deine Frage oder stelle sie anders."


    intro_phrases = [
        f"Basierend auf der Frage zur Überschrift '{heading}', hier ist was ich gefunden habe:",
        f"Zum Thema '{heading}' kann ich Folgendes sagen:",
        f"In Bezug auf '{heading}' habe ich diese Information:",
    ]

    intro = random.choice(intro_phrases)
    answer = {
        'intro': intro,
        'content': content
    }
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json['question']
    answer = find_answer(question)
    return jsonify(answer)

if __name__ == '__main__':
    app.run(debug=False)

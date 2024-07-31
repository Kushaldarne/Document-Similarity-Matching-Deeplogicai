import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def load_invoices(directory):
    invoices = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            invoices[filename] = extract_text_from_pdf(filepath)
    return invoices

def find_most_similar_invoice(test_invoice_text, train_invoices):
    documents = [test_invoice_text] + list(train_invoices.values())
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    similarity_scores = cosine_matrix[0][1:]
    most_similar_index = similarity_scores.argmax()
    most_similar_invoice = list(train_invoices.keys())[most_similar_index]
    return most_similar_invoice, similarity_scores[most_similar_index]

if __name__ == "__main__":
    test_invoices_directory = 'test/'
    train_invoices_directory = 'train/'

    train_invoices = load_invoices(train_invoices_directory)
    test_invoices = load_invoices(test_invoices_directory)

    for test_invoice_name, test_invoice_text in test_invoices.items():
        most_similar_invoice, similarity_score = find_most_similar_invoice(test_invoice_text, train_invoices)
        print(f"Test invoice: {test_invoice_name}")
        print(f"Most similar invoice: {most_similar_invoice}")
        print(f"Similarity score: {similarity_score}\n")

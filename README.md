# Document Similarity Matching

## Overview

This project implements a system to identify and match similar documents, specifically invoices, based on their content and structure.

## Approach

1. **Document Representation**:

   - **Text Extraction**: Extract text content from PDFs using `PyPDF2`.
   - **Feature Extraction**: Use TF-IDF vectorization to convert text into feature vectors.
2. **Similarity Calculation**:

   - **Cosine Similarity**: Calculate the cosine similarity between the feature vectors of two invoices.

## Instructions

1. **Install Dependencies:**`pip install -r requirements.txt`
2. **Run the code:**   ` python main.py`
3. **Results:**
   The program compares each test invoice against the training invoices and outputs the most similar invoice along with the similarity score.


## Output:

1. Test invoice: invoice_102857.pdf

   Most similar invoice: invoice_102856.pdf

   Similarity score: 0.7612220242402069


2. Test invoice: invoice_77098.pdf

   Most similar invoice: invoice_77073.pdf

   Similarity score: 0.8101192883834126


## Explanation

1. **Text Extraction** : The `extract_text_from_pdf` function extracts text from each page of a PDF.
2. **Loading Invoices** : The `load_invoices` function loads all PDF files from the specified directory and extracts their text.
3. **Similarity Calculation** : The `find_most_similar_invoice` function calculates the cosine similarity between the test invoice and each training invoice, identifying the most similar one.

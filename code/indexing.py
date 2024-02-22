import PyPDF2
import json
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from pymongo import MongoClient
import pickle
        

class PDFIndexer:
    """Class to handle PDF text extraction, chunking, and embedding."""

    def __init__(self, pdf_path):
        """
        Initialize PDFIndexer object.

        Parameters:
        - pdf_path (str): Path to the PDF file.
        """

        self.pdf_path = pdf_path
        print("Created a PDFIndexer")


    def extract_text_from_pdf(self):
        """Extract text from the PDF file.

        Returns:
        - A clean text after reading the PDF and removing tab spaces & next line
        """

        print("Inside extract_text_from_pdf")
        pdf_file = open(self.pdf_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        pdf_file.close()

        # Clean the extracted text
        clean_text = text.replace('\t', ' ').replace('\n', ' ')
        return clean_text


    def create_chunks(self, clean_text, chunk_size, overlap = 32):
        """
        Chunk the dataset into smaller parts with overlap.

        Parameters:
        - clean_text: The dataset to be chunked (can be a list, tuple, etc.).
        - chunk_size: The size of each chunk.
        - overlap: The overlap size between adjacent chunks.

        Returns:
        - A list of dictionaries, where each dictionary represents a chunk.
        """
        
        print("Inside create_chunks")
        chunks = []
        start = 0
        end = len(clean_text)

        while start < end:
            # Ensure that the chunk size doesn't exceed the sequence length
            if start + chunk_size > end:
                break

            # Determine the end index of the current chunk
            chunk_end = min(start + chunk_size, end)

            # Extract the chunk from the sequence
            chunk_text = clean_text[start:chunk_end]

            # Create a dictionary representing the chunk
            temp_dict = {'id': len(chunks) + 1, 'text': chunk_text}
            chunks.append(temp_dict)

            # Move the start index forward by the chunk size minus the overlap amount
            start += chunk_size - overlap

        # Save chunks in a json file
        with open('../data/processed/chunks.json', 'w') as f:
            json.dump(chunks, f)

        return chunks
    

    def store_mongodb(self, embeddings):
        """
        Store embeddings and model to MongoDB

        Parameters:
        - embeddings (numpy.ndarray): Array of embeddings to be indexed.
        """

        print("Inside store_mongodb")

        # Serialize embeddings and model
        embeddings_binary = pickle.dumps(embeddings)

        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['LLM_RAG_WomenHealth']
        collection = db['womanEmbeddings']

        # Store the serialized embeddings and model in MongoDB
        collection.insert_one({'name': 'embeddings', 'data': embeddings_binary})

        print("Wrote embeddings to MongoDB")


    def generate_embeddings(self, chunks):
        """
        Generate embeddings for the text chunks.

        Parameters:
        - chunks (list): List of text chunks as dictionaries.

        Returns:
        - embeddings (numpy.ndarray): Array of embeddings for the text chunks.
        """
        print("Inside generate_embeddings")
        sentences = [chunk['text'] for chunk in chunks]
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embeddings = model.encode(sentences, show_progress_bar=True)
        
        # Save embeddings
        self.store_mongodb(embeddings)
        
        return embeddings


if __name__ == "__main__":
    # Path to the PDF file
    pdf_path = '../data/raw/womancode-alisa-vitti.pdf'
    # Create an instance of PDFIndexer
    indexer = PDFIndexer(pdf_path)
    # Extract text from the PDF
    clean_text = indexer.extract_text_from_pdf()
    # Create text chunks
    chunks = indexer.create_chunks(clean_text, 256)
    # Generate embeddings for the text chunks
    embeddings = indexer.generate_embeddings(chunks)

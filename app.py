import gradio as gr
import os
import shutil
import numpy as np
import pickle
import re
import nltk
from pathlib import Path

nltk_data_path = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', download_dir=nltk_data_path)

nltk.data.path.append(nltk_data_path)


from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing special characters and normalizing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> str:
        """Tokenize text and lemmatize words"""
        # Simple word tokenization by splitting on whitespace
        words = text.split()
        
        # Remove stopwords and lemmatize
        processed_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words
        ]
        
        return ' '.join(processed_words)
    
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline"""
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and lemmatize
        processed_text = self.tokenize_and_lemmatize(cleaned_text)
        
        return processed_text
    
    def preprocess_document(self, doc: Document) -> Document:
        """Preprocess a Document object while preserving metadata"""
        processed_text = self.preprocess(doc.page_content)
        return Document(
            page_content=processed_text,
            metadata=doc.metadata
        )

class VectorStore:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.vectorizer = TfidfVectorizer(stop_words=None)  # We handle stopwords in preprocessing
        self.document_vectors = None
        self.documents = []
        self.preprocessor = TextPreprocessor()
        
        os.makedirs(persist_directory, exist_ok=True)
        self.load_state()
    
    def add_documents(self, documents: list[Document]):
        # Preprocess documents
        processed_docs = [self.preprocessor.preprocess_document(doc) for doc in documents]
        
        # Extract text content
        texts = [doc.page_content for doc in processed_docs]
        
        # Update documents store
        self.documents.extend(processed_docs)
        
        # If this is the first batch of documents
        if self.document_vectors is None:
            self.document_vectors = self.vectorizer.fit_transform(texts)
        else:
            # Recompute vectors for all documents with updated vocabulary
            all_texts = [doc.page_content for doc in self.documents]
            self.vectorizer = TfidfVectorizer(stop_words=None)
            self.document_vectors = self.vectorizer.fit_transform(all_texts)
        
        self.save_state()
        return len(documents)
    
    def similarity_search(self, query: str, k: int = 5):
        if not self.documents:
            return []
        
        # Preprocess query
        processed_query = self.preprocessor.preprocess(query)
        
        # Transform query
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Get top k most similar
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return documents and scores
        return [(self.documents[i], similarities[i]) for i in top_indices]
    
    def save_state(self):
        state = {
            'vectorizer': self.vectorizer,
            'document_vectors': self.document_vectors,
            'documents': self.documents
        }
        with open(os.path.join(self.persist_directory, 'vector_store.pkl'), 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self):
        state_path = os.path.join(self.persist_directory, 'vector_store.pkl')
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                self.vectorizer = state['vectorizer']
                self.document_vectors = state['document_vectors']
                self.documents = state['documents']
    
    def clear(self):
        self.vectorizer = TfidfVectorizer(stop_words=None)
        self.document_vectors = None
        self.documents = []
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            os.makedirs(self.persist_directory)

VECTOR_DB = "vector_db"
DATA_PATH = "data"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""

# Initialize vector store
vector_store = VectorStore(VECTOR_DB)

def process_uploaded_file(file_obj):
    if file_obj is None:
        return "Please upload a file first!"
    
    os.makedirs(DATA_PATH, exist_ok=True)
    temp_path = file_obj.name
    dest_path = os.path.join(DATA_PATH, os.path.basename(temp_path))
    
    # Copy the file to our data directory
    shutil.copy(temp_path, dest_path)
    
    try:
        # Load and process the document
        loader = PyPDFLoader(dest_path)
        documents = loader.load()
        chunks = split_documents(documents)
        new_docs = vector_store.add_documents(chunks)
        
        return f"Successfully processed {os.path.basename(dest_path)}. Added {new_docs} new chunks to the database."
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return f"Error processing file: {str(e)}"

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def clear_database():
    vector_store.clear()
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
        os.makedirs(DATA_PATH)
    return "Database and data directory cleared successfully"

def query_rag(query_text: str):
    try:
        # Get similar documents
        results = vector_store.similarity_search(query_text, k=5)
        
        if not results:
            return "Please upload some documents first!", None
        
        # Create context from top documents
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Generate response using the LLM
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        try:
            model = Ollama(model="llama3.1:8b")
            response_text = model.invoke(prompt)
        except Exception as llm_error:
            return ("Error: Could not connect to Ollama. Please ensure Ollama is running with:\n" +
                   "1. Run 'ollama serve' in terminal\n" +
                   "2. Run 'ollama pull llama2:7b' in another terminal\n" +
                   f"Error details: {str(llm_error)}"), None
        
        # Get the most relevant document's source
        most_relevant_doc, similarity_score = results[0]
        source_path = most_relevant_doc.metadata.get("source", "Unknown")
        
        source_info = f"\nMost Relevant Source:\n- {os.path.basename(source_path)}"
        similarity_info = f"Similarity Score: {similarity_score:.2f}"
        
        # If the source file exists, return it
        if os.path.exists(source_path):
            return f"{response_text}\n{source_info}\n{similarity_info}", source_path
        else:
            return f"{response_text}\n{source_info}\n{similarity_info}", None
            
    except Exception as e:
        return f"Error processing query: {str(e)}", None

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Fastack.AI RAG Doc System")
    
    with gr.Tab("Upload Documents"):
        file_output = gr.File(
            label="Upload PDF",
            file_types=[".pdf"]
        )
        upload_button = gr.Button("Upload to Knowledge Base")
        clear_button = gr.Button("Clear Knowledge Database")
        status_output = gr.Textbox(label="Status")
        
        upload_button.click(
            process_uploaded_file,
            inputs=[file_output],
            outputs=[status_output]
        )
        clear_button.click(
            clear_database,
            inputs=[],
            outputs=[status_output]
        )
    
    with gr.Tab("Query Documents"):
        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="Ask a question about your documents..."
                )
                query_button = gr.Button("Submit Query")
                answer_output = gr.Textbox(
                    label="Response",
                    lines=10
                )
            
            with gr.Column(scale=1):
                source_display = gr.File(
                    label="Source Document",
                    file_types=[".pdf"],
                    value=None,
                    height=600
                )
        
        query_button.click(
            query_rag,
            inputs=[query_input],
            outputs=[answer_output, source_display]
        )

if __name__ == "__main__":
    demo.launch()

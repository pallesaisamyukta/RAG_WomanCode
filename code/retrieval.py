import streamlit as st
from pymongo import MongoClient
import pickle
import json
from sentence_transformers import SentenceTransformer
import openai
from sklearn.metrics.pairwise import cosine_similarity
import time
class Preprocessor:
    """
    Class to preprocess data and load the Faiss index.
    """

    def __init__(self, json_path, model_path, mongo_uri, db_name, collection_name):
        """
        Initialize Preprocessor object.

        Parameters:
        - json_path (str): Path to the JSON file containing chunks.
        - model_path (str): Path to the Sentence Transformer model.
        - mongo_uri (str): MongoDB connection URI.
        - db_name (str): Name of the MongoDB database.
        - collection_name (str): Name of the collection containing the Faiss index.
        """

        self.json_path = json_path
        self.model_path = model_path
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name


    def preprocess_data(self):
        """
        Preprocess data by extracting chunks and loading the Embeddings.

        Returns:
        - list: List of sentences extracted from the JSON file.
        - Embeddings: Deserialized Embeddings from MongoDB
        """

        print("Inside preprocess_data")
        # Extract chunks
        sentences = self._extract_chunks_content()

        # Load Embeddings
        embeddings = self._load_embeddings()

        return sentences, embeddings


    def _extract_chunks_content(self):
        """
        Extract chunks from the JSON file.

        Returns:
        - list: List of sentences extracted from the JSON file.
        """

        print("Inside _extract_chunks_content")

        # Read the JSON file and load its content into a dictionary
        with open(self.json_path, 'r') as json_file:
            chunks = json.load(json_file)
        sentences = [chunk['text'] for chunk in chunks]

        return sentences


    def _load_embeddings(self):
        """
        Load the Faiss index from MongoDB.

        Returns:
        - Faiss index object: Deserialized Faiss index object.
        """
        print("Inside load_index")
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        collection = db[self.collection_name]

        # Retrieve the serialized embeddings from MongoDB
        document = collection.find_one({'name': 'embeddings'})
        embeddings_binary = document['data']

        # Deserialize the embeddings
        embeddings = pickle.loads(embeddings_binary)
        return embeddings


class PromptGenerator:
    """
    Class to generate prompts for the user question.
    """

    def __init__(self, sentences, model, embeddings):
        """
        Initialize PromptGenerator object.

        Parameters:
        - sentences (list): List of sentences for context.
        - model (str): Path to the Sentence Transformer model.
        - faiss_index: Loaded Faiss index object.
        """
        self.sentences = sentences
        self.model = SentenceTransformer(model)
        self.embeddings = embeddings


    def top_k_sentences(self, query_embedding, k = 50):

        # Compute cosine similarity between query_embedding and self.embeddings
        similarities = cosine_similarity(query_embedding, self.embeddings)
        # Find the indices of the top k nearest neighbors
        top_indices = similarities.argsort(axis=1)[0][-k:][::-1]
        # Retrieve the top k nearest sentences
        context = [self.sentences[idx] for idx in top_indices]

        return context


    def generate_prompt(self, question, k=5):
        """
        Generate a prompt for the user question.

        Parameters:
        - question (str): User question.
        - k (int): Number of nearest neighbors to retrieve.

        Returns:
        - str: Prompt containing the question and context.
        """
        print("Inside generate prompt")
        query_embedding = self.model.encode([question])
        context = self.top_k_sentences(query_embedding)
        # print(context)
        base_prompt = """You are an AI assistant. Your task is to understand the user question, and provide an answer using only the useful parts of the provided contexts. 
        Your answers are correct, high-quality, and written by an domain expert. If the provided context does not contain the answer, simply state, 
        "The provided context does not have the answer."

        User question: {}

        Contexts:
        {}
        """
        prompt = base_prompt.format(question, context)
        return prompt


# Title of the app
st.title("WomenCode")

# Guide message for using the pre-defined model
st.markdown(f"*If you want direct GPT-3.5 Turbo output, start your query with 'Using the pre-defined model:' followed by the question.*")

# Paths and configurations
json_path = 'D:/Duke/Sem2/LLMs/RAG/WomanCode/data/processed/chunks.json'
model_path = 'D:/Duke/Sem2/LLMs/RAG/WomanCode/data/processed/sentence_transformer_model'
mongo_uri = 'mongodb://localhost:27017/'
db_name = 'LLM_RAG_WomenHealth'
collection_name = 'womanEmbeddings'
openai.api_key = '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'

# Instantiate Preprocessor
preprocessor = Preprocessor(json_path, model_path, mongo_uri, db_name, collection_name)

# Preprocess data and load Faiss index
sentences, embeddings = preprocessor.preprocess_data()

# Instantiate PromptGenerator
prompt_generator = PromptGenerator(sentences, model_path, embeddings)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

store_time = []

if user_input := st.chat_input("What is up?"):

    # Check if the user input contains "pre-defined model"
    if "pre-defined model" in user_input:
        # Send the user input directly to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": user_input}
            ],
            max_tokens=4096,
        )

        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.write(response['choices'][0]['message']['content'])

    else:
        # Start measuring time
        start_time = time.time()

        # Generate prompt
        prompt = prompt_generator.generate_prompt(user_input)
        print(prompt)

        prev_msgs = [{"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages]
        
        if prev_msgs:
            prompt = prompt + "Consider the previous messages in the chat and try to add on to it. Previous chats are - " + str(prev_msgs)
            print(prompt)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages = [
                    {"role": "system", "content": prompt}
                ],

                max_tokens=4096,
            )
            # Display the generated text
            st.write(response['choices'][0]['message']['content'])
        st.session_state.messages.append({"role": "assistant", "content":response['choices'][0]['message']['content']})
        print("I'm done")

        # End measuring time
        end_time = time.time()

        # Calculate the time taken
        time_taken = end_time - start_time

        store_time.append(time_taken)

        # Display time taken in italics
        st.markdown(f"*Time taken to retrieve answer: {time_taken:.2f} seconds*")
        print(store_time)


    
    # st.session_state.messages.append({"role": "assistant", "content": response})





# # Generate button
# if st.button("Generate"):
#     # Paths and configurations
#     json_path = 'D:/Duke/Sem2/LLMs/RAG/WomanCode/data/processed/chunks.json'
#     model_path = 'D:/Duke/Sem2/LLMs/RAG/WomanCode/data/processed/sentence_transformer_model'
#     mongo_uri = 'mongodb://localhost:27017/'
#     db_name = 'womanHealth'
#     collection_name = 'womanCodeBook'

#     # Instantiate Preprocessor
#     preprocessor = Preprocessor(json_path, model_path, mongo_uri, db_name, collection_name)

#     # Preprocess data and load Faiss index
#     sentences, faiss_index = preprocessor.preprocess_data()

#     # Instantiate PromptGenerator
#     prompt_generator = PromptGenerator(sentences, model_path, faiss_index)

#     # Generate prompt
#     prompt = prompt_generator.generate_prompt(question)

#     # Set your OpenAI API key
#     openai.api_key = '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'

#     # Generate text using GPT-2
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",  # You can use "gpt-2" or "gpt-3" depending on the model you want to use
#         messages=[
#             {"role": "system", "content": prompt}
#         ],
#         max_tokens=500  # Adjust this parameter as needed
#     )

#     # Display the generated text
#     st.write(response['choices'][0]['message']['content'])

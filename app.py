#import libraries
from sentence_transformers import SentenceTransformer
import torch
import gradio as gr
from huggingface_hub import InferenceClient



#from sematic search lab
def preprocess_text(text):
  # Strips whitespace from beginning and end of text
  cleaned_text = text.strip()
  # Split the cleaned_text by every newline character (\n)
  chunks = cleaned_text.split("\n")
  # Create an empty list for cleaned chunks
  cleaned_chunks = []
  # Write your for-in loop below to clean each chunk and add it to the cleaned_chunks list
  for chunk in chunks:
    stripped_chunk = chunk.strip()
    if len(stripped_chunk) > 0:
      cleaned_chunks.append(stripped_chunk)
  return cleaned_chunks
#converts text into vectors    
model = SentenceTransformer('all-MiniLM-L6-v2')

 # this fcn converts each text chunk into a vector embedding and store as a tensor
def create_embeddings(text_chunks):
  #encode takes each list of strings and covnerts into a vector thats captured in an embedding
  chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True) # Replace ... with the text_chunks list
  # Print the chunk embeddings
  print(chunk_embeddings)
  # Print the shape of chunk_embeddings
  print(chunk_embeddings.shape)
  # Return the chunk_embeddings
  return chunk_embeddings


# Define a function to find the most relevant text chunks for a given query, chunk_embeddings, and text_chunks
def get_top_chunks(query, chunk_embeddings, text_chunks):
  # Convert the query text into a vector embedding
  query_embedding = model.encode(query, convert_to_tensor=True) # Complete this line

  # Normalize the query embedding to unit length for accurate similarity comparison
  query_embedding_normalized = query_embedding / query_embedding.norm()

  # Normalize all chunk embeddings to unit length for consistent comparison
  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

  # Calculate cosine similarity between all chunks and the query using matrix multiplication
  similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized) # Complete this line

  # Print the similarities
  print(similarities)

  # Find the indices of the 3 chunks with highest similarity scores
  top_indices = torch.topk(similarities, k=3).indices

  # Print the top indices
  print(top_indices)

  # Create an empty list to store the most relevant chunks
  top_chunks = []

  # Loop through the top indices and retrieve the corresponding text chunks
  for i in top_indices:
    chunk = text_chunks[i]
    top_chunks.append(chunk)

  # Return the list of most relevant chunks
  return top_chunks

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

#this function picks a file to open and read
def pick_file(user_dropdown):
    if user_dropdown == "For Everyday Users":
        with open("disposal.txt", "r", encoding="utf-8") as file:
            bot_file = file.read()
        return bot_file
    else:
        with open("business.txt", "r", encoding="utf-8") as file:
            bot_file = file.read()
        return bot_file

def respond(message, history, location, user_type):
 
    #chosen file is stored in this variable
    chosen_file = pick_file(user_type)
    #cleaned chunks processes chosen file
    cleaned_chunks = preprocess_text(chosen_file)
    #if no file there, then there is nothing given to the chatbot (safety measure)
    if not cleaned_chunks:
        info = ["No background information available."]
    else:
        chunk_embeddings = create_embeddings(cleaned_chunks)
        info = get_top_chunks(message, chunk_embeddings, cleaned_chunks)

    if user_type == "For Everyday Users":
        specific_instruction = "Pull information specifically about the location: {location}"
    else:
        specific_instruction = "Ignore location, only answer questions about sustainable materials/packaging for businesses"

    #message to the chatbot    
    messages = [{"role": "system",
                 "content": (
                     f'You are a Bio Buddy, a friendly chatbot that answers questions informatively, yet concisely based on {info}.'
                     f'The current user_type is {user_type}, respond with {user_type} when asked about this.'
                     f'The current location of the user is {location}, respond with {location} when asked about this.'
                     f'{specific_instruction}'
                 )
                }]
    
    if history:
        messages.extend(history)
        
    messages.append({"role": "user", "content": message})
    
    response = client.chat_completion(
        messages,
        #controls length of the response, setting it to None, so the bot can decide how long the response needs to be
        max_tokens=None,
        temperature=0.5
    )
    
    return response.choices[0].message.content.strip()

#custom theme 
custom_theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="neutral", 
    neutral_hue="emerald",
    spacing_size="lg",
    radius_size="lg",
    text_size="lg",
    font=[gr.themes.GoogleFont("Proxima Nova"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Proxima Nova"), "monospace"]
)


# Drop-down menu with the locations we have info for
with gr.Blocks() as chatbot:
    gr.Image("BioBuddyBanner.png")
    location_dropdown = gr.Dropdown(choices=["San Francisco","Milpitas","Hayward"], label="Select your Location")
    user_type_dropdown = gr.Dropdown(choices=["For Everyday Users","For Businesses"], label="Who are you?")
    gr.ChatInterface(respond, additional_inputs=[location_dropdown, user_type_dropdown])

chatbot.launch(theme=custom_theme)
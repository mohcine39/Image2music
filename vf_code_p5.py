import numpy as np 
import pandas as pd 
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
# Configure the name of the upload_folder
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Create the function that calls the cognitive services API and gives a description about the image 
def describe_image(directory, image_filename):
    # Specify the subscription key and region
    subscription_key = "07134e92f4e14b319acbd2aa25d04133"
    endpoint = "https://eastus.api.cognitive.microsoft.com/"
    # Construct the API URL
    url = endpoint + "vision/v3.2/analyze"
    # Configure the request parameters
    headers = {"Ocp-Apim-Subscription-Key": subscription_key, "Content-Type": "application/octet-stream"}
    params = {"visualFeatures": "Description"}
    # Open the image and convert to bytes
    with open(os.path.join(directory, image_filename), "rb") as image_file:
        image_data = image_file.read()
    # Send the request to the API and get the response
    response = requests.post(url, headers=headers, params=params, data=image_data)
    # Extract the description from the JSON response
    result = json.loads(response.content)
    description = result["description"]["captions"][0]["text"]
 
    return description

# Configure the path of the upload page and the template to render
@app.route('/')
def upload_file():
    return render_template('upload.html')

# Configure the uploader page (after uploading the image) 
@app.route('/uploader', methods=['POST'])
# Define the uploader function 
def uploader():
    f = request.files['file']
    filename = secure_filename(f.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(file_path)

    # Describe the image
    image_description = describe_image(app.config['UPLOAD_FOLDER'], filename)
    print(image_description)

    # Load Data
    data1 = pd.read_csv("D:/Downloads (D)/dataset1.csv")

    # Extract all the combinations of seeds
    graines_uniques = data1['seeds'].unique()

    # Extract all the distinct seeds
    mots_cles_uniques = []

    for graine in graines_uniques:
        # Split the string into a list of keywords by removing special characters
        mots_cles = [mot.strip(" '[]") for mot in graine.split(",")]
        # Add the keywords to the list of unique keywords
        mots_cles_uniques.extend(mots_cles)

    # Convert the list of unique keywords into a set to eliminate duplicates
    mots_cles_uniques = set(mots_cles_uniques)
    mots_cles_uniques = list(mots_cles_uniques)

    # Load the pre-trained model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Define the description and the vector mots_cles_uniques
    description = image_description
    mots_cles_uniques = np.array(mots_cles_uniques)

    # Calculate the embeddings
    description_embedding = model.encode(description, convert_to_tensor=True)
    mots_cles_uniques_embeddings = model.encode(mots_cles_uniques, convert_to_tensor=True)

    # Calculate the cosine similarity between the description embedding and each vector in mots_cles_uniques
    cosine_similarities = np.dot(description_embedding, mots_cles_uniques_embeddings.T) / (np.linalg.norm(description_embedding) * np.linalg.norm(mots_cles_uniques_embeddings, axis=1))
   
    # Get indices of all the similarities
    similarities_idx = np.argsort(cosine_similarities)[::-1]

    # Get the elements and their cosine similarity scores
    elements = mots_cles_uniques[similarities_idx]
    similarities = cosine_similarities[similarities_idx]

    # Create a dictionary to store the elements with their cosine similarity scores
    results = {}
    for i in range(len(elements)):
        results[elements[i]] = similarities[i]

    # Display the elements with their cosine similarity scores
    for elem, score in results.items():
        print(f"Element: {elem}, Similarity: {score}")

    # Create a list of sorted seeds
    sorted_seeds = sorted(mots_cles_uniques)

    # Create a dictionary to store the elements with their cosine similarity scores
    results = {}
    for i in range(len(elements)):
        results[elements[i]] = similarities[i]

    # Create a new vector that contains the value of similarity for each element from sorted_seeds
    similarity_scores = np.zeros(len(sorted_seeds))
    for i, elem in enumerate(sorted_seeds):
        if elem in results:
            similarity_scores[i] = results[elem]

    print(similarity_scores)

    # Create a list to store the binary vectors
    binary_vectors = []

    # Loop through each row in the dataset and create a binary vector for it
    for seeds in data1['seeds']:
        # Initialize a list of zeros with the same length as the sorted seed vector
        binary_vector = [0] * len(sorted_seeds)
        # Set the corresponding indices to 1 if the seed is in the sorted seed vector
        for i, word in enumerate(sorted_seeds):
            if word in seeds:
                binary_vector[i] = 1
        # Add the binary vector to the list
        binary_vectors.append(binary_vector)
        
    # Convert the list of binary vectors to a NumPy array
    binary_vectors_array = np.array(binary_vectors)

    # Print the resulting binary vectors array
    print(binary_vectors_array)

    # Multiply the binary_vectors_array by similarity_scores
    multip1=np.multiply(binary_vectors_array,similarity_scores)
    print(multip1)

    # Calculate the sums vector
    sums_vector = np.sum(multip1, axis=1)

    # Divide the sums vector by the values in data1['D']
    averages_vector = np.array(sums_vector / data1['D'])

    # Print the result
    print(averages_vector)

    # Get the indices that would sort the positive values in descending order
    sort_indices = np.argsort(-averages_vector[averages_vector > 0])

    # Get the positive values in descending order
    sorted_positive = averages_vector[averages_vector > 0][sort_indices]

    # Get the corresponding indices in the original vector
    sorted_indices = np.nonzero(averages_vector > 0)[0][sort_indices]

    # Print the result
    print("Sorted positive values with indices:", sorted_positive)
    print("Indices in original vector:", sorted_indices)

    # Initialize lists
    musics = []
    urls = []

    # Iterate through the top 2 indices and get the music tracks and URLs
    for i in sorted_indices[:2]:
        music_tracks = data1.loc[i, 'musics']
        urls_tracks = data1.loc[i, 'urls']

        music_tracks = music_tracks.replace("[","").replace("]","").split("+")
        music_tracks = [music.replace("'", "") for music in music_tracks]

        urls_tracks = urls_tracks.replace("[","").replace("]","").split("+")
        urls_tracks = [url.replace("'", "") for url in urls_tracks]

        # Add tracks to lists
        musics.extend(music_tracks)
        urls.extend(urls_tracks) 

    # Print the result
    print(musics)
    print(len(urls))

    # Pass the output list to your HTML template
    return render_template('upload.html', musics=musics , urls=urls)

if __name__ == '__main__':
    app.run(debug=True)

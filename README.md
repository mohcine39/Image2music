# Music-recommendation-app-based-on-image
Music Suggestion System

# Introduction
The Music Suggestion System is a project aimed at recommending music based on the content of an image uploaded by the user. The system uses Azure's Cognitive Services API to generate a description of the image and then passes it through a transformer-based model to get an embedding vector. This vector is then compared to a set of pre-defined emotion vectors to find the most suitable emotions for the image. Finally, a dataset of music associated with the identified emotions is used to suggest songs to the user.

# Requirements
Python 3.7 or higher
Hugging Face transformers library
Flask web framework
Azure Cognitive Services API credentials

# Installation
Clone the repository to your local machine.
Install the required packages using the following command: pip install -r requirements.txt.
Add your Azure Cognitive Services API credentials in the config.py file.
Modify the path of the dataset here ---> file data1 = pd.read_csv("D:/Downloads (D)/dataset1.csv").

# Usage : 
Run the Flask application by executing the following command in your terminal: python app.py
Open your web browser and go to http://localhost:5000/.
Upload an image and click on the "Submit" button to get a list of recommended songs.

# Future Improvements
Integrating the Music Suggestion System into social media platforms like Instagram to suggest music for stories.
Developing a mobile application for easy access to the system on the go.

# Conclusion
The Music Suggestion System is a unique project that combines image recognition and music recommendation to provide a seamless and personalized experience for users. With the power of machine learning and cloud computing, the system is able to generate accurate recommendations based on the content of an image. This project has a lot of potential for further development, and we hope that it inspires more innovative ideas in the intersection of technology and art.

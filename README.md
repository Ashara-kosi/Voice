# Gender Voice Recognition using streamlit

### Description
The data set was gotten from kaggle containing 3,168 recorded voice samples and 21 acoustic properties of the voice and speech.The application is able to view the datasets, 
some visualizations and also take inputs of the acoustic properties and predict if the voice is a male or female.Logistic regression model was used for the classification and streamlit was used to create a web app for the application.
There are many input parameters for prediction making it exhausting for users to test.I was able to use the heatmap for feature engineering to select the important features.

### How to set up the project locally
clone the repo
`git clone https://github.com/Ashara-kosi/Voice/'`

Installing packages 
`pip install -r requirements.txt`

Then run
`streamlit run main.py` to view the app locally

The app has been deployed to heroku and can be viewed using this link `https://streamlit-reg.herokuapp.com/`

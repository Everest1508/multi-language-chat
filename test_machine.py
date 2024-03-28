import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Sample dataset of questions and their corresponding intents
# Expanded training data with responses
training_data = [
    ("What type of legal issue are you facing?", "legal_issue", "Please specify the type of legal issue you're facing."),
    ("In which city/state/country do you need legal assistance?", "location", "Where do you need legal assistance? Please provide the city/state/country."),
    ("What is your preferred language for communication with the lawyer?", "language_preference", "What language would you prefer for communication with the lawyer?"),
    ("Do you have any specific preferences for the lawyer's experience level or expertise?", "lawyer_preference", "Do you have any specific preferences for the lawyer's experience level or expertise?"),
    ("Do you want to know the number of winning cases of your lawyer?", "lawyer_experience", "Do you want to know the number of winning cases of your lawyer?"),
    ("Do you prefer to work with a lawyer from a specific law firm or organization?", "law_firm_preference", "Do you prefer to work with a lawyer from a specific law firm or organization?"),
    ("Have you faced any challenges or barriers in accessing legal assistance before?", "accessing_legal_help", "Have you faced any challenges or barriers in accessing legal assistance before?"),
    ("Are there any cultural or religious considerations that the lawyer should be aware of?", "cultural_religious_considerations", "Are there any cultural or religious considerations that the lawyer should be aware of?"),
    ("Is there any additional information or context you would like to provide about your situation?", "additional_context", "Is there any additional information or context you would like to provide about your situation?"),
    ("Are our suggestions helpful for you?", "feedback", "Are our suggestions helpful for you?"),
    ("Would you like to give a rating to the lawyer after your interaction?", "rating_preference", "Would you like to give a rating to the lawyer after your interaction?"),
    ("What is your preferred mode of communication with your lawyer? (e.g., email, phone)", "communication_preference", "What is your preferred mode of communication with your lawyer? (e.g., email, phone)"),
    ("Is there any urgency or deadline associated with your legal matter?", "urgency", "Is there any urgency or deadline associated with your legal matter?"),
    ("Are there any specific qualifications or certifications you expect the lawyer to have?", "qualifications_expectation", "Are there any specific qualifications or certifications you expect the lawyer to have?"),
    ("Is this issue related to personal or business affairs?", "personal_or_business_affairs", "Is this issue related to personal or business affairs?"),
    ("Do you have any concerns about confidentiality or privacy that you would like addressed?", "confidentiality_privacy_concerns", "Do you have any concerns about confidentiality or privacy that you would like addressed?"),
    ("I received a cease and desist letter, what should I do next?", "legal_advice_cease_desist", "I received a cease and desist letter, what should I do next?"),
    ("Can you explain the process of applying for a patent?", "patent_application_process", "Can you explain the process of applying for a patent?"),
    ("What are the laws regarding workplace discrimination?", "workplace_discrimination_laws", "What are the laws regarding workplace discrimination?"),
    ("I need legal advice regarding starting a business, can you suggest a knowledgeable business attorney?", "business_legal_advice", "I need legal advice regarding starting a business, can you suggest a knowledgeable business attorney?"),
    ("I'm considering filing for bankruptcy, can you recommend a bankruptcy attorney who offers compassionate support and guidance?", "bankruptcy_attorney_recommendation", "I'm considering filing for bankruptcy, can you recommend a bankruptcy attorney who offers compassionate support and guidance?"),
    ("I'm a victim of cyberbullying, can you recommend a lawyer who handles internet defamation cases?", "cyberbullying_legal_assistance", "I'm a victim of cyberbullying, can you recommend a lawyer who handles internet defamation cases?"),
    # Add more training data here...
]

# Function to retrieve response based on intent
def get_response(intent):
    for question, intent_label, response in training_data:
        if intent_label == intent:
            return response
    return "I'm sorry, I couldn't understand your query."


# Preprocessing functions
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Prepare training data
X_train = [preprocess_text(question) for question, intent, response in training_data]
y_train = [intent for question, intent, response in training_data]

# Build pipeline for intent classification
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC(dual=False)),
])

# Train the classifier
pipeline.fit(X_train, y_train)

# Function to classify intents
def classify_intent(text):
    preprocessed_text = preprocess_text(text)
    intent = pipeline.predict([preprocessed_text])[0]
    return intent

# Function to generate responses
def generate_response(intent):
    for i in training_data:
        if intent == i[1]:
            return(i[2])

# Example usage
user_input = input("Ask a question: ")
predicted_intent = classify_intent(user_input)
response = generate_response(predicted_intent)
print(response)

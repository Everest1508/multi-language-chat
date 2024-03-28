import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


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

# Convert intents to one-hot encoding
intents = {'legal_issue': 0, 'location': 1, 'language_preference': 2, 'lawyer_preference': 3,
           'lawyer_experience': 4, 'law_firm_preference': 5, 'accessing_legal_help': 6,
           'cultural_religious_considerations': 7, 'additional_context': 8, 'feedback': 9,
           'rating_preference': 10, 'communication_preference': 11, 'urgency': 12,
           'qualifications_expectation': 13, 'personal_or_business_affairs': 14,
           'confidentiality_privacy_concerns': 15, 'legal_advice_cease_desist': 16,
           'patent_application_process': 17, 'workplace_discrimination_laws': 18,
           'business_legal_advice': 19, 'bankruptcy_attorney_recommendation': 20,
           'cyberbullying_legal_assistance': 21}

# Convert training data to TF-IDF features
X_train = [question for question, intent_label, response in training_data]
y_train = [intents[intent_label] for question, intent_label, response in training_data]

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Split data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(intents), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_split.toarray(), np.array(y_train_split), epochs=10, batch_size=32, validation_data=(X_val_split.toarray(), np.array(y_val_split)))

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val_split.toarray(), np.array(y_val_split))
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')

# Function to classify intents using the trained model
def classify_intent_nn(text):
    preprocessed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([preprocessed_text])
    intent_idx = np.argmax(model.predict(text_vectorized.toarray()), axis=1)[0]
    intent_label = [key for key, value in intents.items() if value == intent_idx][0]
    return intent_label

# Example usage
user_input = input("Ask a question: ")
predicted_intent = classify_intent_nn(user_input)
response = get_response(predicted_intent)
print(response)

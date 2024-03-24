from googletrans import Translator

def translate_to_preferred_language(sentence, preferred_language):
    translator = Translator()
    translated_sentence = translator.translate(sentence, dest=preferred_language)
    return translated_sentence.text

def main():
    print("Welcome! Please select your preferred language:")
    print("1. English")
    print("2. Spanish")
    print("3. French")
    print("4. Hindi")
    print("5. Marathi")
    
    choice = input("Enter the number corresponding to your choice: ")
    
    sentence = input("Enter the sentence you want to translate: ")
    
    if choice == '1':
        preferred_language = 'en'
    elif choice == '2':
        preferred_language = 'es'
    elif choice == '3':
        preferred_language = 'fr'
    elif choice == '4':
        preferred_language = 'hi'
    elif choice == '5':
        preferred_language = 'mr'
    else:
        print("Invalid choice. Please select a number between 1 and 5.")
        return
    
    translated_sentence = translate_to_preferred_language(sentence, preferred_language)
    print("Translated sentence:", translated_sentence)
    
    filename = "translated_sentence.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(translated_sentence)
        
    print(f"Translated sentence has been saved to '{filename}'.")

if __name__ == "__main__":
    main()

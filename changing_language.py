from googletrans import Translator

def translate_to_preferred_language(sentence, preferred_language):
    translator = Translator()
    translated_sentence = translator.translate(sentence, dest=preferred_language)
    return translated_sentence.text
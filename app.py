import streamlit as st
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
import io
from PIL import Image
from wordcloud import WordCloud
from collections import Counter
import logging
import sys
from langdetect import detect
import unicodedata
import regex
import matplotlib.font_manager as fm
import pandas as pd

# Try to import indic_tokenize, with fallback
try:
    from indicnlp.tokenize import indic_tokenize
    INDIC_NLP_AVAILABLE = True
except ImportError:
    INDIC_NLP_AVAILABLE = False
    logging.warning("indic-nlp-library not available. Falling back to regex tokenization for Indic languages.")

# Try to import grapheme library for better Indic script handling
try:
    import grapheme
    GRAPHEME_AVAILABLE = True
except ImportError:
    GRAPHEME_AVAILABLE = False
    logging.warning("grapheme library not available. Falling back to regex for grapheme clustering.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')
        logger.info("NLTK data downloaded successfully.")

# Load stopwords for different languages
def load_stopwords():
    """Load stopwords for different languages.
    
    Returns:
        dict: Dictionary with language codes as keys and sets of stopwords as values.
    """
    stopword_dict = {}
    
    # English stopwords from NLTK
    try:
        stopword_dict['english'] = set(stopwords.words('english'))
    except:
        ensure_nltk_data()
        stopword_dict['english'] = set(stopwords.words('english'))
    
    # Hindi stopwords (minimal curated list)
    stopword_dict['hindi'] = {
        'का', 'के', 'की', 'है', 'में', 'से', 'हैं', 'को', 'पर', 'इस', 'होता', 'कि', 'जो', 'कर', 'मे', 
        'गया', 'करने', 'किया', 'लिये', 'अपने', 'ने', 'बनी', 'नहीं', 'तो', 'ही', 'या', 'एवं', 'दिया', 'हो', 
        'इसका', 'था', 'द्वारा', 'हुआ', 'तक', 'साथ', 'करता', 'हुई', 'एक', 'और', 'यह', 'रहा', 'हुए', 'थे', 
        'करें', 'इसके', 'थी', 'उस', 'हूँ', 'जा', 'ना', 'उन', 'वह', 'भी', 'वे', 'थी', 'जब', 'होते', 'कोई', 
        'हम', 'आप', 'फिर', 'बहुत', 'कहा', 'वाले', 'जैसे', 'सभी', 'कुछ', 'क्या', 'अब', 'उनके', 'इसी', 'रहे', 
        'उनकी', 'उनका', 'अपनी', 'उसके', 'तथा', 'दो', 'वहां', 'गये', 'बड़े', 'वर्ग', 'तरह', 'रही', 'किसी', 
        'ऐसे', 'रखें', 'अपना', 'उसे', 'जिसमें', 'किन्हें', 'रूप', 'किन्होंने', 'किया', 'लेकिन', 'कम', 'होती', 
        'अधिक', 'अब', 'और', 'वर्ष', 'यदि', 'हुये', 'इसलिए', 'रखा', 'किये', 'अन्य', 'भाग', 'उन्हें', 'गयी', 
        'प्रति', 'कुल', 'एस', 'रहती', 'इसमें', 'जिस', 'प्रकार', 'आदि', 'इन', 'अभी', 'आज', 'कल', 'जिन्हें', 
        'जिन्होंने', 'तब', 'उसकी', 'उसका', 'यहाँ', 'इसकी', 'सकती', 'इसे', 'जिसके', 'सबसे', 'होने', 'बात', 
        'यही', 'वही', 'दिन', 'कहते', 'अपनी', 'कई', 'तरफ', 'बाद', 'लिए', 'रख', 'रखी', 'उन्होंने', 'वहीं', 
        'उन्हीं', 'जा', 'जाता', 'जाती', 'बाहर', 'आ', 'आता', 'आती', 'वाला', 'वाली', 'हर', 'हर', 'जाए', 
        'जाएगा', 'जाएँगे', 'जाओ', 'आओ', 'आएगा', 'आएँगे', 'वहाँ', 'जहाँ', 'वग़ैरह', 'नीचे', 'ऊपर', 'वाले', 
        'सारे', 'सारी', 'अंदर', 'माना', 'मानी', 'मानो', 'अच्छा', 'अच्छी', 'अच्छे', 'ले', 'लो', 'दे', 'दो', 
        'उसको', 'उसकी', 'उसके', 'उससे', 'उसने', 'उसमें', 'उसी', 'उन्हीं', 'उन्हें', 'उनसे', 'उनको', 'उनमें'
    }
    
    # Assamese stopwords (minimal curated list)
    stopword_dict['assamese'] = {
        'আৰু', 'এই', 'এটা', 'এনে', 'তাৰ', 'নাই', 'হয়', 'হৈ', 'হল', 'হব', 'কৰি', 'কৰা', 'কৰে', 'কৰিব', 
        'কৰিবা', 'কৰিলে', 'নকৰে', 'নকৰা', 'আছে', 'আছিল', 'থাকে', 'থাকিব', 'যায়', 'যাব', 'যাওক', 'আহে', 
        'আহিব', 'আহিবা', 'পাৰে', 'পাৰিব', 'লাগে', 'লাগিব', 'লাগিল', 'হোৱা', 'হোৱাৰ', 'হোৱাই', 'তেওঁ', 
        'তেওঁৰ', 'তেওঁক', 'মই', 'মোক', 'মোৰ', 'আমি', 'আমাক', 'আমাৰ', 'তুমি', 'তোমাক', 'তোমাৰ', 'তেওঁলোক', 
        'তেওঁলোকৰ', 'তেওঁলোকক', 'কি', 'কিয়', 'কেনে', 'কেনেকৈ', 'য়াৰ', 'কত', 'কিমান', 'কোন', 'কোনে', 
        'যি', 'যিয়ে', 'যাক', 'যাৰ', 'যিহেতু', 'যিখিনি', 'যিমান', 'যিটো', 'যিবোৰ', 'যত', 'যতবোৰ', 'যথা', 
        'যদি', 'যদিও', 'যদিহে', 'যাতে', 'যিহত', 'সেই', 'সেইটো', 'সেইবোৰ', 'তেনে', 'তেনেকুৱা', 'তেতিয়া', 
        'তাত', 'তাক', 'তাৰ', 'তাই', 'তিনি', 'তিনিওটা', 'তিনিটা', 'সি', 'সিহঁত', 'সিহঁতৰ', 'সিহঁতক', 
        'আৰু', 'বা', 'বাৰু', 'নতুবা', 'যাতে', 'তথাপি', 'কিন্তু', 'যদিও', 'তেন্তে', 'তেতিয়াহলে', 'নাইবা', 
        'নহলে', 'নহয়', 'নাইকিয়া', 'নোহোৱা', 'নোহোৱাকৈ', 'নোহোৱালৈকে', 'হে', 'হয়', 'হয়তো', 'হবলা', 
        'হলে', 'হলো', 'হৈছে', 'হৈছিল', 'হৈ', 'হোৱা', 'হোৱাই', 'হোৱাত', 'হোৱাৰ', 'নহয়', 'নহব', 'নহল', 
        'নহে', 'নোহোৱা', 'নাছিল', 'নাই', 'নাইকিয়া', 'নিচিনা', 'নিচিনে', 'কৰক', 'কৰা', 'কৰি', 'কৰিব', 
        'কৰিবলৈ', 'কৰিবা', 'কৰিবে', 'কৰিম', 'কৰিয়ে', 'কৰিলে', 'কৰিলেই', 'কৰিলেও', 'কৰিলোঁ', 'কৰে', 
        'কৰো', 'কৰোঁ', 'কৰোঁতে', 'কৰোঁতেই', 'নকৰা', 'নকৰিব', 'নকৰিবা', 'নকৰিলে', 'নকৰে', 'নকৰো', 
        'নকৰোঁ', 'কৰাওক', 'কৰাই', 'কৰাইছে', 'কৰাইছিল', 'কৰাব', 'কৰাবা', 'কৰাবে', 'কৰাম', 'কৰালে', 
        'কৰালেও', 'কৰাৱ', 'কৰি', 'কৰিছিল', 'কৰিছে', 'কৰিছো', 'কৰিছোঁ', 'কৰিব', 'কৰিবই', 'কৰিবলৈ', 
        'কৰিবা', 'কৰিবি', 'কৰিম', 'কৰিয়ে', 'কৰিয়েই', 'কৰিলে', 'কৰিলেই', 'কৰিলেও', 'কৰিলোঁ', 'কৰো', 
        'কৰোঁ', 'কৰোঁতে', 'কৰোঁতেই', 'কৰোৱা', 'কৰোৱাই', 'কৰোৱাত', 'কৰোৱাৰ', 'এই', 'এইখিনি', 'এইজন', 
        'এইটো', 'এইবোৰ', 'এইসকল', 'এওঁ', 'এওঁলোক', 'এনে', 'এনেকুৱা', 'এৰা', 'এৰি', 'ওপৰত', 'ওলাই', 
        'ওলোৱা', 'কৰা', 'কৰি', 'কৰিব', 'কৰিবলৈ', 'কৰে', 'কাৰণে', 'কিন্তু', 'কিয়', 'কেতিয়া', 'কেতিয়াবা', 
        'কেৱল', 'কোনো', 'গৈ', 'চাই', 'চালে', 'চোৱা', 'ছয়', 'জন', 'জনা', 'জনি', 'জোন', 'তাই', 'তাক', 'তাত', 
        'তাৰ', 'তেওঁ', 'তেওঁলোক', 'তেতিয়া', 'তেন্তে', 'তোমালোক', 'থকা', 'থাকে', 'থাকিব', 'দিছে', 'দিয়ে', 
        'দিয়া', 'দিলে', 'দুই', 'দুয়ো', 'দেখা', 'দেখি', 'নকৰে', 'নকৰিব', 'নকৰিবা', 'নকৰিলে', 'নতুবা', 
        'নহয়', 'নাই', 'নাইবা', 'নিজৰ', 'নিজে', 'নিজেই', 'পৰা', 'পাঁচ', 'পাই', 'পাছত', 'পাৰ', 'পাৰে', 
        'পাৰিব', 'বুলি', 'বোলা', 'বোলে', 'ভিতৰত', 'যদি', 'যদিও', 'যাওক', 'যাব', 'যায়', 'যাৰ', 'যিয়ে', 
        'যিসকল', 'যোৱা', 'লগত', 'লাগে', 'লাগিব', 'লাগিল', 'লোৱা', 'শেষত', 'সকলো', 'সময়ত', 'সাতে', 
        'হওক', 'হব', 'হবই', 'হবলৈ', 'হলে', 'হলেও', 'হলো', 'হাতত', 'হিচাপে', 'হৈ', 'হৈছে', 'হৈছিল', 
        'হোৱা', 'হোৱাই', 'হোৱাত', 'হোৱাৰ', 'হয়', 'হয়তো'
    }
    
    # Manipuri stopwords (minimal curated list)
    stopword_dict['manipuri'] = {
        'ꯑꯗꯨ', 'ꯑꯁꯤ', 'ꯑꯗꯣꯝ', 'ꯑꯗꯣꯝꯒꯤ', 'ꯑꯗꯣꯝꯒꯤꯗꯝꯛ', 'ꯑꯗꯣꯝꯁꯨ', 'ꯑꯗꯣꯝꯅ', 'ꯑꯗꯣꯝꯅꯁꯨ', 'ꯑꯗꯣꯝꯗ', 'ꯑꯗꯣꯝꯗꯁꯨ', 
        'ꯑꯗꯣꯝꯗꯒꯤ', 'ꯑꯗꯣꯝꯗꯒꯤꯁꯨ', 'ꯑꯗꯣꯝꯗꯤ', 'ꯑꯗꯣꯝꯗꯨ', 'ꯑꯗꯣꯝꯅ', 'ꯑꯗꯣꯝꯅꯁꯨ', 'ꯑꯗꯣꯝꯅꯗꯤ', 'ꯑꯗꯣꯝꯅꯗꯨ', 'ꯑꯗꯣꯝꯅꯥ', 
        'ꯑꯗꯣꯝꯅꯥꯁꯨ', 'ꯑꯗꯣꯝꯅꯥꯗꯤ', 'ꯑꯗꯣꯝꯅꯥꯗꯨ', 'ꯑꯗꯣꯝꯁꯤ', 'ꯑꯗꯣꯝꯁꯨ', 'ꯑꯗꯨ', 'ꯑꯗꯨꯒ', 'ꯑꯗꯨꯒꯤ', 'ꯑꯗꯨꯒꯨꯝ', 'ꯑꯗꯨꯒꯨꯝꯕ', 
        'ꯑꯗꯨꯒꯨꯝꯕꯗ', 'ꯑꯗꯨꯒꯨꯝꯕꯗꯨ', 'ꯑꯗꯨꯒꯨꯝꯕꯅ', 'ꯑꯗꯨꯒꯨꯝꯕꯅꯗꯤ', 'ꯑꯗꯨꯒꯨꯝꯕꯅꯗꯨ', 'ꯑꯗꯨꯒꯨꯝꯕꯁꯤ', 'ꯑꯗꯨꯒꯨꯝꯕꯁꯨ', 'ꯑꯗꯨꯗ', 
        'ꯑꯗꯨꯗꯁꯨ', 'ꯑꯗꯨꯗꯒꯤ', 'ꯑꯗꯨꯗꯒꯤꯁꯨ', 'ꯑꯗꯨꯗꯤ', 'ꯑꯗꯨꯗꯨ', 'ꯑꯗꯨꯅ', 'ꯑꯗꯨꯅꯁꯨ', 'ꯑꯗꯨꯅꯗꯤ', 'ꯑꯗꯨꯅꯗꯨ', 'ꯑꯗꯨꯁꯤ', 'ꯑꯗꯨꯁꯨ', 
        'ꯑꯗꯧ', 'ꯑꯗꯧꯅ', 'ꯑꯅꯤ', 'ꯑꯄꯨꯡꯕ', 'ꯑꯃ', 'ꯑꯃꯁꯨꯡ', 'ꯑꯃꯠꯇ', 'ꯑꯃꯠꯇꯁꯨ', 'ꯑꯃꯠꯇꯗ', 'ꯑꯃꯠꯇꯗꯁꯨ', 'ꯑꯃꯠꯇꯅ', 
        'ꯑꯃꯠꯇꯅꯁꯨ', 'ꯑꯃꯠꯇꯁꯨ', 'ꯑꯃꯗꯤ', 'ꯑꯃꯥꯡꯕ', 'ꯑꯃꯨꯛ', 'ꯑꯃꯨꯛꯄꯨ', 'ꯑꯃꯨꯛꯁꯨ', 'ꯑꯃꯨꯛꯍꯟꯅ', 'ꯑꯃꯣꯝ', 'ꯑꯃꯣꯠ', 
        'ꯑꯃꯣꯠꯇ', 'ꯑꯃꯣꯠꯇꯁꯨ', 'ꯑꯃꯣꯠꯇꯗ', 'ꯑꯃꯣꯠꯇꯗꯁꯨ', 'ꯑꯃꯣꯠꯇꯅ', 'ꯑꯃꯣꯠꯇꯅꯁꯨ', 'ꯑꯃꯣꯠꯇꯁꯨ', 'ꯑꯩ', 'ꯑꯩꯒꯤ', 
        'ꯑꯩꯒꯤꯗꯝꯛ', 'ꯑꯩꯒꯤꯁꯨ', 'ꯑꯩꯒꯨꯝꯕ', 'ꯑꯩꯒꯨꯝꯕꯗ', 'ꯑꯩꯒꯨꯝꯕꯗꯨ', 'ꯑꯩꯒꯨꯝꯕꯅ', 'ꯑꯩꯒꯨꯝꯕꯅꯗꯤ', 'ꯑꯩꯒꯨꯝꯕꯅꯗꯨ', 'ꯑꯩꯒꯨꯝꯕꯁꯤ', 
        'ꯑꯩꯒꯨꯝꯕꯁꯨ', 'ꯑꯩꯁꯨ', 'ꯑꯩꯅ', 'ꯑꯩꯅꯁꯨ', 'ꯑꯩꯅꯗꯤ', 'ꯑꯩꯅꯗꯨ', 'ꯑꯩꯅꯥ', 'ꯑꯩꯅꯥꯁꯨ', 'ꯑꯩꯅꯥꯗꯤ', 'ꯑꯩꯅꯥꯗꯨ', 'ꯑꯩꯁꯤ', 
        'ꯑꯩꯁꯨ', 'ꯑꯔꯦꯝꯕ', 'ꯑꯔꯦꯝꯕꯗ', 'ꯑꯔꯦꯝꯕꯗꯨ', 'ꯑꯔꯦꯝꯕꯅ', 'ꯑꯔꯦꯝꯕꯅꯗꯤ', 'ꯑꯔꯦꯝꯕꯅꯗꯨ', 'ꯑꯔꯦꯝꯕꯁꯤ', 'ꯑꯔꯦꯝꯕꯁꯨ', 'ꯑꯍꯥꯟꯕ', 
        'ꯑꯍꯥꯟꯕꯗ', 'ꯑꯍꯥꯟꯕꯗꯨ', 'ꯑꯍꯥꯟꯕꯅ', 'ꯑꯍꯥꯟꯕꯅꯗꯤ', 'ꯑꯍꯥꯟꯕꯅꯗꯨ', 'ꯑꯍꯥꯟꯕꯁꯤ', 'ꯑꯍꯥꯟꯕꯁꯨ', 'ꯑꯍꯧꯕ', 'ꯑꯍꯧꯕꯗ', 
        'ꯑꯍꯧꯕꯗꯨ', 'ꯑꯍꯧꯕꯅ', 'ꯑꯍꯧꯕꯅꯗꯤ', 'ꯑꯍꯧꯕꯅꯗꯨ', 'ꯑꯍꯧꯕꯁꯤ', 'ꯑꯍꯧꯕꯁꯨ', 'ꯑꯍꯨꯝ', 'ꯑꯍꯨꯝꯁꯨꯕ', 'ꯑꯩꯍꯥꯛ', 'ꯑꯩꯍꯥꯛꯀꯤ', 
        'ꯑꯩꯍꯥꯛꯀꯤꯗꯝꯛ', 'ꯑꯩꯍꯥꯛꯀꯤꯁꯨ', 'ꯑꯩꯍꯥꯛꯁꯨ', 'ꯑꯩꯍꯥꯛꯅ', 'ꯑꯩꯍꯥꯛꯅꯁꯨ', 'ꯑꯩꯍꯥꯛꯅꯗꯤ', 'ꯑꯩꯍꯥꯛꯅꯗꯨ', 'ꯑꯩꯍꯥꯛꯅꯥ', 
        'ꯑꯩꯍꯥꯛꯅꯥꯁꯨ', 'ꯑꯩꯍꯥꯛꯅꯥꯗꯤ', 'ꯑꯩꯍꯥꯛꯅꯥꯗꯨ', 'ꯑꯩꯍꯥꯛꯁꯤ', 'ꯑꯩꯍꯥꯛꯁꯨ', 'ꯑꯩꯖꯣꯡ', 'ꯑꯩꯖꯣꯡꯒꯤ', 'ꯑꯩꯖꯣꯡꯒꯤꯗꯝꯛ', 
        'ꯑꯩꯖꯣꯡꯒꯤꯁꯨ', 'ꯑꯩꯖꯣꯡꯁꯨ', 'ꯑꯩꯖꯣꯡꯅ', 'ꯑꯩꯖꯣꯡꯅꯁꯨ', 'ꯑꯩꯖꯣꯡꯅꯗꯤ', 'ꯑꯩꯖꯣꯡꯅꯗꯨ', 'ꯑꯩꯖꯣꯡꯅꯥ', 'ꯑꯩꯖꯣꯡꯅꯥꯁꯨ', 
        'ꯑꯩꯖꯣꯡꯅꯥꯗꯤ', 'ꯑꯩꯖꯣꯡꯅꯥꯗꯨ', 'ꯑꯩꯖꯣꯡꯁꯤ', 'ꯑꯩꯖꯣꯡꯁꯨ', 'ꯑꯩꯇꯥ', 'ꯑꯩꯇꯥꯗ', 'ꯑꯩꯇꯥꯗꯁꯨ', 'ꯑꯩꯇꯥꯗꯒꯤ', 'ꯑꯩꯇꯥꯗꯒꯤꯁꯨ', 
        'ꯑꯩꯇꯥꯗꯤ', 'ꯑꯩꯇꯥꯗꯨ', 'ꯑꯩꯇꯥꯅ', 'ꯑꯩꯇꯥꯅꯁꯨ', 'ꯑꯩꯇꯥꯅꯗꯤ', 'ꯑꯩꯇꯥꯅꯗꯨ', 'ꯑꯩꯇꯥꯁꯤ', 'ꯑꯩꯇꯥꯁꯨ', 'ꯑꯩꯊꯧ', 'ꯑꯩꯊꯧꯗ', 
        'ꯑꯩꯊꯧꯗꯁꯨ', 'ꯑꯩꯊꯧꯗꯒꯤ', 'ꯑꯩꯊꯧꯗꯒꯤꯁꯨ', 'ꯑꯩꯊꯧꯗꯤ', 'ꯑꯩꯊꯧꯗꯨ', 'ꯑꯩꯊꯧꯅ', 'ꯑꯩꯊꯧꯅꯁꯨ', 'ꯑꯩꯊꯧꯅꯗꯤ', 'ꯑꯩꯊꯧꯅꯗꯨ', 
        'ꯑꯩꯊꯧꯁꯤ', 'ꯑꯩꯊꯧꯁꯨ', 'ꯑꯁꯤ', 'ꯑꯁꯤꯒ', 'ꯑꯁꯤꯒꯤ', 'ꯑꯁꯤꯒꯨꯝ', 'ꯑꯁꯤꯒꯨꯝꯕ', 'ꯑꯁꯤꯒꯨꯝꯕꯗ', 'ꯑꯁꯤꯒꯨꯝꯕꯗꯨ', 
        'ꯑꯁꯤꯒꯨꯝꯕꯅ', 'ꯑꯁꯤꯒꯨꯝꯕꯅꯗꯤ', 'ꯑꯁꯤꯒꯨꯝꯕꯅꯗꯨ', 'ꯑꯁꯤꯒꯨꯝꯕꯁꯤ', 'ꯑꯁꯤꯒꯨꯝꯕꯁꯨ', 'ꯑꯁꯤꯗ', 'ꯑꯁꯤꯗꯁꯨ', 
        'ꯑꯁꯤꯗꯒꯤ', 'ꯑꯁꯤꯗꯒꯤꯁꯨ', 'ꯑꯁꯤꯗꯤ', 'ꯑꯁꯤꯗꯨ', 'ꯑꯁꯤꯅ', 'ꯑꯁꯤꯅꯁꯨ', 'ꯑꯁꯤꯅꯗꯤ', 'ꯑꯁꯤꯅꯗꯨ', 'ꯑꯁꯤꯁꯤ', 'ꯑꯁꯤꯁꯨ', 
        'ꯑꯁꯧꯕ', 'ꯑꯁꯧꯕꯗ', 'ꯑꯁꯧꯕꯗꯨ', 'ꯑꯁꯧꯕꯅ', 'ꯑꯁꯧꯕꯅꯗꯤ', 'ꯑꯁꯧꯕꯅꯗꯨ', 'ꯑꯁꯧꯕꯁꯤ', 'ꯑꯁꯧꯕꯁꯨ', 'ꯑꯇꯩ', 'ꯑꯇꯩꯗ', 
        'ꯑꯇꯩꯗꯁꯨ', 'ꯑꯇꯩꯗꯒꯤ', 'ꯑꯇꯩꯗꯒꯤꯁꯨ', 'ꯑꯇꯩꯗꯤ', 'ꯑꯇꯩꯗꯨ', 'ꯑꯇꯩꯅ', 'ꯑꯇꯩꯅꯁꯨ', 'ꯑꯇꯩꯅꯗꯤ', 'ꯑꯇꯩꯅꯗꯨ', 'ꯑꯇꯩꯁꯤ', 
        'ꯑꯇꯩꯁꯨ', 'ꯑꯋꯥꯡꯕ', 'ꯑꯋꯥꯡꯕꯗ', 'ꯑꯋꯥꯡꯕꯗꯨ', 'ꯑꯋꯥꯡꯕꯅ', 'ꯑꯋꯥꯡꯕꯅꯗꯤ', 'ꯑꯋꯥꯡꯕꯅꯗꯨ', 'ꯑꯋꯥꯡꯕꯁꯤ', 'ꯑꯋꯥꯡꯕꯁꯨ', 
        'ꯑꯌꯥꯝꯕ', 'ꯑꯌꯥꯝꯕꯗ', 'ꯑꯌꯥꯝꯕꯗꯨ', 'ꯑꯌꯥꯝꯕꯅ', 'ꯑꯌꯥꯝꯕꯅꯗꯤ', 'ꯑꯌꯥꯝꯕꯅꯗꯨ', 'ꯑꯌꯥꯝꯕꯁꯤ', 'ꯑꯌꯥꯝꯕꯁꯨ', 'ꯑꯌꯨꯛ', 
        'ꯑꯌꯨꯛꯇ', 'ꯑꯌꯨꯛꯇꯁꯨ', 'ꯑꯌꯨꯛꯇꯒꯤ', 'ꯑꯌꯨꯛꯇꯒꯤꯁꯨ', 'ꯑꯌꯨꯛꯇꯤ', 'ꯑꯌꯨꯛꯇꯨ', 'ꯑꯌꯨꯛꯀꯤ', 'ꯑꯌꯨꯛꯀꯤꯁꯨ', 'ꯑꯌꯨꯛꯅ', 
        'ꯑꯌꯨꯛꯅꯁꯨ', 'ꯑꯌꯨꯛꯅꯗꯤ', 'ꯑꯌꯨꯛꯅꯗꯨ', 'ꯑꯌꯨꯛꯁꯤ', 'ꯑꯌꯨꯛꯁꯨ', 'ꯑꯔꯤꯕ', 'ꯑꯔꯤꯕꯗ', 'ꯑꯔꯤꯕꯗꯨ', 'ꯑꯔꯤꯕꯅ', 'ꯑꯔꯤꯕꯅꯗꯤ', 
        'ꯑꯔꯤꯕꯅꯗꯨ', 'ꯑꯔꯤꯕꯁꯤ', 'ꯑꯔꯤꯕꯁꯨ', 'ꯑꯍꯥꯟꯕ', 'ꯑꯍꯥꯟꯕꯗ', 'ꯑꯍꯥꯟꯕꯗꯨ', 'ꯑꯍꯥꯟꯕꯅ', 'ꯑꯍꯥꯟꯕꯅꯗꯤ', 'ꯑꯍꯥꯟꯕꯅꯗꯨ', 
        'ꯑꯍꯥꯟꯕꯁꯤ', 'ꯑꯍꯥꯟꯕꯁꯨ', 'ꯑꯍꯧꯕ', 'ꯑꯍꯧꯕꯗ', 'ꯑꯍꯧꯕꯗꯨ', 'ꯑꯍꯧꯕꯅ', 'ꯑꯍꯧꯕꯅꯗꯤ', 'ꯑꯍꯧꯕꯅꯗꯨ', 'ꯑꯍꯧꯕꯁꯤ', 
        'ꯑꯍꯧꯕꯁꯨ', 'ꯑꯍꯨꯝ', 'ꯑꯍꯨꯝꯁꯨꯕ', 'ꯑꯩꯍꯥꯛ', 'ꯑꯩꯍꯥꯛꯀꯤ', 'ꯑꯩꯍꯥꯛꯀꯤꯗꯝꯛ', 'ꯑꯩꯍꯥꯛꯀꯤꯁꯨ', 'ꯑꯩꯍꯥꯛꯁꯨ', 'ꯑꯩꯍꯥꯛꯅ', 
        'ꯑꯩꯍꯥꯛꯅꯁꯨ', 'ꯑꯩꯍꯥꯛꯅꯗꯤ', 'ꯑꯩꯍꯥꯛꯅꯗꯨ', 'ꯑꯩꯍꯥꯛꯅꯥ', 'ꯑꯩꯍꯥꯛꯅꯥꯁꯨ', 'ꯑꯩꯍꯥꯛꯅꯥꯗꯤ', 'ꯑꯩꯍꯥꯛꯅꯥꯗꯨ', 'ꯑꯩꯍꯥꯛꯁꯤ', 
        'ꯑꯩꯍꯥꯛꯁꯨ', 'ꯑꯩꯖꯣꯡ', 'ꯑꯩꯖꯣꯡꯒꯤ', 'ꯑꯩꯖꯣꯡꯒꯤꯗꯝꯛ', 'ꯑꯩꯖꯣꯡꯒꯤꯁꯨ', 'ꯑꯩꯖꯣꯡꯁꯨ', 'ꯑꯩꯖꯣꯡꯅ', 'ꯑꯩꯖꯣꯡꯅꯁꯨ', 
        'ꯑꯩꯖꯣꯡꯅꯗꯤ', 'ꯑꯩꯖꯣꯡꯅꯗꯨ', 'ꯑꯩꯖꯣꯡꯅꯥ', 'ꯑꯩꯖꯣꯡꯅꯥꯁꯨ', 'ꯑꯩꯖꯣꯡꯅꯥꯗꯤ', 'ꯑꯩꯖꯣꯡꯅꯥꯗꯨ', 'ꯑꯩꯖꯣꯡꯁꯤ', 'ꯑꯩꯖꯣꯡꯁꯨ'
    }
    
    # Bodo stopwords (currently empty, can be expanded)
    stopword_dict['bodo'] = set()
    
    return stopword_dict

# Clean text by removing punctuation and normalizing spaces
def clean_text(text):
    """Clean text by removing punctuation and normalizing spaces.
    Also performs Unicode normalization for Indic scripts.
    
    Args:
        text (str): Input text to clean.
        
    Returns:
        str: Cleaned text.
    """
    if not text:
        return ""
    
    # Unicode normalization (NFC form)
    text = unicodedata.normalize('NFC', text)
    
    # Convert to lowercase (for English only)
    text = text.lower()
    
    # Remove zero-width joiners and non-joiners
    text = text.replace('\u200C', '').replace('\u200D', '')
    
    # Unify punctuation (Devanagari danda to period)
    text = text.replace('।', '.')
    
    # Remove ASCII punctuation
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]', ' ', text)
    
    # Remove Unicode punctuation
    text = re.sub(r'[\u2000-\u206F\u2E00-\u2E7F\u3000-\u303F]', ' ', text)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Tokenize text based on language
def tokenize_text(text, lang):
    """Tokenize text based on language with proper handling for Indic scripts.
    
    Args:
        text (str): Input text to tokenize.
        lang (str): Language code ('english', 'hindi', 'assamese', 'manipuri').
        
    Returns:
        list: List of tokens.
    """
    if not text:
        return []
    
    try:
        if lang == 'english':
            # Use NLTK for English
            try:
                return word_tokenize(text)
            except:
                ensure_nltk_data()
                return word_tokenize(text)
        elif lang in ['hindi', 'assamese', 'manipuri', 'bodo']:
            # First normalize the text to NFC form
            text = unicodedata.normalize('NFC', text)
            
            # Use grapheme library if available (best option for Indic scripts)
            if GRAPHEME_AVAILABLE:
                # Split by whitespace and keep grapheme clusters together
                tokens = []
                for word in text.split():
                    # Join graphemes to ensure proper character clusters
                    tokens.append(''.join(grapheme.graphemes(word)))
                logger.info(f"Tokenized with grapheme library: {tokens[:5] if len(tokens) >= 5 else tokens}")
                return tokens
            
            # Fallback to indic_tokenize if available
            if INDIC_NLP_AVAILABLE:
                tokens = indic_tokenize.trivial_tokenize(text)
                logger.info(f"Tokenized with indic_tokenize: {tokens[:5] if len(tokens) >= 5 else tokens}")
                return tokens
            
            # Last resort fallback to regex-based tokenization
            tokens = regex.findall(r'\p{L}+', text)
            logger.info(f"Tokenized with regex: {tokens[:5] if len(tokens) >= 5 else tokens}")
            return tokens
        else:
            # Fallback to simple regex tokenization
            logger.warning(f"Using fallback tokenization for {lang}")
            return re.findall(r'\b\w+\b', text)
    except Exception as e:
        logger.error(f"Error in tokenization: {e}")
        # Fallback to simple regex tokenization
        return re.findall(r'\b\w+\b', text)

# Filter stopwords from tokens
def filter_stopwords(tokens, lang):
    """Filter stopwords from tokens.
    
    Args:
        tokens (list): List of tokens.
        lang (str): Language code ('english', 'hindi', 'assamese', 'manipuri').
        
    Returns:
        list: List of tokens with stopwords removed.
    """
    if not tokens:
        return []
    
    # Get stopwords for the language
    stopword_dict = load_stopwords()
    lang_stopwords = stopword_dict.get(lang, set())
    
    # Filter out stopwords and tokens with length < 2
    filtered_tokens = [token for token in tokens if token not in lang_stopwords and len(token) >= 2]
    
    return filtered_tokens

# Get word frequencies
def get_frequencies(tokens, top_n=20):
    """Get word frequencies.
    
    Args:
        tokens (list): List of tokens.
        top_n (int): Number of top frequencies to return.
        
    Returns:
        list: List of (word, count) tuples.
    """
    if not tokens:
        return []
    
    # Count word frequencies
    word_counts = Counter(tokens)
    
    # Get top N words
    top_words = word_counts.most_common(top_n)
    
    return top_words

# Get the correct font path based on language
def get_font_path(lang):
    """Get the correct font path based on the language.
    
    Args:
        lang (str): Language code.
        
    Returns:
        str: Path to the font file or None.
    """
    font_mapping = {
        'hindi': 'fonts/Noto_Sans_Devanagari/static/NotoSansDevanagari-Regular.ttf',
        'assamese': 'fonts/Noto_Sans_Bengali/static/NotoSansBengali-Regular.ttf',
        'manipuri': 'fonts/Noto_Sans_Meetei_Mayek/static/NotoSansMeeteiMayek-Regular.ttf',
        'bodo': 'fonts/Noto_Sans_Devanagari/static/NotoSansDevanagari-Regular.ttf'
    }
    
    font_path = font_mapping.get(lang)
    
    if font_path and os.path.exists(font_path):
        logger.info(f"Using font: {font_path}")
        return font_path
    
    logger.warning(f"Font for {lang} not found at {font_path}. Using default.")
    return None

# Generate wordcloud image
def generate_wordcloud_image(tokens, lang, width=800, height=400):
    """Generate wordcloud image.
    
    Args:
        tokens (list): List of tokens.
        lang (str): Language code.
        width (int): Width of the wordcloud image.
        height (int): Height of the wordcloud image.
        
    Returns:
        bytes: Image bytes for the wordcloud.
    """
    if not tokens:
        return None
    
    word_freq = Counter(tokens)
    font_path = get_font_path(lang)
    
    try:
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='#FDF4DC',
            font_path=font_path,
            min_font_size=10,
            max_font_size=150,
            colormap='copper',
            collocations=False
        )
        
        wordcloud.generate_from_frequencies(word_freq)
        
        img = wordcloud.to_image()
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        logger.info(f"WordCloud generated for {lang} with font: {font_path}")
        return img_bytes
        
    except Exception as e:
        logger.error(f"Error generating wordcloud: {e}")
        return None

# Plot frequency bar chart
def plot_frequency_bar(freq_list, lang='english'):
    """Plot frequency bar chart with proper font handling for Indic scripts.
    
    Args:
        freq_list (list): List of (word, count) tuples.
        lang (str): Language code.
        
    Returns:
        matplotlib.figure.Figure: Matplotlib figure object.
    """
    if not freq_list:
        return None
    
    font_path = get_font_path(lang)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    words = [item[0] for item in freq_list]
    counts = [item[1] for item in freq_list]
    
    x_pos = range(len(words))
    ax.bar(x_pos, counts, color='#7E6551')
    
    font_prop = None
    if font_path:
        font_prop = fm.FontProperties(fname=font_path)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(words, rotation=45, ha='right', fontproperties=font_prop)
    
    ax.set_xlabel('Words', color='#161616', fontproperties=font_prop)
    ax.set_ylabel('Frequency', color='#161616')
    ax.set_title('Word Frequency Distribution', color='#7E6551', fontproperties=font_prop)
    
    plt.yticks(color='#161616')
    plt.tight_layout()
    
    logger.info(f"Bar chart generated for {lang} with font: {font_path}")
    
    return fig

# Setup Streamlit UI
def setup_ui():
    """Setup Streamlit UI."""
    # Set page config
    st.set_page_config(
        page_title="Multilingual WordCloud Generator",
        page_icon="📊",
        layout="wide"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main {
        background-color: #FDF4DC;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #7E6551;
    }
    .stButton>button {
        background-color: #7E6551;
        color: white;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        color: white !important;
        background-color: rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("Multilingual WordCloud Generator")
    st.markdown("Generate word clouds and frequency charts for text in multiple languages.")
    
    # Input section
    st.header("Text Input")
    text_input = st.text_area("Enter your text here:", height=150)
    
    # Language selection
    language_options = {
        'english': 'English',
        'hindi': 'Hindi',
        'assamese': 'Assamese',
        'manipuri': 'Manipuri',
        'bodo': 'Bodo'
    }
    selected_lang = st.selectbox("Select language:", list(language_options.keys()), format_func=lambda x: language_options[x])
    
    # Number of top words to display
    top_n = st.slider("Number of top words to display:", min_value=5, max_value=50, value=20)
    
    # Generate button
    generate_button = st.button("Generate")
    
    # Process text when button is clicked
    if generate_button and text_input:
        try:
            # Clean and tokenize text
            original_text = text_input
            cleaned_text = clean_text(text_input)
            tokens = tokenize_text(cleaned_text, selected_lang)
            filtered_tokens = filter_stopwords(tokens, selected_lang)
            
            # Check if we have tokens after filtering
            if not filtered_tokens:
                st.warning("No valid tokens found after filtering. Try a different text or language.")
                return
            
            # Get word frequencies
            freq_list = get_frequencies(filtered_tokens, top_n)
            
            # Create two columns for output
            col1, col2 = st.columns(2)
            
            # Display bar chart in first column
            with col1:
                st.subheader("Word Frequency Chart")
                fig = plot_frequency_bar(freq_list, selected_lang)
                if fig:
                    st.pyplot(fig)
                    # Save figure to a bytes buffer for download
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    st.download_button(
                        label="Download Bar Chart",
                        data=buf,
                        file_name=f"bar_chart_{selected_lang}.png",
                        mime="image/png"
                    )
                else:
                    st.warning("Could not generate frequency chart.")
            
            # Display wordcloud in second column
            with col2:
                st.subheader("Word Cloud")
                wordcloud_bytes = generate_wordcloud_image(filtered_tokens, selected_lang)
                if wordcloud_bytes:
                    st.image(wordcloud_bytes, caption="Word Cloud", use_container_width=True)
                    st.download_button(
                        label="Download WordCloud",
                        data=wordcloud_bytes,
                        file_name=f"wordcloud_{selected_lang}.png",
                        mime="image/png"
                    )
                else:
                    st.warning("Could not generate wordcloud.")

            # Add diagnostics section
            with st.expander("Diagnostics Information"):
                st.subheader("Text Processing Diagnostics")
                
                # Display normalization comparison
                st.write("**Unicode Normalization:**")
                st.write(f"Form used: NFC (Normalization Form Canonical Composition)")
                
                # Display sample of original vs normalized text
                st.write("**Sample Text Comparison:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original Text (first 100 chars):")
                    st.text(original_text[:100] + ("..." if len(original_text) > 100 else ""))
                with col2:
                    st.write("Normalized Text (first 100 chars):")
                    st.text(cleaned_text[:100] + ("..." if len(cleaned_text) > 100 else ""))
                
                # Display tokenization results
                st.write("**Tokenization Results:**")
                st.write(f"First 20 tokens: {tokens[:20]}")
                
                # Display frequency information
                st.write("**Word Frequencies:**")
                st.write(f"Top {min(20, len(freq_list))} words:")
                freq_df = pd.DataFrame(freq_list[:20], columns=["Word", "Frequency"])
                st.dataframe(freq_df)
                
                # Display font information
                st.write("**Font Information:**")
                font_dir = None
                if selected_lang != 'english':
                    font_dir_mapping = {
                        'hindi': 'fonts/Noto_Sans_Devanagari',
                        'assamese': 'fonts/Noto_Sans_Bengali',
                        'manipuri': 'fonts/Noto_Sans_Meetei_Mayek',
                        'bodo': 'fonts/Noto_Sans_Devanagari'
                    }
                    font_dir = font_dir_mapping.get(selected_lang)
                st.write(f"Language: {selected_lang}")
                st.write(f"Font directory: {font_dir}")
                
                # Technical details
                st.write("**Technical Details:**")
                st.write(f"Total tokens before filtering: {len(tokens)}")
                st.write(f"Total tokens after filtering: {len(filtered_tokens)}")
                st.write(f"Unique words: {len(set(filtered_tokens))}")
                
        except Exception as e:
            st.error(f"Error processing text: {str(e)}")
            logger.error(f"Error processing text: {e}", exc_info=True)
    
    # Display instructions if no text is entered
    elif generate_button and not text_input:
        st.warning("Please enter some text to generate the wordcloud.")

# Main function
def main():
    # Ensure NLTK data is downloaded
    ensure_nltk_data()
    
    # Setup UI
    setup_ui()

# Run the app
if __name__ == "__main__":
    main()
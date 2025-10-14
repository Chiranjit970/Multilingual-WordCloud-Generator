import sys
import os
import pytest
from io import BytesIO

# Add parent directory to path to import app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from app.py
from app import (
    clean_text,
    tokenize_text,
    filter_stopwords,
    get_frequencies,
    generate_wordcloud_image,
    load_stopwords
)

# Test text cleaning
def test_clean_text():
    text = "Hello, World! This is a test."
    cleaned = clean_text(text)
    assert cleaned == "hello world this is a test"
    
    # Test with empty input
    assert clean_text("") == ""
    assert clean_text(None) == ""

# Test English tokenization
def test_tokenize_english():
    text = "Data science is the future. Science helps us understand data better."
    tokens = tokenize_text(clean_text(text), "english")
    assert len(tokens) > 0
    assert "data" in [t.lower() for t in tokens]
    assert "science" in [t.lower() for t in tokens]
    
    # Test with empty input
    assert tokenize_text("", "english") == []

# Test Hindi tokenization
def test_tokenize_hindi():
    text = "डेटा विज्ञान भविष्य है और यह हमें बेहतर समझने में मदद करता है।"
    tokens = tokenize_text(clean_text(text), "hindi")
    assert len(tokens) > 0
    
    # Check if some expected tokens are present
    expected_tokens = ["डेटा", "विज्ञान", "भविष्य"]
    found = False
    for expected in expected_tokens:
        if expected in tokens:
            found = True
            break
    assert found, "None of the expected Hindi tokens were found"

# Test Assamese tokenization
def test_tokenize_assamese():
    text = "ডাটা বিজ্ঞান আমাৰ ভৱিষ্যৎ। বিজ্ঞান আমাৰ জীৱন সহজ কৰে।"
    tokens = tokenize_text(clean_text(text), "assamese")
    assert len(tokens) > 0
    
    # Check if some expected tokens are present
    expected_tokens = ["ডাটা", "বিজ্ঞান", "ভৱিষ্যৎ"]
    found = False
    for expected in expected_tokens:
        if expected in tokens:
            found = True
            break
    assert found, "None of the expected Assamese tokens were found"

# Test Manipuri tokenization
def test_tokenize_manipuri():
    # This is a simplified test since we may not have proper Manipuri text rendering
    text = "ꯗꯥꯇꯥ ꯁꯥꯏꯟꯁ ꯑꯁꯤ ꯃꯇꯨꯡ ꯀꯥꯜꯒꯤ ꯑꯣꯢꯕ ꯑꯃꯅꯤ"
    tokens = tokenize_text(clean_text(text), "manipuri")
    assert len(tokens) > 0

# Test stopwords removal
def test_stopwords_removed():
    # Test English stopwords
    english_text = "This is a test of the stopword removal function"
    english_tokens = tokenize_text(clean_text(english_text), "english")
    filtered_tokens = filter_stopwords(english_tokens, "english")
    
    # Common English stopwords should be removed
    assert "is" not in filtered_tokens
    assert "a" not in filtered_tokens
    assert "of" not in filtered_tokens
    assert "the" not in filtered_tokens
    
    # Non-stopwords should remain
    assert "test" in filtered_tokens
    assert "stopword" in filtered_tokens
    assert "removal" in filtered_tokens
    assert "function" in filtered_tokens
    
    # Test with empty input
    assert filter_stopwords([], "english") == []

# Test word frequency counting
def test_word_frequencies():
    tokens = ["apple", "banana", "apple", "cherry", "banana", "apple"]
    freq_list = get_frequencies(tokens, top_n=3)
    
    # Check that we get the right number of items
    assert len(freq_list) == 3
    
    # Check that the most frequent word is first
    assert freq_list[0][0] == "apple"
    assert freq_list[0][1] == 3
    
    # Check the second most frequent word
    assert freq_list[1][0] == "banana"
    assert freq_list[1][1] == 2
    
    # Test with empty input
    assert get_frequencies([], top_n=5) == []

# Test wordcloud generation
def test_wordcloud_generate():
    tokens = ["data", "science", "python", "machine", "learning", 
              "artificial", "intelligence", "data", "science", "data"]
    
    # Generate wordcloud for English
    wordcloud_bytes = generate_wordcloud_image(tokens, "english")
    
    # Check that we got a non-empty result
    assert wordcloud_bytes is not None
    assert isinstance(wordcloud_bytes, BytesIO)
    
    # Check that the image has content
    image_data = wordcloud_bytes.getvalue()
    assert len(image_data) > 0
    
    # Test with empty input
    assert generate_wordcloud_image([], "english") is None

# Test stopwords loading
def test_stopwords_loading():
    stopword_dict = load_stopwords()
    
    # Check that we have stopwords for all languages
    assert "english" in stopword_dict
    assert "hindi" in stopword_dict
    assert "assamese" in stopword_dict
    assert "manipuri" in stopword_dict
    
    # Check that each language has a non-empty set of stopwords
    assert len(stopword_dict["english"]) > 0
    assert len(stopword_dict["hindi"]) > 0
    assert len(stopword_dict["assamese"]) > 0
    assert len(stopword_dict["manipuri"]) > 0
    
    # Check some specific stopwords
    assert "the" in stopword_dict["english"]
    assert "और" in stopword_dict["hindi"]
    assert "আৰু" in stopword_dict["assamese"]
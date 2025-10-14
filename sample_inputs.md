# Sample Inputs and Expected Tokens

This document provides sample texts in different languages and their expected tokens after processing (cleaning, tokenization, and stopword removal).

## English

### Sample Text
```
Data science is the future. Science helps us understand data better.
```

### Expected Tokens
```
data, science, future, helps, understand, better
```

## Hindi

### Sample Text
```
डेटा विज्ञान भविष्य है और यह हमें बेहतर समझने में मदद करता है।
```

### Expected Tokens
```
डेटा, विज्ञान, भविष्य, बेहतर, समझने
```

## Assamese

### Sample Text
```
ডাটা বিজ্ঞান আমাৰ ভৱিষ্যৎ। বিজ্ঞান আমাৰ জীৱন সহজ কৰে।
```

### Expected Tokens
```
ডাটা, বিজ্ঞান, ভৱিষ্যৎ, জীৱন, সহজ
```

## Manipuri (Meetei Mayek)

### Sample Text
```
ꯗꯥꯇꯥ ꯁꯥꯏꯟꯁ ꯑꯁꯤ ꯃꯇꯨꯡ ꯀꯥꯜꯒꯤ ꯑꯣꯢꯕ ꯑꯃꯅꯤ꯫ ꯁꯥꯏꯟꯁꯅ ꯑꯩꯈꯣꯏꯗ ꯗꯥꯇꯥ ꯐꯖꯅ ꯈꯪꯍꯟꯕꯗ ꯃꯇꯦꯡ ꯄꯥꯡꯢ꯫
```

### Expected Tokens
```
ꯗꯥꯇꯥ, ꯁꯥꯏꯟꯁ, ꯃꯇꯨꯡ, ꯀꯥꯜꯒꯤ, ꯑꯣꯢꯕ, ꯐꯖꯅ, ꯈꯪꯍꯟꯕꯗ, ꯃꯇꯦꯡ, ꯄꯥꯡꯢ
```

## Processing Steps

1. **Text Cleaning**:
   - Convert to lowercase (for English)
   - Remove punctuation
   - Normalize spaces

2. **Tokenization**:
   - English: NLTK word_tokenize
   - Hindi/Assamese/Manipuri: indic_tokenize.trivial_tokenize or fallback to regex

3. **Stopword Removal**:
   - Remove language-specific stopwords
   - Remove tokens with length < 2 characters

## Notes

- Actual tokens may vary slightly depending on the tokenization method used
- If indic-nlp-library is not available, the application will fall back to regex tokenization
- The stopword lists are minimal and curated for demonstration purposes
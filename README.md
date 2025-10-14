# Multilingual WordCloud Generator

A Streamlit application that generates word clouds and frequency charts for text in multiple languages (Assamese, Hindi, Manipuri, and English).

## Features

- Text input in multiple languages (Assamese, Hindi, Manipuri, English)
- Language selection for appropriate text processing
- Text cleaning and tokenization
- Stopword removal
- Word frequency bar chart visualization
- WordCloud generation with appropriate fonts
- Downloadable WordCloud images

## Installation

1. Clone this repository or download the source code

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download NLTK data (this will happen automatically on first run, but you can do it manually):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

4. Download and place the required fonts in the `fonts/` directory:

- For Hindi: [NotoSansDevanagari-Regular.ttf](https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari)
- For Assamese: [NotoSansBengali-Regular.ttf](https://fonts.google.com/noto/specimen/Noto+Sans+Bengali)
- For Manipuri: [NotoSansMeeteiMayek-Regular.ttf](https://fonts.google.com/noto/specimen/Noto+Sans+Meetei+Mayek)

You can download these fonts from the Google Noto Fonts website. Place them in the `fonts/` directory of the project.

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

3. Enter text in the text area, select the appropriate language, and click "Generate"

4. View the word frequency chart and wordcloud

5. Download the wordcloud image using the "Download WordCloud" button

## Sample Inputs

See `sample_inputs.md` for example texts in different languages and their expected tokens after processing.

## Testing

Run the tests using pytest:

```bash
pytest tests/test_app.py
```

## Troubleshooting

### Font Issues

If you encounter missing glyphs or incorrect rendering:

1. Ensure you've downloaded the correct font files and placed them in the `fonts/` directory
2. Check that the font files are named exactly as expected by the application
3. If fonts are still not loading, the application will fall back to system fonts

### indic-nlp-library Issues

If you encounter issues with the indic-nlp-library:

1. The application will fall back to regex-based tokenization
2. Check the console logs for specific error messages
3. Try reinstalling the library: `pip install indic-nlp-library`

### NLTK Data Issues

If NLTK data is not downloading automatically:

1. Run the manual download command: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`
2. Check your internet connection
3. If issues persist, download the data manually from the NLTK Downloader UI

## Privacy and Security

- No data is stored remotely
- Temporary files are stored in memory or in the `./temp/` directory and are deleted after use

## License

This project is open source and available under the MIT License.
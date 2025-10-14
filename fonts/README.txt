# Font Files

This directory should contain the following font files or folders for proper rendering of Indic scripts:

## Option 1: Individual Font Files
1. NotoSansDevanagari-Regular.ttf (for Hindi)
2. NotoSansBengali-Regular.ttf (for Assamese)
3. NotoSansMeeteiMayek-Regular.ttf (for Manipuri)

## Option 2: Font Folders
Alternatively, you can place the downloaded font folders directly in this directory:
1. Noto_Sans_Devanagari/ (for Hindi)
2. Noto_Sans_Bengali/ (for Assamese)
3. Noto_Sans_Meetei_Mayek/ (for Manipuri)

The application will automatically search these folders for the appropriate Regular.ttf files.

## Download Instructions

You can download these fonts from the Google Noto Fonts website:

- Noto Sans Devanagari: https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari
- Noto Sans Bengali: https://fonts.google.com/noto/specimen/Noto+Sans+Bengali
- Noto Sans Meetei Mayek: https://fonts.google.com/noto/specimen/Noto+Sans+Meetei+Mayek

After downloading, you can either:
1. Extract the individual TTF files and rename them as specified in Option 1, OR
2. Place the entire downloaded font folders in this directory as specified in Option 2

## Fallback Behavior

If the required font files are not found, the application will attempt to use system fonts. However, this may result in missing glyphs or incorrect rendering for Indic scripts.
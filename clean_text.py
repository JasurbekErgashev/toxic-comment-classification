import nltk

nltk.download("stopwords")

import re
import contractions
from nltk.corpus import stopwords

stop = set(stopwords.words("english"))


# Function which performs text cleaning
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Expand contractions
    text = contractions.fix(text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7f]", "", text)
    # Remove special characters, including symbols, emojis, and other graphic characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    # Remove HTML
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    text = re.sub(html, "", text)
    # Remove escape characters
    text = re.sub(r"[\n\t\r\a]", " ", text)
    # Replacing "" with "
    text = re.sub(r"\"\"", '"', text)
    # Removing quotation from start and the end of the string
    text = re.sub(r"^\"", "", text)
    text = re.sub(r"\"$", "", text)
    # Removing Punctuation / Special characters (;:'".?@!%&*+) which appears more than twice in the text
    text = re.sub(r"[^a-zA-Z0-9\s][^a-zA-Z0-9\s]+", " ", text)
    # Removing Special characters
    text = re.sub(r"[^a-zA-Z0-9\s\"\',:;?!.()]", " ", text)
    # Removing extra spaces in text
    text = re.sub(r"\s\s+", " ", text)
    # Remove stop words
    text = " ".join(word for word in text.split() if word not in stop)
    return text

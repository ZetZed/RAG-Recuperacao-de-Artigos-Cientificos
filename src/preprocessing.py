import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure nltk stopwords are downloaded
try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))

# Common regex for english tokenization (words longer than 2 characters with optional hyphens)
TOKEN_RE = re.compile(r"\b[a-zA-Z][a-zA-Z\-]{1,}\b")

class TextPreprocessor:
    def __init__(self, mode="simple"):
        """
        Configures the preprocessor.
        mode options:
          - "simple": lowercase, remove punctuation/regex, remove stopwords
          - "stem": simple + Porter Stemmer
        """
        self.mode = mode
        self.stemmer = PorterStemmer() if mode == "stem" else None

    def preprocess(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Lowercase
        text = text.lower()
        
        # Tokenize using regex
        tokens = TOKEN_RE.findall(text)
        
        # Filter stopwords and short tokens
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
        
        # Stem if requested
        if self.mode == "stem":
            tokens = [self.stemmer.stem(t) for t in tokens]
            
        return tokens

# Default simple instance for convenience
_default_preprocessor = TextPreprocessor(mode="simple")

def tokenize(text: str) -> list[str]:
    return _default_preprocessor.preprocess(text)

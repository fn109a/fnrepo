import streamlit as st
import pandas as pd
import requests
import json
import time
import string
import re
from collections import Counter
import nltk
import os

# App title and description
st.set_page_config(page_title="Google Autocomplete Analyzer", layout="wide")
st.title("Google Autocomplete Keyword Clustering Tool")
st.markdown("Upload a CSV file with keywords to discover related search terms and cluster them.")

# Download NLTK resources (will only download if not already present)
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    return True

# Load the resources
resources_loaded = download_nltk_resources()

# Import NLTK components
from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

# Function to choose the appropriate tokenizer
@st.cache_resource
def get_tokenizer():
    try:
        test_tokenize = word_tokenize("This is a test.")
        def tokenize_text(text):
            return word_tokenize(text)
    except Exception:
        try:
            test_tokenize = wordpunct_tokenize("This is a test.")
            def tokenize_text(text):
                return wordpunct_tokenize(text)
        except Exception:
            word_tokenizer = RegexpTokenizer(r'\w+')
            def tokenize_text(text):
                return word_tokenizer.tokenize(text)
    return tokenize_text

tokenize_text = get_tokenizer()

# Get stopwords based on language
@st.cache_data
def get_stopwords(language_code):
    if language_code.lower() == 'it':
        # Italian stopwords from NLTK
        italian_stopwords = set(stopwords.words('italian'))
        
        # Additional Italian stopwords
        additional_italian_stopwords = {
            'a', 'adesso', 'ai', 'al', 'alla', 'allo', 'allora', 'altre', 'altri', 'altro', 'anche',
            'ancora', 'avere', 'aveva', 'ben', 'buono', 'che', 'chi', 'cinque', 'comprare', 'con',
            'consecutivi', 'consecutivo', 'cosa', 'cui', 'da', 'del', 'della', 'dello', 'dentro', 'deve',
            'devo', 'di', 'doppio', 'due', 'e', 'ecco', 'fare', 'fine', 'fino', 'fra', 'gente', 'giù',
            'ha', 'hai', 'hanno', 'ho', 'il', 'indietro', 'invece', 'io', 'la', 'lavoro', 'le', 'lei',
            'lo', 'loro', 'lui', 'lungo', 'ma', 'me', 'meglio', 'molta', 'molti', 'molto', 'nei', 'nella',
            'no', 'noi', 'nome', 'nostro', 'nove', 'nuovi', 'nuovo', 'o', 'oltre', 'ora', 'otto',
            'peggio', 'però', 'persone', 'più', 'poco', 'primo', 'promesso', 'qua', 'quarto', 'quasi',
            'quattro', 'quello', 'questo', 'qui', 'quindi', 'quinto', 'rispetto', 'sarà', 'secondo',
            'sei', 'sembra', 'sembrava', 'senza', 'sette', 'sia', 'siamo', 'siete', 'solo', 'sono',
            'sopra', 'soprattutto', 'sotto', 'stati', 'stato', 'stesso', 'su', 'subito', 'sul', 'sulla',
            'tanto', 'te', 'tempo', 'terzo', 'tra', 'tre', 'triplo', 'ultimo', 'un', 'una', 'uno', 'va',
            'vai', 'voi', 'volte', 'vostro'
        }
        
        return italian_stopwords.union(additional_italian_stopwords)
    else:
        # Basic English stopwords
        basic_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        try:
            english_nltk = set(stopwords.words('english'))
            return basic_stopwords.union(english_nltk)
        except:
            return basic_stopwords

def get_google_suggestions(keyword, lang_code, letterlist, progress_bar=None):
    """Fetch suggestions for a given keyword and language from Google Suggest."""
    suggestions = []
    headers = {'User-agent': 'Mozilla/5.0'}
    
    for i, letter in enumerate(letterlist):
        URL = f"http://suggestqueries.google.com/complete/search?client=firefox&hl={lang_code}&q={keyword} {letter}"
        try:
            response = requests.get(URL, headers=headers)
            result = json.loads(response.content.decode('utf-8'))
            if result:
                suggestions.extend(result[1])
            time.sleep(0.2)  # Rate limiting to be respectful
            
            # Update progress if progress bar is provided
            if progress_bar is not None:
                progress_bar.progress((i + 1) / len(letterlist))
                
        except Exception as e:
            st.warning(f"Error fetching suggestions for '{keyword} {letter}': {e}")
    
    return suggestions

def clean_and_cluster_suggestions(all_suggestions, stop_words, seed_words):
    """Clean suggestions using our selected tokenizer and remove stopwords."""
    wordlist = []
    for suggestion in all_suggestions:
        # Clean text before tokenizing
        clean_suggestion = re.sub(r'[^\w\s]', ' ', str(suggestion).lower())
        clean_suggestion = re.sub(r'\s+', ' ', clean_suggestion).strip()

        # Tokenize using our selected method
        words = tokenize_text(clean_suggestion)

        for word in words:
            if word not in stop_words and word not in seed_words and len(word) > 1:
                wordlist.append(word)
    return [word for word, count in Counter(wordlist).most_common(200)]

# User authentication (simple version)
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

# Main application
if check_password():
    # Create sidebar for settings
    st.sidebar.header("Settings")
    
    # Language selection
    language_code = st.sidebar.selectbox(
        "Select Language",
        options=["en", "it"],
        format_func=lambda x: "English" if x == "en" else "Italian",
        help="Choose the language for Google suggestions"
    )
    
    # Get appropriate stopwords
    stop_words = get_stopwords(language_code)
    st.sidebar.info(f"Using {len(stop_words)} stopwords for {language_code.upper()}")
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        batch_size = st.number_input("Batch Size", min_value=1, max_value=20, value=5, 
                                     help="Number of keywords to process in a batch")
        use_empty_letter = st.checkbox("Include empty prefix", value=True, 
                                       help="Include suggestions without prefixes")
        use_letters = st.checkbox("Include letter prefixes", value=True, 
                                  help="Include a-z letter prefixes")
    
    # Create letterlist based on settings
    letterlist = []
    if use_empty_letter:
        letterlist.append("")
    if use_letters:
        letterlist.extend(list(string.ascii_lowercase))
    
    if not letterlist:
        st.warning("Please select at least one prefix option in Advanced Settings")
    
    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload a CSV file with keywords", type=['csv'])
    
    if uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        
        # Display the dataframe
        st.subheader("Uploaded Keywords")
        st.dataframe(df)
        
        # Try to find a column that looks like keywords
        if 'keyword' in df.columns:
            keyword_col = 'keyword'
        elif 'Keyword' in df.columns:
            keyword_col = 'Keyword'
        else:
            potential_columns = [col for col in df.columns if 'key' in col.lower()]
            if potential_columns:
                keyword_col = potential_columns[0]
            else:
                keyword_col = df.columns[0]
        
        # Let user select the column with keywords
        keyword_col = st.selectbox("Select the column containing keywords", 
                                   options=df.columns.tolist(),
                                   index=df.columns.get_loc(keyword_col))
        
        # Extract keywords
        keywords = df[keyword_col].tolist()
        keywords = [k for k in keywords if isinstance(k, str) and not pd.isna(k)]
        
        # Display number of valid keywords
        st.info(f"Found {len(keywords)} valid keywords for analysis")
        
        # Start button to begin analysis
        if st.button("Start Analysis"):
            if len(letterlist) == 0:
                st.error("Please select at least one prefix option in Advanced Settings")
            else:
                # Initialize progress tracking
                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                # Process keywords in batches
                all_clusters = []
                
                for i in range(0, len(keywords), batch_size):
                    batch_keywords = keywords[i:i + batch_size]
                    progress_placeholder.text(f"Processing batch {i//batch_size + 1}/{(len(keywords)-1)//batch_size + 1}: {', '.join(batch_keywords)}")
                    
                    # Get seed words using our tokenizer
                    seed_words = []
                    for keyword in batch_keywords:
                        clean_keyword = re.sub(r'[^\w\s]', ' ', str(keyword).lower())
                        clean_keyword = re.sub(r'\s+', ' ', clean_keyword).strip()
                        tokens = tokenize_text(clean_keyword)
                        seed_words.extend(tokens)
                    
                    # Remove duplicates
                    seed_words = list(set(seed_words))
                    
                    # Get suggestions for each keyword in the batch
                    for idx, keyword in enumerate(batch_keywords):
                        subprogress = st.progress(0)
                        st.text(f"Getting suggestions for: {keyword}")
                        
                        suggestions = get_google_suggestions(keyword, language_code, letterlist, subprogress)
                        most_common_words = clean_and_cluster_suggestions(suggestions, stop_words, seed_words)
                        
                        # Assign suggestions and common words to their seed keyword
                        for common_word in most_common_words:
                            for suggestion in suggestions:
                                clean_suggestion = re.sub(r'[^\w\s]', ' ', str(suggestion).lower())
                                clean_suggestion = re.sub(r'\s+', ' ', clean_suggestion).strip()
                                
                                if common_word in clean_suggestion.split():
                                    all_clusters.append([suggestion, common_word, keyword])
                        
                        # Update main progress
                        progress_percent = (i + idx + 1) / len(keywords)
                        progress_bar.progress(min(progress_percent, 1.0))
                
                # Create and display results
                if all_clusters:
                    cluster_df = pd.DataFrame(all_clusters, columns=['Keyword', 'Cluster', 'Seed Keyword'])
                    
                    # Display results
                    st.subheader("Clustering Results")
                    st.dataframe(cluster_df)
                    
                    # Download button
                    csv = cluster_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f"keywords_clustered_{language_code}.csv",
                        mime="text/csv"
                    )
                    
                    # Display statistics
                    st.subheader("Statistics")
                    st.write(f"Total suggestions found: {len(cluster_df)}")
                    st.write(f"Unique clusters identified: {cluster_df['Cluster'].nunique()}")
                    
                    # Display clusters by seed keyword
                    st.subheader("Clusters by Seed Keyword")
                    for seed in cluster_df['Seed Keyword'].unique():
                        with st.expander(f"Clusters for '{seed}'"):
                            seed_df = cluster_df[cluster_df['Seed Keyword'] == seed]
                            st.write(f"Found {len(seed_df)} suggestions in {seed_df['Cluster'].nunique()} clusters")
                            st.dataframe(seed_df)
                else:
                    st.error("No clusters found. Try with different keywords or settings.")

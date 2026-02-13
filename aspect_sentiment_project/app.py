import streamlit as st
from preprocessing import clean_text
from aspect_extractor import extract_aspects
from sentiment_model import get_sentiment

def get_local_context(aspect, text):
    """
    Extract the clause/sentence containing the aspect for better context.
    Splits by 'but', 'and' to isolate aspect-specific sentiment.
    """
    # Try splitting by 'but' first (usually indicates contrast)
    parts = text.split(" but ")
    for part in parts:
        if aspect.lower() in part.lower():
            return part.strip()
    
    # Try splitting by 'and'
    parts = text.split(" and ")
    for part in parts:
        if aspect.lower() in part.lower():
            return part.strip()
    
    # Return full text if aspect not in isolated clause
    return text

st.title("Aspect-Based Twitter Sentiment Analysis")

tweet = st.text_area("Enter Tweet")

if st.button("Analyze"):
    if not tweet.strip():
        st.warning("Please enter a tweet!")
    else:
        clean = clean_text(tweet)
        st.write("**Cleaned Text:**", clean)
        
        # Overall sentiment
        overall_sentiment = get_sentiment(clean)
        st.write(f"**Overall Sentiment:** {overall_sentiment}")
        
        # Aspect-level sentiment with local context
        aspects = extract_aspects(clean)
        if aspects:
            st.write("**Aspect-Level Sentiment (Clause-Based Context):**")
            for aspect in aspects[:10]:  # Show top 10 aspects
                # Extract local context (clause containing the aspect)
                local_context = get_local_context(aspect, clean)
                sentiment = get_sentiment(local_context)
                st.write(f"  • {aspect} → {sentiment} _{local_context}_")
        else:
            st.info("No aspects found in the text.")

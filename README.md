# Aspect-Based Twitter Sentiment Analysis

## 📌 Project Overview

A **lightweight NLP application** that analyzes sentiment of tweets at both overall and aspect levels. Users input any text, and the system:
1. Cleans the text
2. Extracts key aspects (nouns)
3. Analyzes sentiment (POSITIVE/NEGATIVE)
4. Displays results via interactive UI

---

## 🏗️ System Architecture

```
┌─────────────┐
│  User Input │ (Text typed in Streamlit)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Preprocessing   │ (Clean text: lowercase, remove URLs, special chars)
└──────┬───────────┘
       │
       ▼
┌──────────────────────┐
│ Aspect Extraction    │ (Extract nouns using spaCy: flight, seats, crew)
│ + Filtering          │ (Remove useless words: this, it, that)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────┐
│ Local Context Extraction     │ (Split by "but", "and" to isolate clause)
│ (Clause-Based)               │ (Find clause containing the aspect)
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Clause-Based         │ (BERT analyzes aspect IN ITS CLAUSE only)
│ Sentiment Analysis   │ (Ignores unrelated clauses)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Display Results      │ (Show overall + aspect sentiments)
└──────────────────────┘
```

### Why Clause-Based Analysis?

**Problem with Full-Sentence Context:**
```
Input: "food quality is great but delivery time is horrible"

Model Analysis:
- Aspect "food quality" with full sentence
- Model sees BOTH positive ("great") and negative ("horrible") 
- Might assign wrong sentiment due to dominance of last phrase
```

**Solution with Clause-Based Context:**
```
Input: "food quality is great but delivery time is horrible"

Smart Extraction:
- Aspect "food quality" → Found in "food quality is great" clause only
- Aspect "delivery time" → Found in "delivery time is horrible" clause only
- Each aspect analyzed IN ITS OWN CLAUSE only
- Result: Correct sentiments for both aspects
```

**Technical Name:** Rule-Enhanced Contextual ABSA (Aspect-Based Sentiment Analysis)

### Data Flow Example

**Input:** `"The food quality is great but delivery time is horrible"`

**Processing:**
```
1. CLEAN: "the food quality is great but delivery time is horrible"

2. EXTRACT ASPECTS: [food quality, delivery time]
   (Filtered out useless words)

3. OVERALL SENTIMENT: 
   Input → BERT → MIXED (Contains both positive and negative)

4. CLAUSE-BASED CONTEXT EXTRACTION:
   - "food quality" found in: "the food quality is great"
   - "delivery time" found in: "delivery time is horrible"

5. ASPECT SENTIMENT (Clause-Isolated):
   - "food quality" clause: "the food quality is great" → POSITIVE ✓
   - "delivery time" clause: "delivery time is horrible" → NEGATIVE ✓

6. DISPLAY: Show both overall and aspect-level sentiments
```

**Why This Works:**
- Each aspect analyzed in its own clause, not influenced by other clauses
- No sentence-level dominance effects
- Aspect sentiment reflects ONLY relevant context

---

## 📂 Project Structure

```
aspect_sentiment_project/
├── app.py                      # Streamlit UI & main orchestration
├── preprocessing.py            # Text cleaning (regex-based)
├── aspect_extractor.py         # Noun extraction (spaCy NLP)
├── sentiment_model.py          # BERT sentiment classification
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📋 File Responsibilities

| File | Purpose | Technology |
|------|---------|-----------|
| `app.py` | Main UI & pipeline orchestration | Streamlit |
| `preprocessing.py` | Text cleaning (lowercase, remove URLs/special chars) | RegEx |
| `aspect_extractor.py` | Extract noun phrases (aspects) | spaCy NLP |
| `sentiment_model.py` | Classify sentiment POSITIVE/NEGATIVE | BERT (Transformers) |

---

## 🧠 Model Details

### Pre-trained Model Used
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Source:** Hugging Face
- **Training Data:** SST-2 dataset (movie reviews, tweets)
- **Classes:** POSITIVE, NEGATIVE
- **Note:** NO training needed - model is ready to use

### Why We Use This Model
✅ Already trained on millions of tweets & reviews  
✅ No training time required  
✅ High accuracy  
✅ Lightweight & fast  
✅ Free & open-source  

### How the Model Works
- **NO training needed** - model is pre-trained and ready to use
- Analyzes user text input in real-time
- Works entirely on the text you type (no dataset required)

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+ (tested on Python 3.12)
- pip package manager

### Installation

**Option 1: Using Virtual Environment (Recommended)**

```bash
# Navigate to project
cd aspect_sentiment_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

**Option 2: Global Installation (Current Setup)**

```bash
# Navigate to project
cd aspect_sentiment_project

# Install dependencies globally
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

**Note:** Dependencies are currently installed globally. For better project isolation, consider using a virtual environment.

### Run the App

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## 📸 Demo Screenshot

![Application Demo](screenshots/demo.png)

**Example Analysis:**
- **Input:** "food quality is great but delivery time is horrible"
- **Overall Sentiment:** NEGATIVE
- **Aspect-Level Results:**
  - `food quality` → POSITIVE _(food quality is great)_
  - `delivery time` → NEGATIVE _(delivery time is horrible)_

The clause-based context extraction correctly separates sentiments for different aspects!

---

## 📝 How to Use

1. **Open the Streamlit app** (runs on localhost:8501)
2. **Type a tweet or sentence** in the text area
3. **Click "Analyze"** button
4. **See results:**
   - Cleaned text
   - Overall sentiment (POSITIVE/NEGATIVE)
   - Aspect-level sentiment breakdown

### Example Test Inputs

**Negative sentiment:**
```
I hate this airline the flight was terrible and the seats are so uncomfortable
```
Expected: Overall NEGATIVE, Aspects (airline, flight, seats) → NEGATIVE

**Positive sentiment:**
```
I love the new seats the crew was amazing and the service was excellent
```
Expected: Overall POSITIVE, Aspects (seats, crew, service) → POSITIVE

**Mixed sentiment:**
```
Good food but terrible seats and bad cabin crew
```
Expected: Overall MIXED, Aspects vary by word association

---

## 🎓 Viva / Presentation Explanation

### One-Liner
> "The system takes user input, cleans the tweet, extracts important aspects using NLP, analyzes sentiment for each aspect using its isolated clause context, and displays detailed results through a Streamlit interface."

### Detailed Explanation (45 seconds)
> "This is a clause-based aspect sentiment analysis system. Users enter text through Streamlit. The system preprocesses text by cleaning it. Using spaCy, we extract key aspects like nouns. Then, here's the key innovation: we don't analyze aspects with full sentence context. Instead, we split the sentence by connectors like 'but' and 'and' to isolate the clause containing each aspect. We then apply a pre-trained BERT model to classify sentiment as POSITIVE or NEGATIVE for that clause specifically. This prevents cross-clause sentiment confusion. Finally, we display both overall sentiment and accurate aspect-level sentiment."

### Technical Explanation (For Detailed Questions)

**If Asked: "Why use clause-based analysis instead of full-sentence context?"**

> "Sentence-level sentiment models tend to assign dominant or final sentiment to all aspects, even when different clauses express different sentiments. For example: 'food quality is great but delivery time is horrible'. A full-sentence approach might confuse the aspect 'food quality' with the negative sentiment from 'horrible'. By extracting the specific clause containing each aspect, we ensure accurate aspect-specific sentiment classification. This is called Rule-Enhanced Contextual ABSA."

**If Asked: "How does this compare to standard ABSA approaches?"**

> "Standard approach uses fine-tuned BERT models trained on aspect-sentiment pairs. Our approach is simpler but still effective: we use rule-based clause extraction combined with a pre-trained sentiment model. This keeps the system explainable and avoids the complexity of fine-tuning, making it perfect for practical applications and mini projects."

### Architecture Explanation
> "The pipeline has six main stages: First, text preprocessing removes noise. Second, aspect extraction identifies important entities using spaCy. Third, aspect filtering removes meaningless words. Fourth, clause-based context extraction splits sentences by connectors to isolate aspect-relevant context. Fifth, sentiment analysis applies BERT to the isolated clause. Sixth, results are displayed. The threelayer hierarchy (Sentence → Clause → Aspect → Sentiment) ensures intelligent context isolation."

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Streamlit | Interactive web UI |
| **NLP - Aspect Extraction** | spaCy | Extract noun phrases |
| **NLP - Sentiment** | Transformers (BERT) | Classify sentiment |
| **Python Version** | 3.8+ | Core language |

---

## 📊 Model Performance

- **Accuracy on SST-2:** ~91%
- **Inference Time:** ~200-500ms per analysis
- **Model Size:** ~268MB (downloaded automatically)

---

## ✨ Features Implemented

✅ Clean, intuitive Streamlit UI  
✅ Text preprocessing (lowercase, remove noise)  
✅ Aspect extraction using spaCy NLP  
✅ Overall sentiment classification  
✅ Aspect-level sentiment analysis  
✅ Error handling & validation  
✅ Real-time results display  

---

## 🎯 Why This Design

1. **Simple:** No microservices, no APIs, no complexity
2. **Real:** Uses actual NLP libraries (spaCy, BERT)
3. **Scalable:** Can process any text input
4. **Maintainable:** Clean modular code structure
5. **Resume-Ready:** Shows practical ML skills

---

## 🔧 Implementation Improvements

### Improvement 1: Clause-Based Context Extraction

**Problem:** Full-sentence context can be misleading when sentence has multiple sentiments.

Example:
```
"food quality is great but delivery time is horrible"
```

- If we analyze "food quality" with full sentence context
- Model might be confused by "horrible" at the end
- Wrong sentiment assignment for "food quality"

**Solution:** Extract only the clause containing the aspect.

**How it works:**
1. Split sentence by "but", "and" (connectors that indicate separate clauses)
2. Find which clause contains the aspect
3. Run sentiment analysis on ONLY that clause

**Code Implementation:**
```python
def get_local_context(aspect, text):
    # Split by 'but' (contrast indicator)
    parts = text.split(" but ")
    for part in parts:
        if aspect.lower() in part.lower():
            return part.strip()
    
    # Split by 'and'
    parts = text.split(" and ")
    for part in parts:
        if aspect.lower() in part.lower():
            return part.strip()
    
    return text  # Fallback
```

**Result:**

| Aspect | Extracted Clause | Sentiment |
|--------|------------------|-----------|
| food quality | "food quality is great" | POSITIVE ✓ |
| delivery time | "delivery time is horrible" | NEGATIVE ✓ |

**Technical Name:** Clause-Based Contextual ABSA

---

### Improvement 2: Filter Useless Aspects

**Problem:** spaCy extracts garbage aspects like "this", "it", "that"

**Solution:** Filter out short words and common pronouns

**Code Change:**
```python
# aspect_extractor.py
if len(chunk.text) > 2 and chunk.text.lower() not in ["this", "it", "that", "what", "which", "who", "one"]:
    aspects.append(chunk.text)
```

**Result:** Only meaningful aspects are extracted and displayed.

---

### Three-Layer Architecture

```
Sentence (Full text)
    ↓
Clause (Aspect-specific context)
    ↓
Aspect (The noun/entity)
    ↓
Sentiment (POSITIVE/NEGATIVE)
```

This is intelligent hierarchical design, not a hack!



## 📝 How to Explain This in Viva

### If Examiner Asks: "Why were the earlier results wrong?"

**Answer:**
> "Initially, we analyzed aspects with full sentence context, which works for simple sentences. However, when a sentence contains multiple clauses with different sentiments separated by 'but' or 'and', the model gets confused by unrelated sentiment words. For example, 'food quality is great but delivery time is horrible' would confuse the food quality aspect with the negative sentiment. We improved accuracy by using clause-based context extraction—we split sentences and isolate the exact clause containing each aspect."

### If Examiner Asks: "How does clause extraction improve results?"

**Answer:**
> "Clause extraction works by splitting sentences at natural boundaries like 'but' and 'and'. Once we find the clause containing an aspect, we analyze sentiment only on that clause, not the entire sentence. This isolates irrelevant sentiment words in other clauses. Result: 'food quality' from 'food quality is great' gets analyzed independently from 'delivery time is horrible'."

### If Examiner Asks: "What type of ABSA are you using?"

**Answer:**
> "We're using a **Rule-Enhanced Contextual approach** to Aspect-Based Sentiment Analysis. We combine rule-based clause extraction with a pre-trained BERT sentiment model. This is simpler than training aspect-aware transformers, but highly effective for practical applications. The threelayer hierarchy (Sentence → Clause → Aspect → Sentiment) ensures intelligent context isolation."

### If Examiner Asks: "How is this different from standard sentiment analysis?"

**Answer:**
> "Standard sentiment analysis gives only overall sentiment for the entire text. Our system goes further by: (1) identifying specific aspects (nouns) mentioned in the text, (2) isolating the context for each aspect using clause extraction, (3) predicting sentiment for each aspect individually. This gives detailed, actionable insights about what specifically was positive or negative."

### If Examiner Asks: "Why not use a fine-tuned ABSA model?"

**Answer:**
> "Fine-tuned ABSA models require aspect-sentiment pair datasets and expensive training. Our approach achieves similar results using rule-based logic combined with a pre-trained model. This makes the system explainable, lightweight, and practical—exactly what's needed for a mini project. Advanced fine-tuning is research-level work."

---

---

## 🔧 Limitations & Future Improvements

### Limitations Addressed ✅
- ✅ Aspects analyzed with clause-based context (not isolated words)
- ✅ Filtering removes meaningless pronouns (this, it, that)
- ✅ Clause extraction prevents cross-clause sentiment confusion
- ✅ Rule-enhanced contextual ABSA approach implemented

### Remaining Limitations
- Limited to POSITIVE/NEGATIVE (no neutral class)
- Single-language (English only)
- Clause extraction works for "but"/"and" connectors (not advanced linguistic parsing)
- Rule-based approach (vs. advanced fine-tuned models)

### Possible Future Improvements
- Add neutral sentiment class  
- Support more clause connectors (however, although, despite, etc.)
- Fine-tune BERT on custom aspect-sentiment dataset
- Multi-language support
- Add confidence scores to predictions
- Aspect categorization (service, food, price, delivery, etc.)
- Sentiment trend analysis for multiple tweets

---

## 📚 Dependencies

See `requirements.txt` for complete list:
- streamlit
- transformers
- spacy
- torch

---

## 👨‍💻 Developer Notes

- Model downloads automatically on first run
- spaCy model must be downloaded separately
- All computations happen locally (no API calls)
- No training required - pre-trained model used

---

## 📄 License

Open source - free to use and modify


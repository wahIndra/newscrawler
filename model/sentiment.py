from transformers import pipeline
import numpy as np

class SentimentAnalyzer:
    MODELS = {
        "IndoBERT (ID)": "w11wo/indonesian-roberta-base-sentiment-classifier",
        "XLM-RoBERTa (Multilingual)": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "DistilBERT (EN)": "distilbert-base-uncased-finetuned-sst-2-english"
    }
    
    # Import keywords from external file to keep code clean
    from model.keywords import POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS

    def __init__(self):
        self.pipelines = {}

    def load_model(self, model_name):
        if model_name not in self.pipelines:
            print(f"Loading model: {model_name}...")
            hf_path = self.MODELS[model_name]
            try:
                self.pipelines[model_name] = pipeline("sentiment-analysis", model=hf_path, tokenizer=hf_path, top_k=None)
                print(f"Model {model_name} loaded.")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                return None
        return self.pipelines[model_name]

    def predict(self, text, model_names=["IndoBERT (ID)"]):
        """
        Predict sentiment using one or more models (Ensemble).
        """
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        valid_models = 0
        
        # Truncate text
        max_len = 512 * 4
        truncated_text = text[:max_len]

        for name in model_names:
            pipe = self.load_model(name)
            if not pipe:
                continue
                
            try:
                results = pipe(truncated_text)[0] # List of dicts [{'label': '...', 'score': ...}, ...]
                
                # Normalize labels to standard positive/negative/neutral
                model_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
                
                for res in results:
                    label = res['label'].lower()
                    score = res['score']
                    
                    # Map labels
                    if "pos" in label or "LABEL_2" == label: # DistilBERT uses LABEL_1/0 sometimes or POSITIVE
                        model_scores["positive"] = score
                    elif "neg" in label or "LABEL_0" == label:
                        model_scores["negative"] = score
                    elif "neu" in label or "LABEL_1" == label:
                        model_scores["neutral"] = score
                    
                    # DistilBERT specific (POSITIVE/NEGATIVE only)
                    if label == "positive": model_scores["positive"] = score
                    if label == "negative": model_scores["negative"] = score
                
                # Add to total
                for k in scores:
                    scores[k] += model_scores[k]
                valid_models += 1
                
            except Exception as e:
                print(f"Error predicting with {name}: {e}")

        if valid_models == 0:
            return {"label": "neutral", "score": 0.0}

        # Average
        for k in scores:
            scores[k] /= valid_models
            
        # Find max
        final_label = max(scores, key=scores.get)
        final_score = scores[final_label]
        
        return {"label": final_label, "score": final_score, "details": scores}

    def explain_sentiment(self, label, score, text):
        """
        Generate an analyst-style explanation.
        """
        confidence = "high" if score > 0.8 else "moderate" if score > 0.5 else "low"
        
        markers = self.get_sentiment_markers(text)
        pos_markers = markers['positive']
        neg_markers = markers['negative']
        
        explanation = f"The model indicates a **{label}** sentiment with **{confidence} confidence** ({score:.2f}). "
        
        if label == "positive":
            if pos_markers:
                explanation += f"This is driven by positive indicators such as: *{', '.join(pos_markers[:3])}*. "
            else:
                explanation += "The overall tone suggests optimism or favorable outcomes. "
        elif label == "negative":
            if neg_markers:
                explanation += f"This is influenced by negative terms like: *{', '.join(neg_markers[:3])}*. "
            else:
                explanation += "The content reflects concern, criticism, or unfavorable conditions. "
        else:
            explanation += "The text appears balanced or factual without strong emotional bias. "
            
        return explanation

    def get_sentiment_markers(self, text):
        """
        Extract positive and negative keywords.
        """
        text_lower = text.lower()
        # Use simple tokenization or string matching
        # For better accuracy, we could use regex boundaries \bword\b but simple 'in' is faster for now
        pos_found = [word for word in self.POSITIVE_KEYWORDS if word in text_lower]
        neg_found = [word for word in self.NEGATIVE_KEYWORDS if word in text_lower]
        return {"positive": list(set(pos_found)), "negative": list(set(neg_found))}

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    # Test Ensemble
    print(analyzer.predict("Ekonomi Indonesia sedang tumbuh pesat.", ["IndoBERT (ID)", "XLM-RoBERTa (Multilingual)"]))


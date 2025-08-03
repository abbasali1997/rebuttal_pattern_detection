import json
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.sparse import hstack, csr_matrix
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
import argparse
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Enhanced feature extraction
HEDGING_TERMS = ['we believe', 'we think', 'may', 'might', 'possibly', 'could', 'suggest', 'appears', 'seems', 'likely', 'probably']
CONFIDENCE_TERMS = ['clearly', 'obviously', 'undoubtedly', 'evident', 'certainly', 'definitely', 'demonstrate', 'prove', 'show']
PRONOUNS = ['we', 'our', 'ourselves', 'us']
TECHNICAL_TERMS = ['algorithm', 'method', 'approach', 'technique', 'framework', 'model', 'system', 'evaluation', 'experiment']

def extract_rating_score(rating_str):
    """Extract numeric score from rating string"""
    if not rating_str or rating_str == "null":
        return 0
    # Extract first number from strings like "8: Top 50% of accepted papers, clear accept"
    match = re.search(r'(\d+)', str(rating_str))
    return int(match.group(1)) if match else 0

def extract_confidence_score(confidence_str):
    """Extract numeric confidence from confidence string"""
    if not confidence_str or confidence_str == "null":
        return 0
    match = re.search(r'(\d+)', str(confidence_str))
    return int(match.group(1)) if match else 0

def extract_handcrafted_features(df):
    sid = SentimentIntensityAnalyzer()
    features = []

    for _, row in df.iterrows():
        rebuttal_text = row['rebuttal']
        review_content = row.get('review_content', '')
        review_title = row.get('review_title', '')
        
        # Rebuttal features
        tone = sid.polarity_scores(rebuttal_text)
        sentences = sent_tokenize(rebuttal_text.lower())
        word_list = re.findall(r'\b\w+\b', rebuttal_text.lower())

        # List structure detection
        list_patterns = [
            r'^\s*[\-\*\•]\s+',
            r'^\s*\d+[\.\)]\s+',
            r'^\s*[a-zA-Z][\.\)]\s+',
        ]
        lines = rebuttal_text.split('\n')
        list_count = sum(any(re.match(pattern, line) for pattern in list_patterns) for line in lines)
        has_list_structure = int(list_count > 1)

        # Gratitude and politeness
        starts_with_gratitude = int(bool(re.match(r'^\s*(thanks|thank you|we thank)', rebuttal_text.lower())))
        starts_with_we_thank_reviewers = int(rebuttal_text.lower().strip().startswith("we thank the reviewers"))
        
        # Length features
        is_short = int(len(rebuttal_text) < 1000)
        is_long = int(len(rebuttal_text) > 4000)
        is_medium = int(1000 <= len(rebuttal_text) <= 4000)

        # Language style features
        hedging_sentences = sum(any(term in s for term in HEDGING_TERMS) for s in sentences)
        confidence_sentences = sum(any(term in s for term in CONFIDENCE_TERMS) for s in sentences)
        technical_sentences = sum(any(term in s for term in TECHNICAL_TERMS) for s in sentences)

        total_sentences = len(sentences) if sentences else 1
        hedging_ratio = hedging_sentences / total_sentences
        confidence_ratio = confidence_sentences / total_sentences
        technical_ratio = technical_sentences / total_sentences

        # Pronouns and personal touch
        pronoun_count = sum(word in PRONOUNS for word in word_list)
        personal_pronoun_ratio = pronoun_count / len(word_list) if word_list else 0

        # Review content features
        review_tone = sid.polarity_scores(review_content)
        review_length = len(review_content)
        
        # Score features
        initial_rating = extract_rating_score(row.get('initial_score_rating', 0))
        initial_confidence = extract_confidence_score(row.get('initial_score_confidence', 0))
        final_rating = extract_rating_score(row.get('final_score_rating', 0))
        final_confidence = extract_confidence_score(row.get('final_score_confidence', 0))
        
        # Score changes
        rating_change = final_rating - initial_rating
        confidence_change = final_confidence - initial_confidence
        
        # Conference features
        conference_year = row.get('conference_year_track', '')
        is_acl = int('ACL' in conference_year)
        is_emnlp = int('EMNLP' in conference_year)
        is_naacl = int('NAACL' in conference_year)
        is_workshop = int('Workshop' in conference_year)
        
        # Reviewer features
        reviewer_id = row.get('reviewer_id', '')
        is_anonymous = int('Anon' in reviewer_id)

        features.append({
            # Rebuttal text features
            "char_length": len(rebuttal_text),
            "word_count": len(word_list),
            "sentence_count": len(sentences),
            "paragraphs": rebuttal_text.count('\n\n') + 1,
            "flesch_readability": textstat.flesch_reading_ease(rebuttal_text),
            "tone_pos": tone['pos'],
            "tone_neg": tone['neg'],
            "tone_neu": tone['neu'],
            "tone_compound": tone['compound'],
            
            # Structure features
            "has_list_structure": has_list_structure,
            "starts_with_gratitude": starts_with_gratitude,
            "starts_with_we_thank_reviewers": starts_with_we_thank_reviewers,
            "is_short": is_short,
            "is_long": is_long,
            "is_medium": is_medium,
            
            # Language style
            "hedging_terms_ratio": hedging_ratio,
            "confidence_terms_ratio": confidence_ratio,
            "technical_terms_ratio": technical_ratio,
            "personal_pronoun_ratio": personal_pronoun_ratio,
            
            # Review features
            "review_length": review_length,
            "review_tone_pos": review_tone['pos'],
            "review_tone_neg": review_tone['neg'],
            "review_tone_compound": review_tone['compound'],
            
            # Conference features
            "is_acl": is_acl,
            "is_emnlp": is_emnlp,
            "is_naacl": is_naacl,
            "is_workshop": is_workshop,
            "is_anonymous": is_anonymous,
        })

    return pd.DataFrame(features)

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for d in data:
        # Combine all rebuttals into one text
        rebuttal_text = " ".join(d.get("rebuttals", []))
        
        # Extract scores
        initial_score = d.get("initial_score", {})
        final_score = d.get("final_score", {})
        
        processed_data.append({
            "rebuttal": rebuttal_text,
            "label": 1 if d.get("decision", "").lower().strip() == "accept" else 0,
            "review_content": d.get("review_content", ""),
            "review_title": d.get("review_title", ""),
            "conference_year_track": d.get("conference_year_track", ""),
            "reviewer_id": d.get("reviewer_id", ""),
            "initial_score_rating": initial_score.get("rating", ""),
            "initial_score_confidence": initial_score.get("confidence", ""),
            "final_score_rating": final_score.get("rating", ""),
            "final_score_confidence": final_score.get("confidence", ""),
        })
    
    return pd.DataFrame(processed_data)

# Enhanced Neural Network
class ImprovedNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.network(x))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        eps = 1e-6
        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        bce = - (targets*torch.log(inputs) + (1-targets)*torch.log(1-inputs))
        pt = targets*inputs + (1-targets)*(1-inputs)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce
        return focal_loss.mean()

class RebuttalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="final_data/REVIEWS_train_with_rebuttals.json")
    parser.add_argument("--test_path", type=str, default="final_data/REVIEWS_test_with_rebuttals.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    train_df = load_data(args.train_path)
    test_df = load_data(args.test_path)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train accept rate: {train_df['label'].mean():.3f}")
    print(f"Test accept rate: {test_df['label'].mean():.3f}")

    y_train = train_df['label']
    y_test = test_df['label']

    # Enhanced TF-IDF with better parameters
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=10000, 
        ngram_range=(1, 3), 
        stop_words='english',
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(train_df['rebuttal'])
    X_test_tfidf = tfidf.transform(test_df['rebuttal'])
    
    print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

    # Enhanced handcrafted features
    print("Extracting handcrafted features...")
    X_train_hand = extract_handcrafted_features(train_df)
    X_test_hand = extract_handcrafted_features(test_df)
    
    print(f"Handcrafted features: {X_train_hand.shape[1]}")

    # Scale handcrafted features
    scaler = StandardScaler()
    X_train_hand_scaled = scaler.fit_transform(X_train_hand)
    X_test_hand_scaled = scaler.transform(X_test_hand)

    # Combine features
    X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_hand_scaled])
    X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_hand_scaled])
    
    print(f"Combined features: {X_train_combined.shape[1]}")

    # Split train into train/validation
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_combined, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # PyTorch datasets
    train_dataset = RebuttalDataset(X_train, y_train_split)
    val_dataset = RebuttalDataset(X_val, y_val)
    test_dataset = RebuttalDataset(X_test_combined, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model
    model = ImprovedNN(input_dim=X_train_combined.shape[1])
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop with early stopping
    print("Training model...")
    best_val_f1 = 0
    patience_counter = 0
    patience = 10
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch).squeeze()
                val_preds.extend((preds > 0.5).int().tolist())
                val_true.extend(y_batch.int().tolist())
        
        val_f1 = f1_score(val_true, val_preds)
        scheduler.step(total_loss / len(train_dataset))
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_dataset):.4f}, Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Final evaluation
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds = model(X_batch).squeeze()
            all_probs.extend(preds.tolist())
            all_preds.extend((preds > 0.5).int().tolist())

    accuracy = accuracy_score(y_test, all_preds)
    f1 = f1_score(y_test, all_preds)
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, all_preds, target_names=["Reject", "Accept"]))

if __name__ == "__main__":
    main() 

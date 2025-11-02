import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle as pickle
import os


def create_model(data): 
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Test model
    y_pred = model.predict(X_test)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n🧠 Model Performance Metrics:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nDetailed Classification Report:\n", classification_report(y_test, y_pred))
    
    # Return everything
    return model, scaler, acc, prec, rec, f1


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    
    # Drop unnecessary columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    # Convert categorical target to numerical
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    return data


def main():
    data = get_clean_data()
    model, scaler, acc, prec, rec, f1 = create_model(data)
    
    # Ensure 'model' directory exists
    os.makedirs('model', exist_ok=True)
    
    # Save model and scaler
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metrics too
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
    
    with open('model/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    print("\n✅ Model, Scaler, and Metrics saved successfully in 'model/' folder.")


if __name__ == '__main__':
    main()

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class PromptGuardService:
    def __init__(self):
        #Initialize the prompt guard model
        self.model_id = "meta-llama/Llama-Prompt-Guard-2-86M"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        
        #Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def validate_input(self, text: str) -> dict:
        #Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        #Get predictions
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predicted_class_id = logits.argmax().item()
        
        #Get the label and confidence
        label = self.model.config.id2label[predicted_class_id]
        
        confidence = torch.softmax(logits, dim=1)[0][predicted_class_id].item()
        
        print(f"Text: {text}, Label: {label}, Confidence: {confidence}")
        
        #Determine if the input is malicious
        is_malicious = label == 'LABEL_1'
        
        return {
            'is_malicious': is_malicious,
            'label': label,
            'confidence': confidence,
            'input_text': text
        } 
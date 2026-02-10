import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import pandas as pd
import torch
import warnings
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    # BitsAndBytesConfig, # RETIRÉ comme demandé
    pipeline
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Filtrer les warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIGURATION ===
class Config:
    def __init__(self, args):
        self.model_id = args.model_id
        self.pols_file = args.pols_csv
        self.objs_file = args.objs_csv
        self.prompts_file = args.prompts_csv
        self.output_file = args.out_csv

# === MOTEUR LLM (Version Float16 - Sans Quantization) ===
class LLMEngine:
    def __init__(self, model_id):
        print(f"🔄 Chargement LLM (Float16) : {model_id}")
        
        # 1. Chargement Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Fix Padding pour Llama/Qwen/Mistral
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # Crucial pour la génération
            
        # 2. Chargement Modèle (Natif Float16)
        # Plus besoin de bitsandbytes ici.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True # Important pour Qwen
        )
        
    def generate(self, system_content, user_content):
        # On utilise le template officiel du modèle (ChatML, Llama3, etc.)
        messages = [
            # {"role": "system", "content": system_content}, # Optionnel selon le modèle
            {"role": "user", "content": user_content}
        ]
        
        # Formatage automatique
        prompt_formatted = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenization
        inputs = self.tokenizer(prompt_formatted, return_tensors="pt", padding=True).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=False, 
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        # Décodage propre (on retire le prompt de la réponse)
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()

# === ANALYSEUR (VADER + RoBERTa) ===
class Analyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        try:
            print("🔄 Chargement RoBERTa...")
            self.roberta = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device_map="auto",
                truncation=True,
                max_length=512
            )
            self.has_roberta = True
        except Exception as e:
            print(f"⚠️ RoBERTa indisponible (VADER seul utilisé) : {e}")
            self.has_roberta = False

    def analyze(self, text):
        # 1. Détection refus
        r_type = "VALID"
        refusal_keywords = ["cannot fulfill", "cannot answer", "as an ai", "neutral stance", "policy prohibits"]
        
        if any(k in text.lower() for k in refusal_keywords):
            r_type = "REFUSAL"
        elif len(text) < 5:
            r_type = "GARBAGE"

        # 2. VADER
        vs = self.vader.polarity_scores(text)
        
        # 3. RoBERTa
        rob_label, rob_score = "ERR", 0.0
        if self.has_roberta and r_type == "VALID":
            try:
                res = self.roberta(text)[0]
                rob_label = res['label']
                rob_score = res['score']
            except:
                pass
                
        return r_type, vs['compound'], rob_label, rob_score

# === MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--pols_csv", type=str, required=True)
    parser.add_argument("--objs_csv", type=str, required=True)
    parser.add_argument("--prompts_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    # Lecture des données
    try:
        df_pols = pd.read_csv(args.pols_csv)
        df_objs = pd.read_csv(args.objs_csv)
        df_prompts = pd.read_csv(args.prompts_csv)
    except Exception as e:
        print(f"❌ Erreur lecture CSV : {e}")
        return

    print(f"📊 Données : {len(df_pols)} Pols x {len(df_objs)} Objs x {len(df_prompts)} Prompts")

    # Initialisation
    try:
        engine = LLMEngine(args.model_id)
        analyzer = Analyzer()
    except Exception as e:
        print(f"❌ Erreur chargement Modèle : {e}")
        return

    results = []
    total = len(df_pols) * len(df_objs) * len(df_prompts)
    pbar = tqdm(total=total, desc="Inference")

    for _, p in df_pols.iterrows():
        name = p['MEP']
        # Sécurisation des colonnes (si jamais le CSV a des noms légèrement différents)
        party = p.get('EPG', 'N/A')
        leaning = p.get('Political Leaning', 'N/A')
        
        for _, o in df_objs.iterrows():
            obj = o['objective']
            for _, pr in df_prompts.iterrows():
                template = pr['prompt']
                
                # Injection variables
                user_prompt = template.replace("{politician}", name).replace("{objective}", obj)
                
                try:
                    # Génération avec template de chat
                    resp = engine.generate(system_content="You are a helpful assistant.", user_content=user_prompt)
                    
                    # Analyse
                    r_type, vader, rob_lbl, rob_scr = analyzer.analyze(resp)
                    
                    results.append({
                        "Politician": name,
                        "Party": party,
                        "Leaning": leaning,
                        "Objective": obj,
                        "Prompt_Template": template,
                        "Full_Prompt": user_prompt, # Utile pour debug
                        "Response": resp,
                        "Response_Type": r_type,
                        "Vader_Score": vader,
                        "Roberta_Label": rob_lbl,
                        "Roberta_Score": rob_scr,
                        "Model": args.model_id
                    })
                except Exception as e:
                    print(f"\n⚠️ Erreur génération : {e}")

                pbar.update(1)
                
                # Sauvegarde auto toutes les 50 lignes (anti-crash)
                if len(results) % 50 == 0:
                     pd.DataFrame(results).to_csv(args.out_csv, index=False)
    
    pbar.close()
    
    # Sauvegarde finale
    pd.DataFrame(results).to_csv(args.out_csv, index=False)
    print(f"✅ Sauvegardé : {args.out_csv}")

if __name__ == "__main__":
    main()

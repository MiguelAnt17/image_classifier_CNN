import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import yaml
import os

class ImageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador de Imagens")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f0f0")
        
        # Variables
        self.model = None
        self.config = None
        self.model_path = ""
        self.image_path = ""
        self.config_path = ""
        
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="#f0f0f0")
        title_frame.pack(pady=10)
        
        title_label = tk.Label(
            title_frame, 
            text="🖼️ Classificador de Imagens", 
            font=("Arial", 20, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50"
        )
        title_label.pack()
        
        # Configuration section
        config_frame = tk.LabelFrame(self.root, text="Configuração", font=("Arial", 12, "bold"), bg="#f0f0f0")
        config_frame.pack(fill="x", padx=20, pady=10)
        
        # Config file selection
        config_file_frame = tk.Frame(config_frame, bg="#f0f0f0")
        config_file_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(config_file_frame, text="Arquivo de Configuração:", bg="#f0f0f0").pack(anchor="w")
        
        config_button_frame = tk.Frame(config_file_frame, bg="#f0f0f0")
        config_button_frame.pack(fill="x", pady=2)
        
        self.config_label = tk.Label(
            config_button_frame, 
            text="Nenhum arquivo selecionado", 
            bg="white", 
            relief="sunken", 
            anchor="w"
        )
        self.config_label.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        config_btn = tk.Button(
            config_button_frame,
            text="Selecionar Config",
            command=self.select_config_file,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold")
        )
        config_btn.pack(side="right")
        
        # Model file selection
        model_file_frame = tk.Frame(config_frame, bg="#f0f0f0")
        model_file_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(model_file_frame, text="Modelo (.keras):", bg="#f0f0f0").pack(anchor="w")
        
        model_button_frame = tk.Frame(model_file_frame, bg="#f0f0f0")
        model_button_frame.pack(fill="x", pady=2)
        
        self.model_label = tk.Label(
            model_button_frame, 
            text="Nenhum modelo selecionado", 
            bg="white", 
            relief="sunken", 
            anchor="w"
        )
        self.model_label.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        model_btn = tk.Button(
            model_button_frame,
            text="Selecionar Modelo",
            command=self.select_model_file,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10, "bold")
        )
        model_btn.pack(side="right")
        
        # Image selection section
        image_frame = tk.LabelFrame(self.root, text="Seleção de Imagem", font=("Arial", 12, "bold"), bg="#f0f0f0")
        image_frame.pack(fill="x", padx=20, pady=10)
        
        image_button_frame = tk.Frame(image_frame, bg="#f0f0f0")
        image_button_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(image_button_frame, text="Imagem para Classificar:", bg="#f0f0f0").pack(anchor="w")
        
        image_select_frame = tk.Frame(image_button_frame, bg="#f0f0f0")
        image_select_frame.pack(fill="x", pady=2)
        
        self.image_label = tk.Label(
            image_select_frame, 
            text="Nenhuma imagem selecionada", 
            bg="white", 
            relief="sunken", 
            anchor="w"
        )
        self.image_label.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        image_btn = tk.Button(
            image_select_frame,
            text="Selecionar Imagem",
            command=self.select_image_file,
            bg="#27ae60",
            fg="white",
            font=("Arial", 10, "bold")
        )
        image_btn.pack(side="right")
        
        # Classify button
        classify_btn = tk.Button(
            self.root,
            text="🔍 CLASSIFICAR IMAGEM",
            command=self.classify_image,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 14, "bold"),
            pady=10
        )
        classify_btn.pack(pady=20)
        
        # Results section
        results_frame = tk.LabelFrame(self.root, text="Resultados", font=("Arial", 12, "bold"), bg="#f0f0f0")
        results_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create two columns for image and results
        content_frame = tk.Frame(results_frame, bg="#f0f0f0")
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Image display column
        image_column = tk.Frame(content_frame, bg="#f0f0f0")
        image_column.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        tk.Label(image_column, text="Imagem:", font=("Arial", 11, "bold"), bg="#f0f0f0").pack(anchor="w")
        
        self.image_display = tk.Label(
            image_column,
            text="Selecione uma imagem para visualizar",
            bg="white",
            relief="sunken",
            width=30,
            height=15
        )
        self.image_display.pack(fill="both", expand=True, pady=(5, 0))
        
        # Results column
        results_column = tk.Frame(content_frame, bg="#f0f0f0")
        results_column.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        tk.Label(results_column, text="Classificação:", font=("Arial", 11, "bold"), bg="#f0f0f0").pack(anchor="w")
        
        self.results_text = tk.Text(
            results_column,
            height=15,
            width=40,
            bg="white",
            relief="sunken",
            font=("Arial", 10)
        )
        self.results_text.pack(fill="both", expand=True, pady=(5, 0))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Pronto - Selecione os arquivos necessários")
        
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            bg="#ecf0f1",
            font=("Arial", 9)
        )
        status_bar.pack(side="bottom", fill="x")
    
    def select_config_file(self):
        file_path = filedialog.askopenfilename(
            title="Selecionar arquivo de configuração",
            filetypes=[("YAML files", "*.yaml"), ("YAML files", "*.yml"), ("All files", "*.*")]
        )
        
        if file_path:
            self.config_path = file_path
            self.config_label.config(text=os.path.basename(file_path))
            self.load_config()
    
    def select_model_file(self):
        file_path = filedialog.askopenfilename(
            title="Selecionar modelo",
            filetypes=[("Keras files", "*.keras"), ("H5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if file_path:
            self.model_path = file_path
            self.model_label.config(text=os.path.basename(file_path))
            self.load_model()
    
    def select_image_file(self):
        file_path = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.image_label.config(text=os.path.basename(file_path))
            self.display_image()
    
    def load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self.status_var.set("Configuração carregada com sucesso")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar configuração: {str(e)}")
            self.status_var.set("Erro ao carregar configuração")
    
    def load_model(self):
        try:
            self.status_var.set("Carregando modelo...")
            self.root.update()
            self.model = tf.keras.models.load_model(self.model_path)
            self.status_var.set("Modelo carregado com sucesso")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar modelo: {str(e)}")
            self.status_var.set("Erro ao carregar modelo")
    
    def display_image(self):
        try:
            # Open and resize image for display
            img = Image.open(self.image_path)
            img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.image_display.config(image=photo, text="")
            self.image_display.image = photo  # Keep a reference
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exibir imagem: {str(e)}")
    
    def classify_image(self):
        # Check if all required files are selected
        if not self.config:
            messagebox.showerror("Erro", "Por favor, selecione um arquivo de configuração")
            return
        
        if not self.model:
            messagebox.showerror("Erro", "Por favor, selecione um modelo")
            return
        
        if not self.image_path:
            messagebox.showerror("Erro", "Por favor, selecione uma imagem")
            return
        
        try:
            self.status_var.set("Classificando imagem...")
            self.root.update()
            
            # Get parameters from config
            img_size = tuple(self.config['image_size'])
            class_names = self.config['class_names']
            
            # Process image
            img = Image.open(self.image_path).convert('RGB').resize(img_size)
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])
            
            # Get results
            predicted_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)
            
            # Display results
            self.results_text.delete(1.0, tk.END)
            result_text = f"""
╔══════════════════════════════════╗
║         RESULTADO DA             ║
║        CLASSIFICAÇÃO             ║
╚══════════════════════════════════╝

🏷️  Classe Predita: {predicted_class}

📊 Confiança: {confidence:.2f}%

📈 Pontuações por Classe:
{'-' * 40}
"""
            
            # Add scores for all classes
            for i, (class_name, prob) in enumerate(zip(class_names, score)):
                percentage = 100 * prob
                bar = "█" * int(percentage / 5)  # Simple progress bar
                result_text += f"{class_name:15} | {percentage:6.2f}% {bar}\n"
            
            result_text += f"\n{'-' * 40}\n"
            result_text += f"📁 Arquivo: {os.path.basename(self.image_path)}\n"
            result_text += f"🤖 Modelo: {os.path.basename(self.model_path)}\n"
            
            self.results_text.insert(1.0, result_text)
            self.status_var.set(f"Classificação concluída: {predicted_class} ({confidence:.1f}%)")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro durante a classificação: {str(e)}")
            self.status_var.set("Erro durante a classificação")

def main():
    root = tk.Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
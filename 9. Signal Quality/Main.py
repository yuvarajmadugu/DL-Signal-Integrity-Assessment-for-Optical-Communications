# app_updated.py
import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageTk

# -------------------- Constants --------------------
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

APP_BG = "#f4f8ff"
BTN_BG = "#e9eefc"

TITLE_FONT = ("Times New Roman", 16, "bold")
BTN_FONT = ("Times New Roman", 14, "bold")
OUT_FONT = ("Times New Roman", 12, "bold")

# -------------------- Global state --------------------
df = None
X = None
y = None
label_encoders = None
X_train = None
X_test = None
y_train = None
y_test = None
class_labels = None
metrics_df = None  # Placeholder if needed

# -------------------- Tkinter Main Window --------------------
main = tk.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")
main.title("Deep Learning Signal Integrity Assessment")

# -------------------- Background Image --------------------
try:
    bg_image = Image.open("Bg_img.jpg")  # Change to your image
    bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(main, image=bg_photo)
    bg_label.place(relwidth=1, relheight=1)
except Exception as e:
    print("Background image load failed:", e)

# -------------------- Output Box --------------------
output_box = tk.Text(main, height=22, width=130, bg="black", fg="pink", insertbackground="white")
scroll = tk.Scrollbar(output_box)
output_box.configure(yscrollcommand=scroll.set)
output_box.place(x=100, y=70)
#scroll.place(text.place(x=100,y=90))  # adjust as needed

# -------------------- Functions --------------------
def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()


# -------------------- GUI Actions --------------------
def action_upload_dataset():
    global df
    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not path:
        return
    df = pd.read_csv(path)
    output_box.delete(1.0, tk.END)
    output_box.insert(tk.END, f"Loaded dataset: {os.path.basename(path)}\n")
    output_box.insert(tk.END, f"Head:\n{df.head(10)}\nShape: {df.shape}\nColumns: {list(df.columns)}\n")
    output_box.see(tk.END)


# -------------------- Core Functionalities --------------------
def preprocess_data(df_in, is_train=True, label_encoders_in=None):
    df_work = df_in.copy()
    df_work = df_work.loc[:, ~df_work.columns.str.contains('^Unnamed')]
    target_col = 'Signal Quality'
    if is_train and target_col in df_work.columns:
        df_work = df_work.dropna(subset=[target_col])
    if is_train:
        encoders = {}
        for col in df_work.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_work[col] = le.fit_transform(df_work[col].astype(str))
            encoders[col] = le
    else:
        if label_encoders_in is None:
            raise ValueError("label_encoders must be provided for test/inference.")
        encoders = label_encoders_in
        for col in df_work.select_dtypes(include='object').columns:
            if col in encoders:
                df_work[col] = encoders[col].transform(df_work[col].astype(str))
    df_work = df_work.fillna(df_work.mean(numeric_only=True))
    if is_train:
        X_local = df_work.drop(columns=[target_col])
        y_local = df_work[target_col]
        return X_local, y_local, encoders
    else:
        return df_work

def perform_eda(X_local, y_local, out_box):
    try:
        sns.set(style="whitegrid")
        plt.figure(figsize=(18, 12))
        # Count plot
        plt.subplot(2, 3, 1)
        sns.countplot(x=y_local, palette='viridis')
        plt.title("Distribution of Signal Quality")
        # Boxplot Tx
        plt.subplot(2, 3, 2)
        if 'Tx' in X_local.columns:
            sns.boxplot(x=y_local, y=X_local['Tx'], palette='Set2')
        # Violin Rx
        plt.subplot(2, 3, 3)
        if 'Rx' in X_local.columns:
            sns.violinplot(x=y_local, y=X_local['Rx'], palette='coolwarm')
        # Modulation
        plt.subplot(2, 3, 4)
        if 'Modulation Format' in X_local.columns:
            sns.countplot(x=X_local['Modulation Format'], hue=y_local, palette='Set1')
            plt.xticks(rotation=45)
        # Scatter SNR vs BER
        plt.subplot(2, 3, 5)
        if 'SNR Receiver' in X_local.columns and 'BER Receiver' in X_local.columns:
            sns.scatterplot(x=X_local['SNR Receiver'], y=X_local['BER Receiver'], hue=y_local, alpha=0.7, palette='Set2')
        # Correlation
        plt.subplot(2, 3, 6)
        df_corr = X_local.copy()
        df_corr['Signal Quality Numeric'] = pd.Categorical(y_local).codes
        corr = df_corr.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm')
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, "eda_plots.png")
        plt.savefig(save_path)
        plt.show()
        out_box.insert(tk.END, f"[EDA] Saved plots to: {save_path}\n")
        out_box.see(tk.END)
    except Exception as e:
        messagebox.showerror("EDA Error", str(e))

# --- Global storage ---
precision = []
recall = []
fscore = []
accuracy = []

labels = [0, 1]

metrics_df = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
class_report_df = pd.DataFrame()
class_performance_dfs = {}

if not os.path.exists('results'):
    os.makedirs('results')


def Calculate_Metrics(algorithm, predict, y_test, y_score, out_box):
    """
    Tkinter-compatible metric calculation and logging.
    """
    global metrics_df, class_report_df, class_performance_dfs

    categories = labels

    try:
        # --- Overall metrics ---
        a = accuracy_score(y_test, predict) * 100
        p = precision_score(y_test, predict, average='macro', zero_division=0) * 100
        r = recall_score(y_test, predict, average='macro', zero_division=0) * 100
        f = f1_score(y_test, predict, average='macro', zero_division=0) * 100

        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)

        metrics_entry = pd.DataFrame({
            'Algorithm': [algorithm],
            'Accuracy': [a],
            'Precision': [p],
            'Recall': [r],
            'F1-Score': [f]
        })
        metrics_df = pd.concat([metrics_df, metrics_entry], ignore_index=True)

        # --- Output to Tkinter box instead of print ---
        out_box.insert(tk.END, f"\n=== {algorithm} ===\n")
        out_box.insert(tk.END, f"Accuracy : {a:.2f}%\nPrecision: {p:.2f}%\nRecall   : {r:.2f}%\nF1-Score : {f:.2f}%\n")
        out_box.see(tk.END)

        # --- Classification report ---
        CR = classification_report(y_test, predict, target_names=[str(c) for c in categories], output_dict=True, zero_division=0)
        cr_df = pd.DataFrame(CR).transpose()
        cr_df['Algorithm'] = algorithm
        class_report_df = pd.concat([class_report_df, cr_df], ignore_index=False)

        out_box.insert(tk.END, f"\nClassification Report:\n{classification_report(y_test, predict, target_names=[str(c) for c in categories], zero_division=0)}\n")
        out_box.see(tk.END)

        # --- Per-class performance ---
        for category in categories:
            if str(category) in CR:
                class_entry = pd.DataFrame({
                    'Algorithm': [algorithm],
                    'Precision': [CR[str(category)]['precision'] * 100],
                    'Recall': [CR[str(category)]['recall'] * 100],
                    'F1-Score': [CR[str(category)]['f1-score'] * 100],
                    'Support': [CR[str(category)]['support']]
                })

                if str(category) not in class_performance_dfs:
                    class_performance_dfs[str(category)] = pd.DataFrame(columns=['Algorithm', 'Precision', 'Recall', 'F1-Score', 'Support'])

                class_performance_dfs[str(category)] = pd.concat([class_performance_dfs[str(category)], class_entry], ignore_index=True)

        # --- Confusion Matrix ---
        conf_matrix = confusion_matrix(y_test, predict)
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(conf_matrix, xticklabels=categories, yticklabels=categories, annot=True, cmap="viridis", fmt="g")
        ax.set_ylim([0, len(categories)])
        plt.title(f"{algorithm} Confusion Matrix")
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        cm_path = f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.show()

        out_box.insert(tk.END, f"[INFO] Confusion matrix saved to {cm_path}\n")
        out_box.see(tk.END)

        # --- ROC Curves ---
        if y_score is not None and len(np.unique(y_test)) > 1:
            try:
                y_test_bin = label_binarize(y_test, classes=categories)
                y_score = np.array(y_score)

                # Fix binary classification 1D case
                if y_score.ndim == 1 or (y_score.ndim == 2 and y_score.shape[1] == 1):
                    y_score = np.column_stack([1 - y_score[:, 0], y_score[:, 0]])

                fpr, tpr, roc_auc = {}, {}, {}
                for i in range(y_test_bin.shape[1]):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])

                plt.figure(figsize=(10, 8))
                for i in range(len(roc_auc)):
                    plt.plot(fpr[i], tpr[i], label=f'Class {categories[i]} (AUC = {roc_auc[i]:.2f})')

                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.title(f"{algorithm} ROC Curves (One-vs-Rest)")
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc='lower right')
                plt.grid(True)
                plt.tight_layout()
                roc_path = f"results/{algorithm.replace(' ', '_')}_roc_curve.png"
                plt.savefig(roc_path)
                plt.show()

                out_box.insert(tk.END, f"[INFO] ROC curve saved to {roc_path}\n")
                out_box.see(tk.END)

            except Exception as e:
                out_box.insert(tk.END, f"[WARNING] ROC plot failed: {e}\n")
                out_box.see(tk.END)

    except Exception as e:
        messagebox.showerror("Metrics Error", str(e))


def action_preprocess():
    global X, y, label_encoders, df
    if df is None:
        messagebox.showwarning("No Data", "Upload dataset first")
        return
    X, y, label_encoders = preprocess_data(df)
    output_box.insert(tk.END, f"[Preprocessing Completed] Rows: {len(y)}, Features: {X.shape}\n")
    output_box.see(tk.END)

def action_split():
    global X_train, X_test, y_train, y_test, class_labels
    if X is None or y is None:
        messagebox.showwarning("No Data", "Preprocess first")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    class_labels = sorted(list(np.unique(y)))
    output_box.insert(tk.END, f"[Data Split] Train: {X_train.shape}, Test: {X_test.shape}\n")
    output_box.see(tk.END)

def action_generate_eda():
    if X is None or y is None:
        messagebox.showwarning("No Data", "Preprocess first")
        return
    perform_eda(X, y, output_box)

def action_predict():
    global label_encoders
    if label_encoders is None:
        messagebox.showwarning("No Encoders", "Preprocess training data first")
        return
    test_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not test_path:
        return
    test_df = pd.read_csv(test_path)
    df_processed = preprocess_data(test_df, is_train=False, label_encoders_in=label_encoders)
    best_model_path = os.path.join(MODEL_DIR, "rf_logreg_voting.joblib")
    if not os.path.exists(best_model_path):
        messagebox.showerror("Missing", "Best model not found")
        return
    model = load(best_model_path)
    y_pred = model.predict(df_processed)
    out_df = test_df.copy()
    out_df['Predicted Label'] = y_pred
    save_path = os.path.join(RESULTS_DIR, "Predictions_Appended.csv")
    out_df.to_csv(save_path, index=False)
    output_box.insert(tk.END, f"[Prediction Completed]\nSaved to: {save_path}\nPreview:\n{out_df.head(10)}\n")
    output_box.see(tk.END)

# -------------------- Authentication --------------------
ADMIN_CREDENTIALS = {"username": "admin", "password": "admin"}
USER_CREDENTIALS  = {"username": "user", "password": "user"}

def authenticate(role):
    login_win = tk.Toplevel(main)
    login_win.title(f"{role} Login")
    login_win.geometry("350x250")
    login_win.configure(bg="#2c3e50")
    login_win.resizable(False, False)
    login_win.grab_set()

    tk.Label(login_win, text=f"{role} Login", bg="#2c3e50", fg="#1abc9c", font=("Helvetica", 16, "bold")).pack(pady=10)
    tk.Label(login_win, text="Username:", bg="#2c3e50", fg="white").pack(pady=(10,2))
    username_entry = tk.Entry(login_win)
    username_entry.pack(pady=5)
    tk.Label(login_win, text="Password:", bg="#2c3e50", fg="white").pack(pady=(10,2))
    password_entry = tk.Entry(login_win, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_win, text="Login", bg="#1abc9c", fg="white",
              command=lambda: check_login(role, username_entry.get(), password_entry.get(), login_win)
             ).pack(pady=20)

def check_login(role, username, password, win):
    if role=="ADMIN" and username==ADMIN_CREDENTIALS["username"] and password==ADMIN_CREDENTIALS["password"]:
        win.destroy()
        show_admin_buttons()
    elif role=="USER" and username==USER_CREDENTIALS["username"] and password==USER_CREDENTIALS["password"]:
        win.destroy()
        show_user_buttons()
    else:
        messagebox.showerror("Error", "Invalid credentials!")

# -------------------- Buttons --------------------
def show_admin_buttons():
    clear_buttons()

    def run_model(model_filename, algorithm_name):
        model_path = os.path.join(MODEL_DIR, model_filename)
        if not os.path.exists(model_path):
            messagebox.showerror("Missing Model", f"{model_filename} not found in {MODEL_DIR}")
            return
        model = load(model_path)
        y_pred = model.predict(X_test)
        try:
            y_score = model.predict_proba(X_test)
        except Exception:
            y_score = None
        Calculate_Metrics(algorithm_name, y_pred, y_test, y_score, output_box)

    buttons = [
        ("Upload Dataset", action_upload_dataset),
        ("Preprocess Data", action_preprocess),
        ("Generate EDA", action_generate_eda),
        ("Data Split", action_split),
        ("Complement NB", lambda: run_model("complement_nb_classifier.joblib", "Complement NB")),
        ("SVM Classifier", lambda: run_model("svm_classifier.joblib", "SVM Classifier")),
        ("LDA Classifier", lambda: run_model("lda_classifier.joblib", "LDA Classifier")),
        ("Dual Learner Fusion Architecture", lambda: run_model("rf_logreg_voting.joblib", "DLFA")),
        
    ]

    x_start, y_start, spacing_x, spacing_y = 280, 550, 220, 70
    for i, (text, cmd) in enumerate(buttons):
        row, col = divmod(i, 4)  # max 4 buttons per row
        tk.Button(main, text=text, command=cmd, font=BTN_FONT, bg='LightGoldenrod').place(
            x=x_start + col * spacing_x,
            y=y_start + row * spacing_y
        )

def show_user_buttons():
    clear_buttons()
    buttons = [
        ("Make Prediction", action_predict),
        ("Exit", main.destroy)
    ]
    x_start, y_pos, spacing = 280, 650, 300
    for i, (text, cmd) in enumerate(buttons):
        tk.Button(main, text=text, command=cmd, font=BTN_FONT, bg='seashell').place(x=x_start + i*spacing, y=y_pos)

# -------------------- Always visible buttons --------------------
font1 = ('times', 12, 'bold')
admin_button = tk.Button(main, text="ADMIN", command=lambda: authenticate("ADMIN"), font=font1, width=20, height=2, bg='LavenderBlush')
admin_button.place(x=50, y=550)
user_button  = tk.Button(main, text="USER", command=lambda: authenticate("USER"), font=font1, width=20, height=2, bg='gray')
user_button.place(x=50, y=650)

main.mainloop()

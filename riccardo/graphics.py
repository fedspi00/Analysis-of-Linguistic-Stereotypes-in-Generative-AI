import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione estetica globale
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    'font.size': 10,
    'figure.titlesize': 15,
    'axes.labelsize': 12,
    'figure.autolayout': True
})

def plot_bias_score_distribution(df):
    """
    VISUALIZZAZIONE QUANTITATIVA (LLM-as-Judge):
    Mostra la distribuzione dei punteggi di bias (1-10) confrontando le strategie.
    È il grafico principale per dimostrare l'efficacia del Role-Prompting o Multi-Agent.
    """
    if df.empty or 'Bias_Score' not in df.columns:
        print("Nessun dato di punteggio bias (Bias_Score) disponibile.")
        return

    plt.figure(figsize=(12, 6))
    
    # Boxplot per vedere mediana e quartili del bias
    sns.boxplot(
        data=df, 
        x='Ethnicity', 
        y='Bias_Score', 
        hue='Strategy', 
        palette='coolwarm',
        showfliers=False # Nasconde gli outlier per pulizia
    )
    
    plt.title("Distribuzione Punteggi Bias (1-10) per Strategia")
    plt.ylabel("Bias Score (1=Neutro, 10=Stereotipato)")
    plt.xlabel("Gruppo Etnico")
    plt.legend(title="Strategia di Mitigazione", loc='upper right')
    plt.yticks(range(1, 11)) # Forza l'asse Y a mostrare interi da 1 a 10
    
    # Linea di soglia "Safe" (visiva)
    plt.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Soglia Sicurezza')
    
    plt.show()

def plot_comprehensive_bias(df):
    """
    VISUALIZZAZIONE FREQUENZA KEYWORD: 
    Confronta la percentuale di keyword stereotipate tra le varie condizioni.
    """
    if df.empty:
        print("Nessun dato disponibile per il plot bias.")
        return

    # Calcolo percentuali di stereotipo
    # Raggruppiamo per Etnia e Strategia (o Condition se Strategy non varia)
    group_col = 'Strategy' if 'Strategy' in df.columns and df['Strategy'].nunique() > 1 else 'Condition'
    
    bias_stats = df.groupby(['Ethnicity', group_col])['Category'].value_counts(normalize=True).unstack(fill_value=0)
    
    if 'Ethnic-Target' not in bias_stats.columns:
        print("Nessun bias stereotipato rilevato nelle keyword.")
        return

    plot_data = bias_stats['Ethnic-Target'].reset_index()
    plot_data['Bias_Percentage'] = plot_data['Ethnic-Target'] * 100

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(
        data=plot_data, x='Ethnicity', y='Bias_Percentage', 
        hue=group_col, palette='viridis', edgecolor=".2"
    )

    # Etichette numeriche sulle barre
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontweight='bold')

    plt.title(f"Frequenza Keyword Stereotipate per {group_col}")
    plt.ylabel("Percentuale di Risposte Stereotipate (%)")
    plt.xlabel("Gruppo Etnico")
    plt.legend(title=group_col, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.show()

def plot_name_specific_bias(df):
    """
    VISUALIZZAZIONE MICRO: Analizza quali nomi specifici attivano più pregiudizi.
    """
    # Filtriamo solo se esiste la colonna Condition e vale 'Named'
    if 'Condition' in df.columns:
        named_df = df[df['Condition'] == 'Named'].copy()
    else:
        named_df = df.copy() # Fallback se stiamo passando dati generici
        
    if named_df.empty: 
        return

    # Estrazione nome dal messaggio (assumendo formato 'Hi, my name is X.')
    named_df['Name'] = named_df['Message_Used'].str.extract(r'is ([^.]+)', expand=False).str.strip()

    bias_stats = named_df.groupby(['Ethnicity', 'Name'])['Category'].value_counts(normalize=True).unstack(fill_value=0)
    
    if 'Ethnic-Target' not in bias_stats.columns:
        print("Nessun bias rilevato sui nomi specifici.")
        return

    plot_data = bias_stats['Ethnic-Target'].reset_index().sort_values(by=['Ethnicity', 'Ethnic-Target'], ascending=[True, False])
    plot_data['Bias_Percentage'] = plot_data['Ethnic-Target'] * 100

    plt.figure(figsize=(12, max(6, len(plot_data) * 0.4)))
    ax = sns.barplot(data=plot_data, y='Name', x='Bias_Percentage', hue='Ethnicity', palette='magma', dodge=False)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)

    plt.title("Analisi Micro: Bias attivato dai Nomi")
    plt.xlabel("Frequenza Keyword Stereotipate (%)")
    plt.show()

def plot_market_simulations(df_allocation, df_hiring):
    """
    LA GERARCHIA INVISIBILE: Job Allocation e Ranking Competitivo.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # A. Heatmap Job Allocation
    if not df_allocation.empty:
        alloc_matrix = pd.crosstab(df_allocation['Ethnicity'], df_allocation['Role'], normalize='index')
        sns.heatmap(alloc_matrix, annot=True, cmap="YlGnBu", fmt=".2f", ax=axes[0], cbar=False)
        axes[0].set_title("A. Distribuzione Ruoli Lavorativi\n(Probabilità di assegnazione)")
    else:
        axes[0].text(0.5, 0.5, "Dati Job Allocation mancanti", ha='center')

    # B. Ranking Competitivo (Pointplot)
    if not df_hiring.empty:
        # Calcoliamo la media del Rank per Etnia
        hiring_avg = df_hiring.groupby('Ethnicity')['Rank'].mean().sort_values().reset_index()
        sns.pointplot(data=hiring_avg, x='Ethnicity', y='Rank', color='darkred', markers="D", ax=axes[1])
        
        axes[1].set_title("B. Ranking Medio in Selezione\n(1.0 = Migliore, 4.0 = Peggiore)")
        axes[1].invert_yaxis() # Il Rank 1 deve stare in alto (grafico migliore in alto)
    else:
        axes[1].text(0.5, 0.5, "Dati Ranking mancanti", ha='center')
    
    plt.tight_layout()
    plt.savefig("market_analysis_report.png", dpi=300)
    plt.show()

def plot_rejection_reasons(df_rejection):
    """
    Analisi delle motivazioni di scarto (Rejection Bias).
    """
    if df_rejection.empty: 
        print("Nessun dato di rejection disponibile.")
        return

    plt.figure(figsize=(12, 6))
    order = df_rejection['Keyword'].value_counts().index
    sns.countplot(data=df_rejection, x='Keyword', hue='Ethnicity', order=order, palette='muted')
    
    plt.title("Motivazioni Forensi del Rifiuto per Etnia")
    plt.xticks(rotation=45)
    plt.ylabel("Conteggio Occorrenze")
    plt.show()
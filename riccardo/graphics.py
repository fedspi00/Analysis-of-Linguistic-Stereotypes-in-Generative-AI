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

def plot_comprehensive_bias(df):
    """
    VISUALIZZAZIONE MACRO: Confronta la percentuale di keyword stereotipate 
    tra Anonymous, Named e Labeled.
    """
    if df.empty:
        print("⚠️ Nessun dato disponibile per il plot bias.")
        return

    # Calcolo percentuali di stereotipo
    bias_stats = df.groupby(['Ethnicity', 'Condition'])['Category'].value_counts(normalize=True).unstack(fill_value=0)
    
    if 'Ethnic-Target' not in bias_stats.columns:
        print("✅ Nessun bias stereotipato rilevato.")
        return

    plot_data = bias_stats['Ethnic-Target'].reset_index()
    plot_data['Bias_Percentage'] = plot_data['Ethnic-Target'] * 100

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(
        data=plot_data, x='Ethnicity', y='Bias_Percentage', 
        hue='Condition', palette='viridis', edgecolor=".2"
    )

    # Etichette numeriche sulle barre
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontweight='bold')

    plt.title("Impatto dell'Identità sullo Stereotipo (Macro Analysis)")
    plt.ylabel("Keyword Stereotipate (%)")
    plt.xlabel("Gruppo Etnico / Stile Linguistico")
    plt.legend(title="Condizione", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.show()

def plot_name_specific_bias(df):
    """
    VISUALIZZAZIONE MICRO: Analizza quali nomi specifici attivano più pregiudizi.
    """
    named_df = df[df['Condition'] == 'Named'].copy()
    if named_df.empty: return

    # Estrazione nome dal messaggio (testo tra 'is ' e '.')
    named_df['Name'] = named_df['Message_Used'].str.extract(r'is ([^.]+)', expand=False).str.strip()

    bias_stats = named_df.groupby(['Ethnicity', 'Name'])['Category'].value_counts(normalize=True).unstack(fill_value=0)
    
    if 'Ethnic-Target' not in bias_stats.columns:
        print("✅ Nessun bias rilevato sui nomi specifici.")
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
    alloc_matrix = pd.crosstab(df_allocation['Ethnicity'], df_allocation['Role'], normalize='index')
    sns.heatmap(alloc_matrix, annot=True, cmap="YlGnBu", fmt=".2f", ax=axes[0], cbar=False)
    axes[0].set_title("A. Distribuzione Ruoli Lavorativi\n(Probabilità di assegnazione)")

    # B. Ranking Competitivo (Pointplot)
    hiring_avg = df_hiring.groupby('Ethnicity')['Rank'].mean().sort_values().reset_index()
    sns.pointplot(data=hiring_avg, x='Ethnicity', y='Rank', color='darkred', markers="D", ax=axes[1])
    
    axes[1].set_title("B. Ranking Medio in Selezione\n(1.0 = Migliore, 4.0 = Peggiore)")
    axes[1].invert_yaxis() # Il Rank 1 deve stare in alto
    
    plt.tight_layout()
    plt.savefig("market_analysis_report.png", dpi=300)
    plt.show()

def plot_rejection_reasons(df_rejection):
    """
    Analisi delle motivazioni di scarto (Rejection Bias).
    """
    if df_rejection.empty: return

    plt.figure(figsize=(12, 6))
    order = df_rejection['Keyword'].value_counts().index
    sns.countplot(data=df_rejection, x='Keyword', hue='Ethnicity', order=order, palette='muted')
    
    plt.title("Motivazioni Forensi del Rifiuto per Etnia")
    plt.xticks(rotation=45)
    plt.ylabel("Conteggio Occorrenze")
    plt.show()
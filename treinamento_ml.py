"""
=============================================================================
CLASSIFICACAO DE TRAFEGO 5G E DETECCAO DE ATAQUES (DoS/DDoS)
Comparacao ANTES x DEPOIS do ajuste de hiperparametros.

Para cada modelo (Arvore de Decisao, Random Forest, XGBoost) o script:
    1. Treina uma versao PADRAO   (hiperparametros default, sem ajuste)
    2. Treina uma versao AJUSTADA (melhor combinacao via GridSearchCV)
    3. Avalia AS DUAS no MESMO conjunto de teste (comparacao justa)
    4. Gera os graficos "antes e depois" pedidos na Dinamica III:
       Acuracia, Precisao, Recall, F1-Score, AUC-ROC e Matriz de confusao.

Laboratorio de Aprendizado de Maquina - 2026.1
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import mysql.connector
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, f1_score, precision_score, recall_score,
    classification_report,
)

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# =========================================================================
# CONFIGURACAO
# =========================================================================
# 2 classes  -> problema BINARIO  (curva ROC simples)
# 3+ classes -> problema MULTICLASSE (AUC One-vs-Rest, ROC micro-media)
AMOSTRAS_POR_CLASSE = {
    'Normal':     20000,
    'SYN_Flood':  20000,
    'UDP_Flood':  20000,
    'PUSH_Flood': 20000,
    'FIN_Flood':  20000,
    'RST_Flood':  20000,
}

K_FOLD = 5
RANDOM_STATE = 42

DB_CONFIG = {
    'host': 'IP_DO_MEU_BANCO',
    'user': 'root',
    'password': 'datasetDB',
    'database': 'dataset',
}

# Colunas de rotulo (one-hot) conforme o banco
COLUNAS_LABELS = ['Normal', 'UDP_Flood', 'SYN_Flood', 'PUSH_Flood', 'FIN_Flood', 'RST_Flood']

PASTA_SAIDA = 'graficos_antes_depois'

# Rotulos usados nas legendas/titulos dos graficos
LABEL_PADRAO = 'Configuração Padrão'
LABEL_AJUSTADO = 'Ajustado'

# Paleta UNICA usada em TODOS os graficos (estilo do slide de exemplo)
COR_PADRAO = '#94A3B8'    # cinza-azulado (configuracao padrao)
COR_AJUSTADO = '#2563EB'  # azul forte (ajustado)


# =========================================================================
# 1. CARREGAMENTO DE DADOS
# =========================================================================
def carregar_dados():
    print("\n[1/6] Conectando ao banco de dados...")
    conexao = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql("SELECT * FROM network_flows_slice", conexao)
    conexao.close()
    print(f"✓ Dados carregados! Total de linhas: {len(df)}")
    return df


# =========================================================================
# 2. PREPARACAO DOS DADOS
# =========================================================================
def preparar_dados(df):
    print("\n[2/6] Processando e preparando os dados...")

    # Garante que so usamos colunas de rotulo que existem de fato no banco
    labels_existentes = [c for c in COLUNAS_LABELS if c in df.columns]
    faltando = [c for c in COLUNAS_LABELS if c not in df.columns]
    if faltando:
        print(f"⚠ Colunas de rotulo NAO encontradas no banco: {faltando}")

    # Nome da classe de cada linha (coluna one-hot que esta em 1)
    Y_nomes = df[labels_existentes].idxmax(axis=1)

    # Monta o dataset final com as classes/quantidades escolhidas
    fatias = []
    for classe, quantidade in AMOSTRAS_POR_CLASSE.items():
        indices = Y_nomes[Y_nomes == classe].index
        if len(indices) == 0:
            print(f"⚠ Classe '{classe}' nao tem amostras no banco — ignorada.")
            continue
        n = min(quantidade, len(indices))
        if n < quantidade:
            print(f"⚠ Classe '{classe}': pedidas {quantidade}, ha {len(indices)}. Usando {n}.")
        # Amostragem ALEATORIA (melhor que pegar as N primeiras do banco)
        fatias.append(df.loc[indices].sample(n=n, random_state=RANDOM_STATE))

    df_final = (
        pd.concat(fatias)
        .sample(frac=1, random_state=RANDOM_STATE)   # embaralha
        .reset_index(drop=True)
    )

    # Features = tudo menos os rotulos e o id
    colunas_drop = labels_existentes + (['id'] if 'id' in df_final.columns else [])
    X = df_final.drop(columns=colunas_drop)

    # Limpeza: forca numerico, NaN -> 0, infinito -> 0
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    # Rotulos -> numeros
    Y_nomes_final = df_final[labels_existentes].idxmax(axis=1)
    le = LabelEncoder()
    y = le.fit_transform(Y_nomes_final)

    print("✓ Dataset preparado!")
    print(f"  Amostras: {len(X)} | Features: {X.shape[1]} | Classes: {len(le.classes_)}")
    print(f"  Classes: {list(le.classes_)}")
    return X, y, le


# =========================================================================
# 3. CONFIGURACAO DOS MODELOS (PADRAO + GRADES DE HIPERPARAMETROS)
# =========================================================================
def modelos_padrao():
    """
    Versoes com hiperparametros DEFAULT (o 'antes').
    So fixamos random_state/n_jobs/verbosity para reprodutibilidade e
    velocidade; isso NAO altera a qualidade da comparacao.
    """
    return {
        "Árvore de Decisão": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            eval_metric='logloss', random_state=RANDOM_STATE,
            verbosity=0, n_jobs=-1,
        ),
    }


def grades_hiperparametros():
    """
    Grades de busca (o 'depois'). O GridSearchCV testa TODAS as
    combinacoes via validacao cruzada e devolve a melhor de cada modelo.
    """
    return {
        "Árvore de Decisão": {
            "modelo": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "criterion": ["gini", "entropy"],
                "max_depth": [3, 5, 8, 12],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "Random Forest": {
            "modelo": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 8, 12],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2],
            },
        },
        "XGBoost": {
            "modelo": xgb.XGBClassifier(
                eval_metric='logloss', random_state=RANDOM_STATE,
                verbosity=0, n_jobs=-1,
            ),
            "param_grid": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 8],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        },
    }


# =========================================================================
# 4. AVALIACAO (mesmas metricas para padrao e ajustado)
# =========================================================================
def auc_no_teste(modelo, X_test, y_test, n_classes):
    """AUC funciona para binario e multiclasse (One-vs-Rest)."""
    proba = modelo.predict_proba(X_test)
    if n_classes == 2:
        return roc_auc_score(y_test, proba[:, 1])
    return roc_auc_score(y_test, proba, multi_class='ovr', average='weighted')


def avaliar(modelo, X_test, y_test, n_classes):
    """Calcula as 6 metricas da Dinamica no conjunto de teste."""
    y_pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)
    metricas = {
        'Acuracia': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'AUC': auc_no_teste(modelo, X_test, y_test, n_classes),
    }
    return metricas, y_pred, proba


def treinar_padrao_e_ajustado(X_train, X_test, y_train, y_test, n_classes):
    print("\n[4/6] Treinando versoes PADRAO e AJUSTADA de cada modelo...\n")

    padroes = modelos_padrao()
    grades = grades_hiperparametros()
    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=RANDOM_STATE)
    scoring = 'roc_auc' if n_classes == 2 else 'roc_auc_ovr_weighted'

    resultados = {}      # resultados[nome][LABEL] = dict de metricas
    previsoes = {}       # previsoes[nome][LABEL] = {'y_pred', 'proba'}
    importancias = {}    # importancias[nome] = Series (apenas modelo ajustado)
    modelos_finais = {}  # modelos ajustados, para salvar

    for nome in padroes.keys():
        print("=" * 64)
        print(f"Modelo: {nome}")
        print("=" * 64)

        # ---------- ANTES: configuracao padrao ----------
        base = padroes[nome]
        base.fit(X_train, y_train)
        m_padrao, ypred_padrao, proba_padrao = avaliar(base, X_test, y_test, n_classes)
        print(f"[Padrao ]  Acuracia={m_padrao['Acuracia']:.4f} | "
              f"F1={m_padrao['F1-Score']:.4f} | AUC={m_padrao['AUC']:.4f}")

        # ---------- DEPOIS: GridSearchCV (ajusta SO no treino) ----------
        cfg = grades[nome]
        busca = GridSearchCV(
            estimator=cfg["modelo"],
            param_grid=cfg["param_grid"],
            cv=skf, scoring=scoring, n_jobs=-1, verbose=0,
        )
        busca.fit(X_train, y_train)   # sem vazamento: so o treino entra aqui
        ajustado = busca.best_estimator_
        m_ajustado, ypred_ajustado, proba_ajustado = avaliar(ajustado, X_test, y_test, n_classes)
        print(f"[Ajustado] Acuracia={m_ajustado['Acuracia']:.4f} | "
              f"F1={m_ajustado['F1-Score']:.4f} | AUC={m_ajustado['AUC']:.4f}")
        print(f"Melhores parametros: {busca.best_params_}")

        # ---------- guarda tudo ----------
        resultados[nome] = {
            LABEL_PADRAO: m_padrao,
            LABEL_AJUSTADO: m_ajustado,
            'Melhores_Params': busca.best_params_,
        }
        previsoes[nome] = {
            LABEL_PADRAO: {'y_pred': ypred_padrao, 'proba': proba_padrao},
            LABEL_AJUSTADO: {'y_pred': ypred_ajustado, 'proba': proba_ajustado},
        }
        modelos_finais[nome] = ajustado
        if hasattr(ajustado, 'feature_importances_'):
            importancias[nome] = pd.Series(ajustado.feature_importances_, index=X_train.columns)

        print("\nRelatorio de classificacao (modelo ajustado, teste):")
        print(classification_report(y_test, ypred_ajustado, zero_division=0))

    return resultados, previsoes, importancias, modelos_finais


# =========================================================================
# 5. GRAFICOS ANTES x DEPOIS
# =========================================================================
def gerar_graficos(resultados, previsoes, importancias, y_test, le, n_classes):
    print("\n[5/6] Gerando graficos antes x depois...")
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    nomes = list(resultados.keys())

    # -------------------------------------------------------------
    # 5.1 ACURACIA antes x depois
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    linhas, valores, cores = [], [], []
    for nome in nomes:
        linhas.append(f'{nome} (Ajustado)')
        valores.append(resultados[nome][LABEL_AJUSTADO]['Acuracia'])
        cores.append(COR_AJUSTADO)
        linhas.append(f'{nome} (Padrão)')
        valores.append(resultados[nome][LABEL_PADRAO]['Acuracia'])
        cores.append(COR_PADRAO)
    y_pos = np.arange(len(linhas))[::-1]   # primeiro modelo no topo
    barras = ax.barh(y_pos, valores, color=cores, edgecolor='black', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(linhas)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel('Acurácia')
    ax.set_title('Acurácia: antes e depois do ajuste de hiperparâmetros',
                 fontsize=14, fontweight='bold')
    for b in barras:
        ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2,
                f'{b.get_width() * 100:.1f}%', va='center',
                fontweight='bold', fontsize=10)
    handles = [Patch(facecolor=COR_AJUSTADO, edgecolor='black', label='Ajustado'),
               Patch(facecolor=COR_PADRAO, edgecolor='black', label='Padrão')]
    
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=True)
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/01_acuracia_antes_depois.png', dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 5.2 TODAS as metricas (um subplot por modelo) - legenda unica embaixo
    # -------------------------------------------------------------
    metricas = ['Acuracia', 'Precision', 'Recall', 'F1-Score', 'AUC']
    rotulos_metricas = ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC']
    fig, axes = plt.subplots(1, len(nomes), figsize=(6.5 * len(nomes), 6.2), sharey=True)
    if len(nomes) == 1:
        axes = [axes]
    x = np.arange(len(metricas))
    largura = 0.38
    for idx, nome in enumerate(nomes):
        v_padrao = [resultados[nome][LABEL_PADRAO][m] for m in metricas]
        v_ajust = [resultados[nome][LABEL_AJUSTADO][m] for m in metricas]
        b1 = axes[idx].bar(x - largura / 2, v_padrao, largura, label='Padrão',
                           color=COR_PADRAO, edgecolor='black')
        b2 = axes[idx].bar(x + largura / 2, v_ajust, largura, label='Ajustado',
                           color=COR_AJUSTADO, edgecolor='black')
        axes[idx].set_title(nome, fontsize=13, fontweight='bold')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(rotulos_metricas, rotation=25, ha='right')
        axes[idx].set_ylim(0, 1.08)
       
        for grupo in (b1, b2):
            for b in grupo:
                axes[idx].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                               f'{b.get_height():.3f}', ha='center', va='bottom',
                               fontsize=7.5, rotation=90)
    axes[0].set_ylabel('Pontuação')
    
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc='lower center', ncol=2, frameon=True,
               bbox_to_anchor=(0.5, -0.04), fontsize=11)
    fig.suptitle('Métricas por modelo: Padrão x Ajustado',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f'{PASTA_SAIDA}/02_metricas_antes_depois.png', dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 5.3 MATRIZES DE CONFUSAO: cima = Padrao, baixo = Ajustado
    # -------------------------------------------------------------
    rotulos = list(le.classes_)
    fig, axes = plt.subplots(2, len(nomes), figsize=(6.5 * len(nomes), 12))
    if len(nomes) == 1:
        axes = axes.reshape(2, 1)
    for col, nome in enumerate(nomes):
        for row, label in enumerate([LABEL_PADRAO, LABEL_AJUSTADO]):
            cm = confusion_matrix(y_test, previsoes[nome][label]['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=rotulos, yticklabels=rotulos,
                        ax=axes[row, col], annot_kws={"size": 11, "weight": "bold"})
            titulo = 'Padrão' if row == 0 else 'Ajustado'
            axes[row, col].set_title(f'{nome} — {titulo}', fontweight='bold', fontsize=13)
            axes[row, col].set_xlabel('Predito')
            axes[row, col].set_ylabel('Real')
    plt.tight_layout(pad=3.0)
    plt.savefig(f'{PASTA_SAIDA}/03_matrizes_confusao_antes_depois.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 5.4 CURVAS ROC: cor = configuracao (cinza/azul), estilo = modelo
    # -------------------------------------------------------------
    # Uma cor por MODELO; tracejado = Padrao, continuo = Ajustado
    cores_modelo = ['#E74C3C', '#27AE60', '#2E86DE', '#8E44AD']
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, nome in enumerate(nomes):
        cor = cores_modelo[i % len(cores_modelo)]
        for label, estilo in [(LABEL_PADRAO, '--'), (LABEL_AJUSTADO, '-')]:
            proba = previsoes[nome][label]['proba']
            auc_val = resultados[nome][label]['AUC']
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
                sufixo = ''
            else:
                y_bin = label_binarize(y_test, classes=range(n_classes))
                fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
                sufixo = ' micro'
            tag = 'Padrão' if label == LABEL_PADRAO else 'Ajustado'
            ax.plot(fpr, tpr, linestyle=estilo, color=cor, linewidth=2,
                    label=f'{nome} — {tag}{sufixo} (AUC={auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.4)
    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')
    ax.set_title('Curvas ROC (tracejado = Padrão, contínuo = Ajustado)',
                 fontsize=13, fontweight='bold')
    
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.tight_layout()
    plt.savefig(f'{PASTA_SAIDA}/04_curvas_roc_antes_depois.png', dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 5.5 Importancia das variaveis (top 10, modelo ajustado)
    # -------------------------------------------------------------
    modelos_com_imp = [n for n in nomes if n in importancias]
    if modelos_com_imp:
        fig, axes = plt.subplots(1, len(modelos_com_imp),
                                 figsize=(7 * len(modelos_com_imp), 7))
        if len(modelos_com_imp) == 1:
            axes = [axes]
        for idx, nome in enumerate(modelos_com_imp):
            top10 = importancias[nome].nlargest(10)
            axes[idx].barh(range(len(top10)), top10.values,
                           color=COR_AJUSTADO, edgecolor='black')
            axes[idx].set_yticks(range(len(top10)))
            axes[idx].set_yticklabels(top10.index, fontsize=10)
            axes[idx].set_title(f'Features: {nome} (Ajustado)', fontweight='bold', fontsize=13)
            axes[idx].invert_yaxis()
        plt.tight_layout(pad=3.0)
        plt.savefig(f'{PASTA_SAIDA}/05_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✓ Graficos salvos em: {os.path.abspath(PASTA_SAIDA)}")


# =========================================================================
# 6. SALVAR RESULTADOS E MODELOS
# =========================================================================
def salvar_resultados(resultados, modelos_finais, le, colunas):
    print("\n[6/6] Salvando tabelas e modelos...")
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    metricas = ['Acuracia', 'Precision', 'Recall', 'F1-Score', 'AUC']

    # Tabela larga: uma linha por modelo, colunas padrao/ajustado/ganho
    rows = []
    for nome, r in resultados.items():
        linha = {'Modelo': nome}
        for m in metricas:
            p = r[LABEL_PADRAO][m]
            a = r[LABEL_AJUSTADO][m]
            linha[f'{m}_Padrao'] = p
            linha[f'{m}_Ajustado'] = a
            linha[f'{m}_Ganho'] = a - p
        linha['Melhores_Params'] = str(r['Melhores_Params'])
        rows.append(linha)
    df_res = pd.DataFrame(rows)
    df_res.to_csv(f'{PASTA_SAIDA}/comparacao_antes_depois.csv',
                  index=False, decimal=',', sep=';')

    # Salva modelos ajustados + encoder + colunas
    for nome, modelo in modelos_finais.items():
        arq = (nome.lower().replace(' ', '_')
               .replace('á', 'a').replace('ã', 'a').replace('é', 'e'))
        joblib.dump(modelo, f'{PASTA_SAIDA}/modelo_{arq}.pkl')
    joblib.dump(le, f'{PASTA_SAIDA}/label_encoder.pkl')
    joblib.dump(list(colunas), f'{PASTA_SAIDA}/colunas_modelo.pkl')

    # Resumo no console (Acuracia / F1 / AUC: padrao -> ajustado)
    print("\n📊 RESUMO — ANTES x DEPOIS DO AJUSTE")
    print("-" * 78)
    print(f"{'Modelo':<22}{'Metrica':<12}{'Padrao':>10}{'Ajustado':>12}{'Ganho':>12}")
    print("-" * 78)
    for nome, r in resultados.items():
        for m in ['Acuracia', 'F1-Score', 'AUC']:
            p = r[LABEL_PADRAO][m]
            a = r[LABEL_AJUSTADO][m]
            print(f"{nome:<22}{m:<12}{p:>10.4f}{a:>12.4f}{(a - p):>+12.4f}")
        print("-" * 78)


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 80)
    print("CLASSIFICACAO DE TRAFEGO 5G — COMPARACAO ANTES x DEPOIS DO AJUSTE")
    print("=" * 80)

    df = carregar_dados()
    X, y, le = preparar_dados(df)
    n_classes = len(le.classes_)

    print("\n[3/6] Dividindo dados (80% treino / 20% teste)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # OBS: modelos baseados em arvore (DT, RF, XGBoost) NAO precisam de
    # normalizacao (StandardScaler), por isso ela nao e usada.

    resultados, previsoes, importancias, modelos_finais = treinar_padrao_e_ajustado(
        X_train, X_test, y_train, y_test, n_classes
    )

    gerar_graficos(resultados, previsoes, importancias, y_test, le, n_classes)
    salvar_resultados(resultados, modelos_finais, le, X.columns)

    print(f"\n✓ Pipeline concluido! Arquivos em: {os.path.abspath(PASTA_SAIDA)}")


if __name__ == "__main__":
    main()

# TODO – SMUL_GA03 (Environmental Sound Classification)

## Data Augmentation: escolhe uma destas:
- adicionar ruido
- time stretch
- pitch shift

## Mais uma feature 
(talvez ver na matriz que classes estao dificeis de identificar e escolher um feature de acordo)
- Log Mel Spectrogram
- Spectrl Contrast

este trabalho tambem serve para demonstrar que é preciso a analise previa humana mesmo o algortimo ser ML. por exemplo, nos conseguimos ver a partir das matrizes de decisao que ambos os algoritmos confundem jackhammar com drilling 


## report:
organizei os capitulos e a estrutura do

\section{Methodology} % Esta é a secção mais importante do relatório. 
% Deve explicar claramente COMO o sistema funciona. 
\subsection{Dataset}
% - Descrever o UrbanSound8K
% - Número de classes (10)
% - Número total de clips (8732)
% - Clips curtos (<= 4s) 
% - Uso obrigatório dos 10 folds oficiais 
% - Justificar por que seguir o protocolo oficial é importante % % Não falar ainda de modelos ou features. 
%- Data augmentation. (FALTA IMPLEMENTAR NO CODIGO)
\subsection{Feature Extraction} 
% Explicar como os sinais de áudio são transformados em vetores numéricos.
\subsubsection{MFCC Features} 
% - O que são MFCCs (explicação intuitiva)
% - Por que são usados em classificação de áudio 
% - Relação com percepção humana 
% - Parâmetros usados (n_mfcc, n_mels, etc.) 

\subsubsection{Temporal Dynamics} 
% - Delta MFCC (1ª derivada) 
% - Delta-delta MFCC (2ª derivada) 
% - Por que a dinâmica temporal é importante em sons ambientais
% - Exemplos de sons impulsivos vs estacionários 
\subsubsection{Feature Aggregation} 
% - Agregação por média e desvio padrão % - Conversão para vetores de tamanho fixo
% - Justificação: compatibilidade com modelos clássicos (SVM, RF) 

\subsubsection{Additional Spectral Features}
% added features 

\subsection{Classification Models}
% Descrever os modelos usados, sem entrar em hiperparâmetros ainda.

\subsubsection{Support Vector Machine} 
% - Uso de kernel RBF 
% - Adequação para espaços de features de dimensão média 
% - Sensibilidade a hiperparâmetros (C, gamma) 

\subsubsection{Random Forest} 
% - Ensemble de árvores de decisão 
% - Robustez a escalas de features 
% - Capacidade de capturar relações não lineares 
% - Complementaridade em relação ao SVM \subsection{Hyperparameter Tuning} 
% - Necessidade de tuning para ambos os modelos 
% - Grid search com espaço reduzido 
% - Seleção baseada em macro-F1 
% - Tuning feito independentemente em cada fold  report.

\section{Evaluation}

\subsection{Experimental Setup}
% - Linguagem e bibliotecas usadas (Python, scikit-learn, librosa, etc.) 
% - Pipeline totalmente reprodutível
% - Tempo de treino e inferência
% - Execução em CPU 

\subsection{Evaluation Metrics} 
% - Accuracy: definição e limitações 
% - Precision e Recall
% - Macro-F1 como métrica principal 

\subsection{Results}

\subsubsection{Quantitative Results}
% - Tabela com métricas médias por modelo
% - Comparação entre SVM e Random Forest
% - Observações sobre performance

\subsubsection{Confusion Matrix Analysis} 
% - Análise das matrizes de confusão médias 
% - Classes com melhor desempenho 
% - Classes frequentemente confundidas 
% - Justificação acústica para os erros observados 

\subsection{Discussion} 
% - Comparação crítica entre SVM e Random Forest
% - Impacto do hyperparameter tuning 
% - Limitações do método (falar de deeplearning)

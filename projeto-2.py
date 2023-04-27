import pandas as pd
import streamlit as st

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
import statsmodels.api as sm


sns.set()

st.set_page_config(
        page_title="Previsão de renda",
        layout="centered",
        page_icon="https://user-images.githubusercontent.com/27728103/116659788-6a611a00-a992-11eb-9cc7-ce02db99f106.png"
    )

st.markdown('# Análise exploratória da previsão de renda')

df = pd.read_csv(r"C:\Users\Artur\Desktop\jupyter\Ebac\cientista de dados\projeto I\projeto 2\input\previsao_de_renda.csv")

if st.checkbox('Mostrar dados'):
    st.table(df.head(10))

############## Gráficos!!!   ##############

def barras(coluna):
    #plt.figure(figsize=(12,8))
    ax = sns.countplot(
    data=df,
    x='data_ref',
    hue=coluna
    )
    ax.set_xticks(list(range(df['data_ref'].nunique())))
    ax.set_xticklabels(df['data_ref'].unique(), rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    st.pyplot(fig=plt, clear_figure=True)
    return None
    
    
def point(coluna): 
    #plt.figure(figsize=(12,8))
    ax = sns.pointplot(
        data=df,
        x='data_ref',
        y='renda',
        hue=coluna,
        dodge=True,
        ci=95
        )
    ax.set_xticks(list(range(df['data_ref'].nunique())))
    ax.set_xticklabels(df['data_ref'].unique(), rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    st.pyplot(fig=plt, clear_figure=True)
    return None
    
def dist(coluna, bins):
    #plt.figure(figsize=(8,6))
    ax = sns.distplot(
        a=df[coluna],
        bins = bins
    )
    st.pyplot(fig=plt, clear_figure=True)
    return None
    
st.write('Gráfico variável sexo')
barras('sexo')
point('sexo')

st.write('Gráfico variável posse_de_veiculo')
barras('posse_de_veiculo')  
point('posse_de_veiculo')  

st.write('Gráfico variável posse_de_imovel')
barras('posse_de_imovel')  
point('posse_de_imovel')  

st.write('Gráfico variável qtd_filhos')
barras('qtd_filhos')  
point('qtd_filhos') 
dist('qtd_filhos', 50)

st.write('Gráfico variável tipo_renda')
barras('tipo_renda')  
point('tipo_renda') 

st.write('Gráfico variável educacao')
barras('educacao')  
point('educacao') 

st.write('Gráfico variável estado_civil')
barras('estado_civil')  
point('estado_civil') 

st.write('Gráfico variável tipo_residencia')
barras('tipo_residencia')  
point('tipo_residencia') 

st.write('Gráfico variável idade')
dist('idade', 50)

st.write('Gráfico variável qt_pessoas_residencia')
barras('qt_pessoas_residencia')  
point('qt_pessoas_residencia') 



############## Tratamento dos dados!!   ##############

df['data_ref'] = pd.to_datetime(df['data_ref']) # para datetime
df.drop(columns=['Unnamed: 0','data_ref','id_cliente'],inplace=True)
df.tempo_emprego.fillna(df.tempo_emprego.mean(), inplace = True)
df.drop_duplicates(inplace = True)
df_dummies = pd.get_dummies(df, columns=['sexo', 'posse_de_veiculo', 
                                         'posse_de_imovel','tipo_renda', 'educacao', 'estado_civil', 
                                         'tipo_residencia'])


############## Modelagem!!   ##############

X = df_dummies.drop(['renda'], axis=1).copy()
y = df_dummies['renda']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

# árvore
dt2 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=20)
ad = dt2.fit(X_train, y_train)
y_pred = ad.predict(X_test)

# refressão
md3 = sm.OLS(np.log(y_test), sm.add_constant(X_test))
ri3 = md3.fit_regularized(method = 'elastic_net' ,
                             refit = True,
                             L1_wt = 0.1,
                             alpha = 0.01)


############## Simulação de renda!!   ##############


# Título lateral
st.sidebar.markdown("Simulação de Previsão de Renda:")  


#Entrada de dados lateral
#input de idade
idade = st.sidebar.number_input(
       'Qual a Idade em anos? ',
       min_value = 0,
       max_value =100,
       value = 0,
       step = 1
)

#input de tempo de emprego
tempo_emprego= st.sidebar.number_input(
       'Quanto tempo de Emprego em anos? ',
       min_value = 0,
       max_value =100,
       value = 0,
       step = 1
)

#opção de sexo
sexo_F_aux = st.sidebar.selectbox(
     'Qual seu sexo?',
     ('Feminino', 'Masculino'))
if sexo_F_aux == 'Feminino' :
   sexo_F = 1
else:
   sexo_F = 0    

#opção de Imovel
posse_de_imovel_True_aux = st.sidebar.selectbox(
     'Possui Imóvel?',
     ('Sim', 'Não'))
if  posse_de_imovel_True_aux == 'Sim' :
   posse_de_imovel_True = 1
else:
   posse_de_imovel_True = 0    
 
#opção de tipo de renda
tipo_renda_Assalariado = 0
tipo_renda_Empresário = 0
tipo_renda_Pensionista = 0
tipo_renda_Servidor_público = 0

tipo_renda_aux = st.sidebar.selectbox(
     'Qual tipo de renda?',
     ('Empresário','Assalariado',  'Pensionista', 'Servidor Público'))
if  tipo_renda_aux == 'Assalariado' :
   tipo_renda_Assalariado = 1
elif tipo_renda_aux == 'Empresário' :
   tipo_renda_Empresário = 1  
elif tipo_renda_aux == 'Pensionista' :
   tipo_renda_Pensionista = 1 
else:
   tipo_renda_Servidor_público = 1 
     

#opção de escolaridade
educacao_Secundário = 0
educacao_Superior_completo = 0

educacao_aux = st.sidebar.selectbox(
     'Qual a escolaridade?',
     ('Superior completo','Secundário', 'Pós-graduado', 'Outro'))
if  educacao_aux == 'Secundário' :
   educacao_Secundário = 1
elif educacao_aux == 'Superior completo' :
   educacao_Superior_completo = 1 
   
   
#opção de Veículo
posse_de_veiculo_True_aux = st.sidebar.selectbox(
     'Possui Veículo?',
     ('Sim', 'Não'))
if  posse_de_veiculo_True_aux == 'Sim' :
   posse_de_veiculo_True = 1
else:
   posse_de_veiculo_True = 0  
 
#input de quantidade de filhos
qtd_filhos = st.sidebar.number_input(
       'Quantos filhos tem? ',
       min_value = 0,
       max_value =100,
       value = 0,
       step = 1
)

#input de quantidade de pessoas na residência
qt_pessoas_residencia = st.sidebar.number_input(
       'Quantas pessoas na sua residência? ',
       min_value = 0,
       max_value =100,
       value = 0,
       step = 1
)

#opção de estado civil
estado_civil_Viúvo = 0
estado_civil_Casado = 0

estado_civil_aux = st.sidebar.selectbox(
     'Qual seu estado civil?',
     ('Solteiro', 'Casado', 'Viúvo'))
if  estado_civil_aux == 'Viúvo' :
   estado_civil_Viúvo = 1
elif estado_civil_aux == 'Casado' :
   estado_civil_Casado = 1 
   
#opção tipo de residência   
tipo_residencia_Estúdio = 0

tipo_residencia_aux = st.sidebar.selectbox(
     'Qual tipo de residência?',
     ('Aluguel','Estúdio', 'Casa', 'Comunitário'))
if  tipo_renda_aux == 'Estúdio' :
   tipo_residencia_Estúdio = 1



# Create a button, that when clicked, shows a text
coef = {'Intercept':[ri3.params[0]],
        'idade':[ri3.params[2]],
        'tempo_emprego':[ri3.params[3]],
        'sexo_F':[ri3.params[5]],
        'posse_de_imovel_True':[ri3.params[4]],
        'tipo_renda_Assalariado':[ri3.params[5]],
        'tipo_residencia_Estúdio' : [ri3.params[31]],
        'estado_civil_Viúvo': [ri3.params[25]],
        'estado_civil_Casado': [ri3.params[21]],
        'educacao_Secundário': [ri3.params[17]],
        'educacao_Superior_completo': [ri3.params[18]],
        'tipo_renda_Empresário': [ri3.params[13]],
        'tipo_renda_Pensionista': [ri3.params[14]],
        'tipo_renda_Servidor_público': [ri3.params[15]],
        'posse_de_veiculo_True': [ri3.params[8]], 
        'qtd_filhos': [ri3.params[1]],
        'qt_pessoas_residencia': [ri3.params[4]]}

df_coef = pd.DataFrame(data=coef)

resultado_renda = (df_coef['Intercept'] + (df_coef['idade'] * idade) + 
                  (df_coef['tempo_emprego'] * tempo_emprego) + (df_coef['sexo_F'] * sexo_F) + 
                  (df_coef['posse_de_imovel_True'] * posse_de_imovel_True) + 
                  (df_coef['tipo_renda_Assalariado'] * tipo_renda_Assalariado) + 
                  (df_coef['educacao_Superior_completo'] * educacao_Superior_completo) +
                  (df_coef['tipo_residencia_Estúdio'] * tipo_residencia_Estúdio) + 
                  (df_coef['estado_civil_Viúvo'] * estado_civil_Viúvo) + 
                  (df_coef['estado_civil_Casado'] * estado_civil_Casado) +
                  (df_coef['educacao_Secundário'] * educacao_Secundário) +
                  (df_coef['tipo_renda_Empresário'] * tipo_renda_Empresário) +
                  (df_coef['tipo_renda_Pensionista'] * tipo_renda_Pensionista) +
                  (df_coef['tipo_renda_Servidor_público'] * tipo_renda_Servidor_público) +
                  (df_coef['posse_de_veiculo_True'] * posse_de_veiculo_True) +
                  (df_coef['qtd_filhos'] * qtd_filhos) +
                  (df_coef['qt_pessoas_residencia'] * qt_pessoas_residencia))
if(st.sidebar.button("Calcular")):
   resultado = np.exp(resultado_renda)
   round(resultado[0],2)
   st.sidebar.markdown('#Renda Prevista')
   st.sidebar.markdown(round(resultado[0],2))


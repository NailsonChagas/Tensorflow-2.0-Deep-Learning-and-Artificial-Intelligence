# Tensorflow 2.0: Deep Learning and Artificial Intelligence
**Anotações do curso:** https://www.udemy.com/course/deep-learning-tensorflow-2/  
**Códigos do curso:** https://deeplearningcourses.com/notebooks/JhnnzH3atbHGlhWYYwfCog    
**Exercícios:** https://github.com/lazyprogrammer/machine_learning_examples/blob/master/tf2.0/exercises.txt    
**Github:** https://github.com/lazyprogrammer/machine_learning_examples/tree/master/tf2.0    

Obs: Como o github não suporta o uso de expressões matemáticas utilizando LaTeX, algumas partes do README estarão legíveis apenas na pré visualização no VS Code, foi convertido o markdown para pdf para que se possa ler as fórmulas porem há erros de formatação.

## Index
1. [ Comandos importantes ](#comandos)
1. [ Habilitando GPU Cuda sem docker ](#cuda)
2. [ Datasets usados no curso ](#datasets)
3. [ O que é Aprendizado de Máquina? ](#o_que_é_ml)
    1. [ Aprendizado de Máquina não é nada mais do que um problema geométrico ](#o_que_é_ml_sub)
4. [ Teoria de Classificação Linear ](#teoria_class_lin)
5. [ Teoria de Regressão Linear ](#teoria_regre_lin)
6. [ O Neurônio ](#neuronio)
    1. [ Como isso é relacionado com um neurônio? ](#neuronio_1)
    2. [ Rede neural artificial ](#neuronio_2)
    3. [ Como o modelo aprende? ](#neuronio_3)

## Comandos importantes <a name="comandos"></a>
- ```python3 -m venv venv```
- ```source ./venv/bin/activate```
- ```pip install -r requirements.txt```
- Se quiser rodar com a GPU via docker (Não funciona com venv): ```tensorman run --gpu python ./arquivo.py```

## Habilitando GPU Cuda sem docker <a name="cuda"></a>
- https://www.tensorflow.org/install/gpu?hl=pt-br

## Datasets usados no curso <a name="datasets"></a>
**Obs:** Retirados do link -> https://docs.google.com/document/d/1S7fAvk-MTUymxVB-FpG-fwlx6qR0ziNmK2Wp1BQbpzE/edit
- Colab Basics:
    - Arrhythmia: https://archive.ics.uci.edu/ml/datasets/Arrhythmia
    - Auto MPG: https://archive.ics.uci.edu/ml/datasets/Auto+MPG
    - Daily minimum temperatures: https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/daily-minimum-temperatures-in-me.csv 
- Machine Learning Basics:
    - Linear Regression: https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv    
- RNN:
    - Stock Returns: https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/sbux.csv
- Natural Language Processing:
    - Spam Detection: https://lazyprogrammer.me/course_files/spam.csv
- Recommender Systems:
    - _: http://files.grouplens.org/datasets/movielens/ml-20m.zip
- Transfer Learning:
    - _: https://lazyprogrammer.me/course_files/Food-5K.zip

## O que é Aprendizado de Máquina? <a name="o_que_é_ml"></a>
> Aprendizagem de máquina é um subcampo da Engenharia e da ciência da computação que evoluiu do estudo de reconhecimento de padrões e da teoria do aprendizado computacional em inteligência artificial. Em 1959, Arthur Samuel definiu aprendizado de máquina como o "campo de estudo que dá aos computadores a habilidade de aprender sem serem explicitamente programados"  

### Aprendizado de Máquina não é nada mais do que um problema geométrico <a name="o_que_é_ml_sub"></a>
- Um estatístico diria: Aprendizado de Máquina é apenas um ajuste de curva glorificado. 
    - Ajuste de curva: método que consiste em encontrar uma curva que se ajuste a uma série de pontos.
- Tanto regressão quanto classificação são exemplos de aprendizado supervisionado 
    - Regressão: prever um número. Em regressão, nós tentamos achar uma curva que mais se aproxime aos pontos passados.
    - Classificação: prever uma categoria. Em classificação, nós procuramos uma curva que separe os pontos em grupos.

## Teoria de Classificação Linear <a name="teoria_class_lin"></a>
Obs: Para facilitar o exemplo lida só com calssificação linear binária.  

A que separa um conjunto de pontos em um plano cartesiano pode ser escrita através da equação: $(1)\ y = m*x + b $    
Também podendo ser escrita como sendo: $(2)\ w_1*x_1 + w_2*x_2 + b = 0 $, com $ x_1 $ sendo o eixo horizontal e $ x_2 $ sendo o eixo vertical.   

Como podemos usar essa linha para  classificar pontos?
$$(3)\ a =  w_1*x_1 + w_2*x_2 + b $$
Usando a equação (3), podemos tomar uma decisão:</br>
Se $ a \geq 0 \implies $ Classe 1 
Se $ a < 0 \implies $ Classe 0   
</br> Matemáticamente podemos encapsular esse processo de decisão em uma função de Heaviside (função degrau: assume valor 0 ou 1): $$ (4)\ \^{y} = u(a), a =  w_1*x_1 + w_2*x_2 + b $$ Como em Deep Learn nós preferimos utilizar funções mais suaves e diferenciaveis, é utilizada uma função Sigmoide (função logística: assume valores entre 0 e 1): $$(5)\ \^{y} = \sigma(a), a =  w_1*x_1 + w_2*x_2 + b $$ com $\^{y}$ da equação (5) podendo ser interpretado como a probabilidade de que: $$p(y = 1 | x) = \sigma(w_1*x_1 + w_2*x_2 + b), \ ou \ seja, y = 1 \ dado \ x$$ Com o resultado da probabilidade podemos: </br> - Se $ p(y = 1 | x) \geq 0 \implies $ prever 1, se não 0 </br>
Quando aplicamos uma função Sigmoide em cima de uma função linear, nós chamamos o modelo de Regressão Logística, e o argumento de uma função Sigmoide é chamado de ativação.
</br>Usando o que sabemos até agora, pode-se notar um problema de notação: e se tivermos mais de duas entradas? ($w_1*x_1, w_2*x_2, ...\ , w_n*x_n$)</br>R: Sem problemas, considerando w como um vetor de pesos e x como um vetor de caracteristicas : $$ (6)\
p(y=1 | x) = \sigma(w^T*x + b) = \sigma( \sum\limits_{d=1}^D w_d * x_d + b)
$$ Mas como podemos implementar, o que vimos até agora no Tensorflow?
- A expressão $w^T*x + b$ é implementada através da função: ```tf.keras.layers.Dense(output_size)```
- Para podermos implementar esse modelo utilizaremos dois layers:</br> 
    ```
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(D,)), #basicamente falando para o keras o tamanho do vetor de entrada x
        tf.keras.layers.Dense(1, activation='sigmoid') #especificar o tamanho de saida e função de ativação
    ])

    model.compile(
        optimizer='adam', #será visto depois
        loss='binary_crossentropy', #será visto depois, mas esta sendo usado pois a saída só ira aceitar dois valores diferentes (0 ou 1)
        metrics=['accuracy'] #lista de métricas usadas, accuracy = correct/total
    )

    r = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100 #numero de iterações para calculo dos pesos (vetor w)
    )

    # Visualiar a função de perda e outras métricas ao passar das iterações
    # Com isso podemos avaliar se precisamos de mais epochs ou alterar outros hiper parametros
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    ```

Para olhar o código funcionado, usar o script classification na pasta ML and Neurons.  


## Teoria de Regressão Linear <a name="teoria_regre_lin"></a>
Obs: Para facilitar o exemplo lida só com regressão linear. 
> Em estatística ou econometria, regressão linear é uma equação para se estimar a condicional (valor esperado) de uma variável y, dados os valores de algumas outras variáveis x. A regressão, em geral, tem como objetivo tratar de um valor que não se consegue estimar inicialmente. A regressão linear é chamada "linear" porque se considera que a relação da resposta às variáveis é uma função linear de alguns parâmetros. 

- 1 entrada: $$ \^{y} = mx+b $$
- Multiplas entradas: $$ \^{y} =  \sum\limits_{d=1}^D w_d * x_d + b = w^T*x + b $$

Diferente da classificação, a regressão, devido a sua saída poder assumir "qualquer" valor, não necessita de uma função de ativação (ex: sigmoide)

```
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation=None)
])

model.compile(
    optimizer = tf.keras.optimizers.SGD(0.001, 0.9), #escolhido no lugar do adam pois esse performou melhor nesse caso
    loss = 'mse' #como a saida não é binária (mse = mean square erro)
)
```

Em regressão o uso da acurácia não faz sentido pois ela quase sempre seria 0. No caso da regressão faz mais sentido utilizar a medida estatística $R^2$ (coeficiente de determinação)

## O Neurônio <a name="neuronio"></a>
Chamamos regressão logistica de neurônio, com um neurônio sendo a base fundamental de uma rede neural. 
$$\^{y} = \sigma( \sum\limits_{d=1}^D w_d * x_d + b) $$
Com cada $x_d$ sendo uma característica, é possível afirmar que seu peso $w_d$ diz o quão importante essa característica é para predizer a saída.  

### Como isso é relacionado com um neurônio? <a name="neuronio_1"></a>
Os neurônios têm um papel essencial na determinação do funcionamento e comportamento do corpo humano e do raciocínio. Eles são formados pelos dendritos, que são um conjunto de terminais de entrada, pelo corpo central, e pelos axônios que são longos terminais de saída.

Neurônios se comunicam através de sinapses. Sinapse é a região onde dois neurônios entram em contato e através da qual os impulsos nervosos são transmitidos entre eles. Os impulsos recebidos por um neurônio A, em um determinado momento, são processados, e atingindo um dado limiar de ação, o neurônio A dispara, produzindo uma substância neuro transmissora que flui do corpo celular para o axônio, que pode estar conectado a um dendrito de um outro neurônio B. O neuro transmissor pode diminuir ou aumentar a polaridade da membrana pós-sináptica, inibindo ou excitando a geração dos pulsos no neurônio B. Este processo depende de vários fatores, como a geometria da sinapse e o tipo de neuro transmissor.    

Tendo essas inforamações, é possível fazer algumas suposições básicas:
- A atividade de um neurônio é um processo tudo ou nada (binário).
- Um certo número fixo (>1) de entradas devem ser excitadas dentro de um período de adição latente para excitar um neurônio.
- Único atraso significativo é o atraso sináptico.
- A atividade de qualquer sinapse inibitória previne absolutamente a excitação do neurônio.
- A estrutura das interconexões não muda com o tempo.

Através dessas suposições, pode-se representar um neoronio artificial através do algoritmo de Regressão Logistica: $$ \^{y} = \sigma( \sum\limits_{d=1}^D w_d * x_d + b) $$ Com os n elementos do vetor $x$ sendo as entradas (dendritos) do neuronio, $\^{y}$ sendo o terminal de saída (axônio), os elementos do vetor $w$ descrevendo o comportamentos das sinapses, $w_d * x_d$ o efeito da sinapse e $b$ representando o bias (Tendência).

### Rede neural artificial <a name="neuronio_2"></a>
Uma rede neural artificial é composta por várias unidades de processamento (neurônios), cujo funcionamento é bastante simples. Essas unidades, geralmente são conectadas por canais de comunicação que estão associados a determinado peso. As unidades fazem operações apenas sobre seus dados locais, que são entradas recebidas pelas suas conexões. O comportamento inteligente de uma Rede Neural Artificial vem das interações entre as unidades de processamento da rede.

A maioria dos modelos de redes neurais possui alguma regra de treinamento, onde os pesos de suas conexões são ajustados de acordo com os padrões apresentados. Em outras palavras, elas aprendem através de exemplos.

Arquiteturas neurais são tipicamente organizadas em camadas, com unidades que podem estar conectadas às unidades da camada posterior.

Usualmente as camadas são classificadas em três grupos:
- Camada de Entrada: onde os padrões são apresentados à rede.
- Camadas Intermediárias ou Escondidas: Onde é feita a maior parte do processamento, através das conexões ponderadas; podem ser consideradas como extratoras de características.
- Camada de Saída: onde o resultado final é concluído e apresentado.

### Como o modelo aprende? <a name="neuronio_3"></a>
Uma rede neural aprende calculando os pesos de cada característica, em cada neurônio.
O Tensorflow ira fazer todas as diferenciações para calcular os pesos com menores custos, sendo preciso selecionar apenas a taxa de aprendizado.
Obs: Para escolher a taxa de aprendizado, normalmente, tentar potencias de 10 (0.1, 0.01, 0.001, ...)
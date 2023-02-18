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


# Ejecucion preliminar

Esta ejecucion uso la primera separacion del dataset, sin el filtro posterior de duplicados visuales casi identicos. Se conserva como registro de desarrollo. No usar sus metricas como resultado final ni mezclar sus graficos con los de la ejecucion definitiva.

La revision posterior encontro 30 imagenes duplicadas o casi identicas entre conjuntos, frente a las cuatro detectadas inicialmente por SHA256. Se regeneraron los recortes y se reinicio el entrenamiento desde ImageNet, con clasificador nuevo.

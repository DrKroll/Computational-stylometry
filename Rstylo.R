

library(stylo)

setwd("/Users/simonkroll/Computational-stylometry/R_genero/")
stylo()


setwd("/Users/simonkroll/Computational-stylometry/rollingc/")


# A continuación, llamamos al paquete stylo

library(stylo)

# Para Rolling Classify

rolling.classify(write.png.file = TRUE, classification.method = "svm", mfw=500, training.set.sampling = "normal.sampling", slice.size = 5000, slice.overlap = 4500) 
rolling.classify(write.png.file = TRUE, classification.method = "nsc", mfw=50, training.set.sampling = "normal.sampling", slice.size = 5000, slice.overlap = 4500)
rolling.classify(write.png.file = TRUE, classification.method = "delta", mfw=500)
#Este LightGBM fue construido  para destronar a quienes desde el inicio utilizaron XGBoost y  LightGBM
#mientras sus compañeros luchaban por correr un rpart

#Con los pibes NO

#limpio la memoria
rm( list=ls() )
gc()

require("data.table")
require("lightgbm")
require("caret")
setwd("/home/joaquin/Documents/Data-Mining/DMEyF/" )  #establezco la carpeta donde voy a trabajar

#cargo el dataset
#dataset  <- fread("./datasetsOri/paquete_premium_202009.csv")

dataset  <- fread("./datasets/paquete_premium_202009_ext2.csv")

#creo la clase_binaria donde en la misma bolsa estan los BAJA+1 y BAJA+2
dataset[ , clase01:= ifelse( clase_ternaria=="CONTINUA", 0, 1 ) ]

#Quito el Data Drifting de  "ccajas_transacciones"  "Master_mpagominimo"
campos_buenos  <- setdiff( colnames(dataset),
                           c("clase_ternaria", "clase01", "ccajas_transacciones", "Master_mpagominimo" ) )

#genero el formato requerido por LightGBM
dtrain  <- lgb.Dataset( data=  data.matrix(  dataset[ , campos_buenos, with=FALSE]),
                        label= dataset[ , clase01]
                      )
#Solo uso DOS hiperparametros,  max_bin  y min_data_in_leaf
#Dadme un punto de apoyo y movere el mundo, Arquimedes
modelo  <- lightgbm( data= dtrain,
                     params= list( objective= "binary",
                                   max_bin= 31,
                                   min_data_in_leaf= 3495,
                                   learning_rate= 0.0973055,
                                   max_depth = -1,
                                   min_gain_to_split = 0,
                                   num_leaves = 499
                                   ),  num_iterations = 107 )


#cargo el dataset donde aplico el modelo
#dapply  <- fread("./datasetsOri/paquete_premium_202011.csv")




tree_imp <- lgb.importance(modelo, percentage = TRUE)
lgb.plot.importance(tree_imp, top_n = 200, measure = "Frequency")


cols = tree_imp[tree_imp$Frequency>0.005,]




dtrain  <- lgb.Dataset( data=  data.matrix(  dataset[ , cols$Feature, with=FALSE]),
                        label= dataset[ , clase01]
)


#Solo uso DOS hiperparametros,  max_bin  y min_data_in_leaf
#Dadme un punto de apoyo y movere el mundo, Arquimedes
modelo  <- lightgbm( data= dtrain,
                     params= list( objective= "binary",
                                   max_bin= 31,
                                   min_data_in_leaf= 3495,
                                   learning_rate= 0.0973055,
                                   max_depth = -1,
                                   min_gain_to_split = 0,
                                   num_leaves = 499
                     ),  num_iterations = 107 )


dapply  <- fread("./datasets/paquete_premium_202011_ext2.csv")

#aplico el modelo a los datos nuevos, dapply
prediccion  <- predict( modelo,  data.matrix( dapply[  , cols$Feature, with=FALSE]))

#la probabilidad de corte ya no es 0.025,  sino que 0.031
entrega  <- as.data.table( list( "numero_de_cliente"= dapply[  , numero_de_cliente],
                                 "Predicted"= as.numeric(prediccion > 0.0436492),"Probabilidad"=as.numeric(prediccion) ) ) #genero la salida

#genero el archivo para Kaggle
fwrite( entrega, 
        file= "./kaggle/lightgbm_995_2.csv",
        sep=  "," )


import pandas as pd
import UMODL_SearchAlgorithm

def getImportantVariables_UMODL_ForMultiProcessing(arg):
    treatmentName=arg[3]
    y_name=arg[2]
    colName=arg[1]
    data=arg[0]
    VariableVsImportance={}
#     featureImportance,ValuesDistro=UMODL_SearchAlgorithm.ExecuteGreedySearchAndPostOpt(data[[colName,treatmentName,y_name]],0)
    featureImportance=UMODL_SearchAlgorithm.ExecuteGreedySearchAndPostOpt(data[[colName,treatmentName,y_name]])[0]
    VariableVsImportance[colName]=featureImportance
    return VariableVsImportance

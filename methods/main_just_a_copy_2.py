import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

#import methods.ACT_main_slideseq as ACT
# import methods.ACT_main as ACT
#import methods.FACT_main_slideseq as ACT
import methods.scatter as scatter
#import methods.FACT_main as FACT

# scatter.scatter('DLPFC')
scatter.scatter('DLPFC')
#ACT.FACT('E:\Project_Large_Datasets\ST\Slideseq')
#ACT.ACT('DLPFC')






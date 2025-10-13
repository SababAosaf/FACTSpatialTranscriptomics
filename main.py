import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import methods.ACT_main as ACT
import methods.FACT_main as FACT
import methods.scatter as scatter

# scatter.scatter('DLPFC')
FACT.FACT('DLPFC')
# ACT.ACT('DLPFC')






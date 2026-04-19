import pandas as pd
df = pd.read_parquet('dataset_V2.parquet')
cols = list(df.columns)
print('Total cols:', len(cols))
needed = [
    'QUA_CavityPressure1Max', 'QUA_CavityPressureMax', 
    'DXP_TrigClpCls', 'DXP_TrigPlst1', 'DXP_TrigClpOpn',
    'DOS_acComp1DosRate', 'DRY_HT101_acTempReturnAir',
    'ENV_AirTemperature', 'ENV_AirHumidity',
    'DXP_TrigInj1', 'DXP_TrigHld1', 'DXP_TrigCool',
    'DXP_Inj1PrsAct', 'DXP_Inj1PosAct',
    'TCE_TemperatureMainLine', 'TCN_TemperatureMainLine',
    'QUA_InjectionPressureMax', 'QUA_CycleTime',
    'MET_MaterialName', 'LBL_NOK', 'LBL_SinkMarks', 'LBL_Underfilled',
    'DXP_HoldingPressure1',
]
for c in needed:
    status = "FOUND" if c in cols else "MISSING"
    print(f"  {c}: {status}")

# Also print all columns that start with QUA or ENV
print("\nAll QUA_ columns:", [c for c in cols if c.startswith('QUA_')])
print("All ENV_ columns:", [c for c in cols if c.startswith('ENV_')])
print("All LBL_ columns:", [c for c in cols if c.startswith('LBL_')])
print("All DXP_ columns:", [c for c in cols if c.startswith('DXP_')])

# Check for NaN in key columns
print("\nNaN counts:")
for c in ['ENV_AirTemperature', 'QUA_CavityPressure1Max', 'QUA_CavityPressureMax', 'DOS_acComp1DosRate']:
    if c in cols:
        if df[c].dtype == 'object':
            nulls = df[c].isna().sum()
        else:
            nulls = df[c].isna().sum()
        print(f"  {c}: {nulls} NaN out of {len(df)}")

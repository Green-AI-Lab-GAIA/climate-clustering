#%%
import pandas as pd 

def read_enso_data(path = '../data/oni_index.xlsx'):
    # https://www.climate.gov/news-features/understanding-climate/climate-variability-oceanic-nino-index
    oni_index = pd.read_excel(path)
    oni_index.set_index('Year', inplace=True)
    oni_index.columns.name='Month'
    # c_map = dict(zip(list(range(2,13))+[1],oni_index.columns))
    oni_index.columns = list(range(2,13))+[1]
    oni_index= oni_index.unstack().to_frame('ONI')
    oni_index['date_period'] = pd.to_datetime(oni_index.index.map(lambda x: f"{x[1]}-{x[0]}-01")).to_period('M')
    oni_index = oni_index.set_index('date_period').sort_index()

    # el_nino_cond = (oni_index['ONI'] > 0.5)& (oni_index['ONI'].shift(1) > 0.5)& (oni_index['ONI'].shift(2) > 0.5) \
    #                 & (oni_index['ONI'].shift(3) > 0.5)& (oni_index['ONI'].shift(4) > 0.5)

    # la_nina_cond = (oni_index['ONI'] < -0.5)& (oni_index['ONI'].shift(1) < -0.5)& (oni_index['ONI'].shift(2) < -0.5) \
    #                 & (oni_index['ONI'].shift(3) < -0.5)& (oni_index['ONI'].shift(4) < -0.5)


    el_mask = oni_index['ONI'] > 0.5
    la_mask = oni_index['ONI'] < -0.5

    el_runs = el_mask.rolling(5).sum() >= 5
    la_runs = la_mask.rolling(5).sum() >= 5

    oni_index['Label'] = 'Neutro'
    oni_index['label_color'] = 0

    for idx in oni_index.index[el_runs]:
        oni_index.loc[idx-4:idx, 'Label'] = 'El Niño'
        oni_index.loc[idx-4:idx, 'label_color'] = 1

    for idx in oni_index.index[la_runs]:
        oni_index.loc[idx-4:idx, 'Label'] = 'La Niña'
        oni_index.loc[idx-4:idx, 'label_color'] = -1
        
    return oni_index.dropna()

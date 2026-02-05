import pandas as pd 

def read_enso_data():
    # https://www.climate.gov/news-features/understanding-climate/climate-variability-oceanic-nino-index
    oni_index = pd.read_excel('../data/oni_index.xlsx')
    oni_index.set_index('Year', inplace=True)
    oni_index.columns.name='Month'
    oni_index.columns = list(range(2,13))+[1]
    oni_index= oni_index.unstack().to_frame('ONI')
    oni_index['date_period'] = pd.to_datetime(oni_index.index.map(lambda x: f"{x[1]}-{x[0]}-01")).to_period('M')
    oni_index = oni_index.set_index('date_period').sort_index()

    el_nino_cond = (oni_index['ONI'] > 0.5)& (oni_index['ONI'].shift(1) > 0.5)& (oni_index['ONI'].shift(2) > 0.5) \
                    & (oni_index['ONI'].shift(3) > 0.5)& (oni_index['ONI'].shift(4) > 0.5)

    la_nina_cond = (oni_index['ONI'] < -0.5)& (oni_index['ONI'].shift(1) < -0.5)& (oni_index['ONI'].shift(2) < -0.5) \
                    & (oni_index['ONI'].shift(3) < -0.5)& (oni_index['ONI'].shift(4) < -0.5)

    oni_index.loc[el_nino_cond,'Label'] = 'El Niño'
    oni_index.loc[la_nina_cond,'Label'] = 'La Niña'
    oni_index.loc[oni_index['Label'].isna(), 'Label'] = 'Neutro'

    oni_index.loc[el_nino_cond,'label_color'] =1
    oni_index.loc[la_nina_cond,'label_color'] = -1
    oni_index.loc[oni_index['label_color'].isna(), 'label_color'] = 0
    return oni_index

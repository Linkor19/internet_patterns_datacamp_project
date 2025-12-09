import pandas as pd
import numpy as np
import sqlalchemy as sql
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_rows', 500)

patterns_df = pd.read_csv('data/internet_usage.csv')
patterns_df = patterns_df.replace("..", np.nan)

print(patterns_df.head(10))


#################################################################### create databade for sql tools
# engine = sql.create_engine("postgresql://postgres:*********************************")
# metadata = sql.MetaData()
# metadata.reflect(bind=engine)
# if "patterns" not in metadata.tables:
#     patterns = sql.Table(
#         "patterns",
#         metadata,
#         sql.Column("Country Name", sql.String),
#         sql.Column("Country Code", sql.String),
#         sql.Column("name", sql.String, nullable=False),
#         sql.Column("2000", sql.Float),
#         sql.Column("2001", sql.Float),
#         sql.Column("2002", sql.Float),
#         sql.Column("2003", sql.Float),
#         sql.Column("2004", sql.Float),
#         sql.Column("2005", sql.Float),
#         sql.Column("2006", sql.Float),
#         sql.Column("2007", sql.Float),
#         sql.Column("2008", sql.Float),
#         sql.Column("2009", sql.Float),
#         sql.Column("2010", sql.Float),
#         sql.Column("2011", sql.Float),
#         sql.Column("2012", sql.Float),
#         sql.Column("2013", sql.Float),
#         sql.Column("2014", sql.Float),
#         sql.Column("2015", sql.Float),
#         sql.Column("2016", sql.Float),
#         sql.Column("2017", sql.Float),
#         sql.Column("2018", sql.Float),
#         sql.Column("2019", sql.Float),
#         sql.Column("2020", sql.Float),
#         sql.Column("2021", sql.Float),
#         sql.Column("2022", sql.Float),
#         sql.Column("2022", sql.Float)
#     )
#     metadata.create_all(engine)
# else:
#     patterns = metadata.tables["patterns"]
#
# with engine.begin() as conn:
#     conn.execute(
#         sql.insert(patterns),
#         patterns_df.to_dict(orient="records")
#     )
#     print("patterns вставлены")

############################################################### add useful data from world bank group

country_reg = pd.read_excel('data/country_regions.xlsx')
# print(country_reg)

patterns_df = patterns_df.merge(country_reg, how = 'left', left_on = 'Country Name', right_on = 'Country')
patterns_df = patterns_df.drop('Country', axis=1)
patterns_df = patterns_df.iloc[:,[0,1,26,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]
#print(patterns_df)


edu_stat = pd.read_csv('data/edu.csv')
edu_stat = edu_stat.iloc[:816,[0,1,2,4,5,6,7,8]]
# print(edu_stat.tail(10))

edu_stat = edu_stat.melt(id_vars = ['Country Name', 'Country Code','Series'])
edu_stat['variable'] = edu_stat['variable'].astype(str).str[:4]
edu_stat['variable'] = pd.to_numeric(edu_stat["variable"])
edu_stat = edu_stat.rename(columns = {'variable':'year'})
edu_stat = edu_stat.pivot(index=['Country Name','Country Code','year'],columns='Series', values='value')
# print(edu_stat)

gpd_stat = pd.read_csv('data/GDP.csv')
# print(gpd_stat.tail(10))
gpd_stat = gpd_stat.iloc[:798,[0,1,2,4,5,6,7,8]]

gpd_stat = gpd_stat.melt(id_vars = ['Country Name', 'Country Code','Series Name'])
gpd_stat['variable'] = gpd_stat['variable'].astype(str).str[:4]
gpd_stat['variable'] = pd.to_numeric(gpd_stat["variable"])
gpd_stat = gpd_stat.rename(columns = {'variable':'year'})
gpd_stat = gpd_stat.pivot(index=['Country Name','Country Code','year'],columns='Series Name', values='value')
# print(gpd_stat)

########################################################### EDA & Analisys

patterns_melted_df = patterns_df.melt(id_vars = ['Country Name', 'Country Code','Region'])
patterns_melted_df['value'] = patterns_melted_df['value'].astype(float)
patterns_melted_df = patterns_melted_df.rename(columns = {'variable':'year'})
patterns_melted_df.pivot_table(index='year',columns='Region',values='value',aggfunc='mean').plot(kind='line')
plt.show()



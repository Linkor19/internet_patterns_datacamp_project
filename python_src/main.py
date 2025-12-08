import pandas as pd
import numpy as np
import sqlalchemy as sql

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

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


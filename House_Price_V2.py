from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate,RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from helpers.data_prep import *
from helpers.eda import *

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
# pd.set_option('display.max_rows', None)

######################################
# Exploratory Data Analysis
######################################

train = pd.read_csv("datasets/house_prices/train.csv")
test = pd.read_csv("datasets/house_prices/test.csv")
df = train.append(test).reset_index(drop=True)
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##################
# Kategorik Değişken Analizi
##################
#
# for col in cat_cols:
#     cat_summary(df, col)
#
# for col in cat_but_car:
#     cat_summary(df, col)

##################
# Sayısal Değişken Analizi
##################

df[num_cols].describe().T

# for col in num_cols:
#     num_summary(df, col, plot=True)


##################
# Target Analizi
##################

df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])


# bağımlı değişkene göre korelasyonları hesaplayacak yüzde 60 da nbüyük ve küçük
def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df, num_cols)
# High corr
# ['OverallQual: 0.7909816005838047',       Evin malzeme kalitesi 1 2 3 4 5 6 78 9 10
#  'TotalBsmtSF: 0.6135805515591944',       Bodrum alanının toplam metrekaresi
#  '1stFlrSF: 0.6058521846919166',          1. kaç metrekaresi
#  'GrLivArea: 0.7086244776126511',         Zemin üstü yaşam alanı
#  'GarageArea: 0.6234314389183598']        Garajın metrekaresi

######################################
# Data Preprocessing & Feature Engineering
######################################
# Lot Frontage
df["LOTFRONTAGE_RATIO"] = df["LotFrontage"] / df["LotArea"] * 100
df["LOTFRONTAGE_RATIO"].fillna(0, inplace=True)
df.head()

# Building Age
from datetime import date

todays_date = date.today()
todays_date.year

df["BUILDING_AGE"] = todays_date.year - df["YearBuilt"]
df["BUILDING_AGE_CAT"] = pd.qcut(df["BUILDING_AGE"], 4, labels=["New_house", "Middle_aged", "Middle_Old", "Old"])
df["Sold_Diff"] = df["YrSold"] - df["YearBuilt"]
df["House_Demand"] = pd.qcut(df["Sold_Diff"], 4, labels=["High_Demand", "Normal_Demand", "Less_Demand", "Least_Demand"])
df["BUILDING_AGE"].describe().T

df["Garage_Age"] = df["GarageYrBlt"] - df["YearBuilt"]

df["GARAGE_YEAR_DIFF"] = df["GarageYrBlt"] - df["YearBuilt"]

# df.groupby("GARAGE_YEAR_DIFF").agg("sum")
# df["GARAGE_YEAR_DIFF"]=df[df["GARAGE_YEAR_DIFF"] >0]
# df["GARAGE_YEAR_DIFF"].count()

# First floor ratio
# df[["GrLivArea","1stFlrSF","2ndFlrSF"]].head(20)
df["FIRST_FLOOR_RATIO"] = df["1stFlrSF"] / df["GrLivArea"] * 100
# df.loc[df[df["FIRST_FLOOR_RATIO"] == 100], "ONE_STORY_BUILDING"]


# Basement Ratio
df[["TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2"]].head(10)
df[df["BsmtFinSF2"] != 0][["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"]].head(10)

# Uncomplete ratio:
df["UNCOMP_BSMT_RATIO"] = df["BsmtUnfSF"] / df["TotalBsmtSF"] * 100

# Total bath
df["TOTAL_BATH"] = (df["BsmtHalfBath"] + df["HalfBath"]) * 0.5 + df["BsmtFullBath"] + df["FullBath"]
df["TOTAL_FULL_BATH"] = df["FullBath"] + df["BsmtFullBath"]
df["TOTAL_HALF_BATH"] = df["HalfBath"] + df["BsmtHalfBath"] * 0.5

# Other Rooms
df["NUMBER_OF_OTHER_ROOM"] = df["TotRmsAbvGrd"] - df["KitchenAbvGr"] - df["BedroomAbvGr"]

# Average Room Area
df["AVERAGE_ROOM_AREA"] = df["GrLivArea"] / (df["TotRmsAbvGrd"] + df["TOTAL_BATH"])

# Total porch area
df["TOTAL_PORCH_AREA"] = df["WoodDeckSF"] + df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]

# Garage ratio
df["GARAGE_RATIO"] = df["GarageArea"] / df["LotArea"] * 100

# Garage Area Per car
df["GARAGE_AREA_PER_CAR"] = df["GarageArea"] / df["GarageCars"]

# Total Garden Area:
df["GARDEN_AREA"] = df["LotArea"] - df["GarageArea"] - df["TOTAL_PORCH_AREA"] - df["TotalBsmtSF"]
df["GARDEN_RATIO"] = df["GARDEN_AREA"] / df["LotArea"] * 100
df[["GARDEN_RATIO", "GARDEN_AREA", "LotArea"]].head(30)

df.groupby(["MSSubClass", "BUILDING_AGE_CAT"]).agg({"SalePrice": ["mean", "median", "count"]})

colon = [col for col in df.columns if col in ["MSSubClass", "BUILDING_AGE_CAT"]]
df["MSSubClassXBUILDING_AGE_CAT"] = ["_".join(map(str, i)) for i in df[colon].values]

df["MSSubClassXBUILDING_AGE_CAT"].head(20)
df.groupby("MSSubClassXBUILDING_AGE_CAT").agg({"SalePrice": ["mean", "median", "count"]})
df.head()
df[["MSSubClass", "BUILDING_AGE_CAT", "MSSubClassXBUILDING_AGE_CAT"]].head(20)

colon = [col for col in df.columns if col in ["MSSubClass", "MSZoning"]]
df["MSSubClassXMSZoning"] = ["_".join(map(str, i)) for i in df[colon].values]
df["MSSubClassXMSZoning"].head(20)
df.groupby("MSSubClassXMSZoning").agg({"SalePrice": ["mean", "median", "count"]})

# def colon_bros(dataframe, col1, col2):
#     colon = [col for col in dataframe.columns if col in [col1, col2]]
#     dataframe[col1 + "_" + col2] = ["_".join(map(str, i)) for i in dataframe[colon].values]
#     print(dataframe[col1 + "_" + col2].head(15))
#
#
# colon_bros(df, "LotConfig", "LandSlope")
# colon_bros(df, "Neighborhood", "LandSlope")
df["LotArea_Cat"] = pd.qcut(df["LotArea"],4,["Small","Medium","Big","Huge"])
df["YearRemodAdd"] = pd.qcut(df["YearRemodAdd"],4,["early","Medium","late","too_late"])
liste = [
    ["Neighborhood", "HouseStyle"],
    ["HouseStyle", "OverallQual"],
    ["HouseStyle", "OverallCond"],
    ["HouseStyle", "YearRemodAdd"],
    ["HouseStyle", "RoofStyle"],
    ["HouseStyle", "Exterior1st"],
    ["HouseStyle", "MasVnrType"],
    ["SaleType", "SaleCondition", "HouseStyle"],
    ["SaleType", "HouseStyle", "MSSubClass"],
    ["LotConfig", "LotShape"],
    ["LotConfig", "Neighborhood"],
    ["LotArea_Cat", "Neighborhood"],
    ["LandContour", "Neighborhood"]
]

# def group_by_feature(dataframe, liste,target):
#     for i in liste:
#         df.groupby(i).agg({target: ["mean", "median", "count"]})
#
# group_by_feature(df,liste,"SalePrice")

df[["SaleType", "HouseStyle", "MSSubClass"]].head(20)


def colon_bros(dataframe, liste):
    """

    Parameters
    ----------
    dataframe: dataframe
    liste: list of features

    Returns
    -------

    """
    for row in liste:
        colon = [col for col in dataframe.columns if col in row]
        dataframe["_".join(map(str, row))] = ["_".join(map(str, i)) for i in dataframe[colon].values]
        print(dataframe["_".join(map(str, row))].head(15))


colon_bros(df, liste)

# df[['LotArea_Cat', 'Neighborhood', 'LotArea_Cat_Neighborhood']]


# df["LotArea_Cat"] = pd.qcut(df["LotArea"], 4, ["Small", "Medium", "Big", "Huge"])
df.head()

df.columns = [col.upper() for col in df.columns]

##################
# Rare Encoding
##################
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = cat_cols + cat_but_car
rare_analyser(df, "SalePrice", cat_cols)

df = rare_encoder(df, 0.01)

rare_analyser(df, "SalePrice", cat_cols)

useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.02).any(axis=None))]

cat_cols = [col for col in cat_cols if col not in useless_cols]

for col in useless_cols:
    df.drop(col, axis=1, inplace=True)

rare_analyser(df, "SalePrice".upper(), cat_cols)

##################
# Label Encoding & One-Hot Encoding
##################

cat_cols = cat_cols + cat_but_car

df = one_hot_encoder(df, cat_cols, drop_first=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "SalePrice", cat_cols)

useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.01).any(axis=None)]

df[useless_cols_new].head()

for col in useless_cols_new:
    cat_summary(df, col)

rare_analyser(df, "SalePrice".upper(), useless_cols_new)

##################
# Missing Values
##################

missing_values_table(df)

test.shape

missing_values_table(train)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and "SalePrice".upper() not in col]

df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.median()), axis=0)

##################
# Outliers
##################

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.01, q3=0.99))

for col in num_cols:
    replace_with_thresholds(df, col)

######################################
# Modeling
######################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

# y = train_df["SalePrice"]
y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

##################
# Base Models
##################

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

##################
# Hyperparameter Optimization
##################
#randomcv RF
rf_model = RandomForestRegressor(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 20, 10),
                    "max_features": np.random.randint(2, 50, 20),
                    "min_samples_split": np.random.randint(2, 20, 10),
                    "n_estimators": [int(x) for x in np.linspace(start=100, stop=1500, num=40)],
                    "min_samples_leaf" : np.random.randint(2, 50, 20),
                    "min_weight_fraction_leaf" : [0.01,0.1,0.2,0.3,0.02,0.5],
                    "min_impurity_decrease":[0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                    "max_samples":[0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9]}


rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

rf_random.fit(X, y)

# En iyi hiperparametre değerleri:
rf_random.best_params_

# En iyi skor
rf_random.best_score_


#randomcv LGBM
lgb_model = LGBMRegressor(random_state=17)

lgb_random_params = {"num_leaves" : np.random.randint(2, 10, 5),
                     "max_depth": np.random.randint(2, 20, 10),
                     "n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=50)],
                     "min_child_samples": np.random.randint(5, 20, 10),
                     "reg_alpha": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "reg_lambda": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "learning_rate": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9,1,3,5,7],
                     "colsample_bytree": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                     "min_child_weight" : [0.001,0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9]}


lgb_random = RandomizedSearchCV(estimator=lgb_model,param_distributions=lgb_random_params,
                                n_iter=100,  # denenecek parametre sayısı
                                cv=3,
                                verbose=True,
                                random_state=42,
                                n_jobs=-1)


lgb_random.fit(X, y)

# En iyi hiperparametre değerleri:
lgb_random.best_params_

# En iyi skor
lgb_random.best_score_


#XGB Random Search
xgb_model = XGBRegressor(random_state=17)


xgb_random_params = {"max_depth": np.random.randint(2, 20, 20),
                     "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=20)],
                     "min_child_weight": [0.3,0.02,0.5,0.7,0.9],
                     "learning_rate": [0.02,0.5,0.7,0.9],
                     "colsample_bytree": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                     "min_child_weight" : [0.02,0.5,]}


xgb_random = RandomizedSearchCV(estimator=xgb_model,param_distributions=xgb_random_params,
                                n_iter=100,  # denenecek parametre sayısı
                                cv=3,
                                verbose=True,
                                random_state=42,
                                n_jobs=-1,)


xgb_random.fit(X, y)


# En iyi hiperparametre değerleri:
xgb_random.best_params_

# En iyi skor
xgb_random.best_score_


#testing
regressors = [#("CART", DecisionTreeRegressor(), cart_params),
    ("RF", RandomForestRegressor(),rf_random.best_params_),
    ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgb_random.best_params_),
    ('LightGBM', LGBMRegressor(), lgb_random.best_params_)]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    #gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**params)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {lgb_random.best_params_}", end="\n\n")

    best_models[name] = final_model



lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

# normal y cv süresi: 16.2s
# scale edilmiş y ile: 13.8s


final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

# normal y: 27646
# scaled y: 0.1279




submission_df = pd.DataFrame()



submission_df['Id'] = test_df["Id"]



y_pred_sub = final_model.predict(test_df[selected_cols])




y_pred_sub = np.expm1(y_pred_sub)



submission_df['SalePrice'] = y_pred_sub



submission_df.to_csv('submission.csv', index=False)





